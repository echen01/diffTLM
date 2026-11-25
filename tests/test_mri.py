"""MRI ray extraction and acoustic latent-fitting regressions."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from difftlm.torch.mri import mri_to_area
from difftlm.torch.mri_video import extract_video_areas
from difftlm.torch.reconstruct import (
    ReconstructOptions, fit_mri, load_audio, synthesize_areas, save_reconstruction,
)
from difftlm.torch.tract import TransmissionLineSimulator, TransmissionOptions


class SmallMRIPrior(torch.nn.Module):
    z_dim, c_dim, img_resolution = 2, 3, 64

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def mapping(self, z, c):
        return (z + c[:, :2])[:, None].repeat(1, 2, 1)

    def synthesis(self, ws, **kwargs):
        gray = 0.035 + 0.02 * (ws.mean(dim=(1, 2)) + self.anchor).tanh()
        # Provide visible walls so the synthetic prior has a measurable airway.
        y, x = torch.arange(64, device=ws.device).view(-1, 1), torch.arange(64, device=ws.device)
        radius = ((x - 31.5).square() + (y - 39.5).square()).sqrt()
        airway = torch.where(x >= 32, (radius - 14).abs() < 2, (y - 25).abs() < 2)
        image = torch.where(airway, gray[:, None, None], 0.45)
        return (image * 2 - 1)[:, None].expand(-1, 3, -1, -1)


class MRITests(unittest.TestCase):
    def test_long_recording_starts_after_leading_silence(self):
        import soundfile as sf

        sr = 16000
        tone = np.cos(np.arange(4 * sr) * .03).astype(np.float32)
        signal = np.concatenate((np.zeros(sr // 2, dtype=np.float32), tone))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            sf.write(path, signal, sr, subtype="FLOAT")
            excerpt, start = load_audio(path)
        self.assertGreater(start, 0.)
        self.assertLessEqual(start, .5)
        offset = round(start * sr)
        np.testing.assert_allclose(excerpt, signal[offset:offset + 3 * sr])

    def test_short_recording_and_explicit_start(self):
        import soundfile as sf

        signal = np.cos(np.arange(4000) * .03).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            sf.write(path, signal, 16000, subtype="FLOAT")
            full, start = load_audio(path, duration=None)
            self.assertEqual(start, 0.)
            np.testing.assert_allclose(full, signal)
            excerpt, start = load_audio(path, start=.05, duration=.1)
            self.assertEqual(start, .05)
            expected = signal[800:2400].astype(np.float64)
            np.testing.assert_allclose(excerpt, expected / np.max(np.abs(expected)))

    def test_vocal_tts_mask_before_and_after_ray_interpolation(self):
        # The endpoints fix both the half-pixel convention and the 16-pixel lip span.
        rng = torch.Generator().manual_seed(101)
        image = (torch.rand(2, 64, 64, generator=rng) * .4).requires_grad_()
        distance = torch.linspace(6, 24, 100)
        horizontal = torch.stack([32 + distance, torch.full_like(distance, 40)], -1)
        lip = torch.stack([torch.full_like(distance, 16), 40 - distance], -1)
        grid = (torch.stack([horizontal, lip]) - 32) / 32
        mask = torch.sigmoid(((1 - image[:, None]) - .88) * 30)
        density = F.grid_sample(mask, grid.expand(2, -1, -1, -1), align_corners=False)[:, 0]
        expected = torch.sigmoid((density - .8) * 30).sum(-1) * .1
        areas = mri_to_area(image)
        actual = areas[:, [0, -1]]
        torch.testing.assert_close(actual, expected)
        overlay = extract_video_areas(image, area_scale=.5)
        torch.testing.assert_close(torch.from_numpy(overlay["areas"]), areas.detach().clamp(.01, 16) * .5)
        expected_grad, = torch.autograd.grad(expected.sum(), image)
        actual_grad, = torch.autograd.grad(actual.sum(), image)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)

    def test_fit_initializes_controls_and_updates_latents(self):
        prior = SmallMRIPrior()
        sim = TransmissionLineSimulator(TransmissionOptions(SR=16000, TF=.13, N=32, fft_size=.1))
        pulse = torch.sin(torch.arange(2001) * .05) * .2
        base = dict(subject_id=1, n_latents=2, batch_size=2, learning_rate=.001)
        initial = fit_mri(pulse, pulse, prior, sim, ReconstructOptions(n_steps=0, **base),
                          loss_fn=F.mse_loss, progress=False)
        fitted = fit_mri(pulse, pulse, prior, sim, ReconstructOptions(n_steps=2, **base),
                         loss_fn=F.mse_loss, progress=False)
        self.assertFalse(prior.anchor.requires_grad)
        torch.testing.assert_close(initial.latents, fitted.initial_w.expand_as(initial.latents))
        self.assertFalse(torch.equal(initial.latents, fitted.latents))
        self.assertEqual(fitted.audio.shape, (2000,))
        self.assertEqual(len(fitted.losses), 2)
        self.assertTrue(np.isfinite(fitted.export_loss))

    def test_checkpointing_replays_synthesis_noise(self):
        class NoisyPrior(SmallMRIPrior):
            def synthesis(self, ws, **kwargs):
                return super().synthesis(ws, **kwargs) + .03 * torch.randn(len(ws), 1, 64, 64)

        prior = NoisyPrior().requires_grad_(False)
        controls = torch.randn(3, 2, 2, requires_grad=True)
        results = []
        for checkpoint in (False, True):
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(12)
                areas = synthesize_areas(prior, controls, 7, batch_size=2, checkpoint_areas=checkpoint,
                                         area_frame_stride=2)
                grad, = torch.autograd.grad(areas.square().mean(), controls)
                results.append((areas, grad, torch.get_rng_state()))
        for direct, recomputed in zip(*results):
            torch.testing.assert_close(direct, recomputed)
        self.assertGreater(results[0][1].abs().max().item(), 0)

    def test_plain_video_and_optional_fixed_ray_overlay(self):
        from contextlib import closing
        import imageio_ffmpeg

        prior = SmallMRIPrior()
        sim = TransmissionLineSimulator(TransmissionOptions(SR=16000, TF=.13, N=32, fft_size=.1))
        pulse = torch.sin(torch.arange(2001) * .05) * .2
        result = fit_mri(pulse, pulse, prior, sim, ReconstructOptions(n_steps=0),
                         loss_fn=F.mse_loss, progress=False)
        for overlay in (False, True):
            with self.subTest(overlay=overlay), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                save_reconstruction(result, prior, pulse.numpy(), {}, output, fps=24, overlay_areas=overlay)
                self.assertEqual((output / "mri_overlay.mp4").exists(), overlay)
                for path in output.glob("*.mp4"):
                    with closing(imageio_ffmpeg.read_frames(str(path))) as reader:
                        info = next(reader)
                        self.assertEqual(info["size"], (256, 256))
                        self.assertEqual(info["audio_codec"], "aac")
                        self.assertEqual(sum(1 for _ in reader), 3)
                meta = json.loads((output / "metadata.json").read_text())
                self.assertEqual(meta["fps"], 24)
                self.assertEqual(meta["overlay_areas"], overlay)
                with np.load(output / "reconstruction.npz") as archive:
                    self.assertEqual(archive["video_areas"].shape, (3, 33))
                    np.testing.assert_array_equal(archive["initial_w"], result.initial_w.numpy())


if __name__ == "__main__":
    unittest.main()
