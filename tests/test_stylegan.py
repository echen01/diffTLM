"""StyleGAN runtime, checkpoint loading, and optional CUDA operator checks."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from difftlm.torch import load_stylegan
from difftlm.torch.stylegan import default_checkpoint, resolve_checkpoint, save_stylegan, speaker_mean_w
from difftlm.ext.stylegan.networks_stylegan2 import Generator
from difftlm.ext.stylegan.torch_utils.ops import bias_act, upfirdn2d


def small_generator():
    return Generator(
        z_dim=8, c_dim=3, w_dim=8, img_resolution=8, img_channels=2,
        mapping_kwargs={"num_layers": 2}, channel_base=64, channel_max=16,
    )


class StyleGANTests(unittest.TestCase):
    def test_speaker_mean_matches_conditional_samples(self):
        generator = small_generator().eval().requires_grad_(False)
        c = torch.tensor([[1., 0., 0.]])
        seed = 101
        n_samples = 64
        draws = torch.randn(n_samples, generator.z_dim,
                            generator=torch.Generator().manual_seed(seed))
        expected_subject = generator.mapping(draws, c.expand(len(draws), -1)).mean(0, keepdim=True)
        rng_state = torch.get_rng_state()
        subject_w = speaker_mean_w(generator, c, seed, n_samples=n_samples)
        torch.testing.assert_close(subject_w, expected_subject)
        self.assertFalse(torch.allclose(subject_w, speaker_mean_w(generator, c.roll(1, 1), seed, n_samples=n_samples)))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))

    def test_generator_latent_gradients(self):
        generator = small_generator().eval().requires_grad_(False)
        z = torch.randn(1, 8, requires_grad=True)
        c = torch.tensor([[1., 0., 0.]])
        image = generator(z, c, noise_mode="const", force_fp32=True)
        self.assertEqual(image.shape, (1, 2, 8, 8))
        image.square().mean().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().max().item(), 0)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "snapshot.safetensors"
            original = small_generator().eval().requires_grad_(False)
            save_stylegan(original, checkpoint)
            loaded = load_stylegan(checkpoint)
        self.assertFalse(loaded.training)
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
        z, c = torch.randn(1, 8), torch.tensor([[1., 0., 0.]])
        with torch.no_grad():
            torch.testing.assert_close(
                loaded(z, c, noise_mode="const"),
                original(z, c, noise_mode="const"),
                rtol=0, atol=0,
            )

    def test_resolve_checkpoint_downloads_from_hf_when_local_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.safetensors"
            cached = Path(directory) / "cached.safetensors"
            with patch("difftlm.torch.stylegan.default_checkpoint", return_value=missing), \
                 patch("difftlm.torch.stylegan.hf_hub_download", return_value=str(cached)) as download:
                self.assertEqual(resolve_checkpoint(), cached)
                download.assert_called_once_with(
                    repo_id="emc348/usc-span-stylegan2",
                    filename="usc-span-stylegan2.safetensors",
                )
                download.reset_mock()
                self.assertEqual(resolve_checkpoint(missing), missing)
                download.assert_not_called()

    @unittest.skipUnless(default_checkpoint().is_file(), "MRI checkpoint is not installed")
    def test_mri_checkpoint_inference(self):
        modules_before = set(sys.modules)
        generator = load_stylegan()
        self.assertEqual((generator.z_dim, generator.c_dim), (512, 75))
        with torch.no_grad():
            z = torch.randn(1, generator.z_dim)
            c = torch.zeros(1, generator.c_dim)
            c[:, 0] = 1
            image = generator(z, c, noise_mode="const", force_fp32=True)
        self.assertEqual(image.shape, (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(image).all())
        new_modules = set(sys.modules) - modules_before
        self.assertFalse(any(name.split('.')[0] in {'dnnlib', 'torch_utils'}
                             for name in new_modules))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
    def test_cuda_operators_match_reference(self):
        x = torch.randn(2, 3, 8, 8, device="cuda", requires_grad=True)
        bias = torch.randn(3, device="cuda", requires_grad=True)
        kernel = upfirdn2d.setup_filter([1, 3, 3, 1], device="cuda")
        operations = [
            (lambda impl: bias_act.bias_act(x, bias, act="lrelu", impl=impl), (x, bias)),
            (lambda impl: upfirdn2d.upfirdn2d(x, kernel, up=2, padding=1, impl=impl), (x,)),
        ]
        for operation, inputs in operations:
            actual, expected = operation("cuda"), operation("ref")
            torch.testing.assert_close(actual, expected, rtol=5e-5, atol=1e-5)
            actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
            expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-5, atol=1e-5)
        self.assertIsNotNone(bias_act._plugin)
        self.assertIsNotNone(upfirdn2d._plugin)


if __name__ == "__main__":
    unittest.main()
