"""Numerical port regressions and differentiation checks for the Torch backend.

Reference samples were generated with vocal-tts/tlm.py and glottal.py using
the deterministic noise below. Tests do not require the source repository.
"""

import unittest
from unittest.mock import patch

import torch

from difftlm.torch.glottis import lf_pulse
from difftlm.torch.perlin import generate_fractal_noise_1d
from difftlm.torch import (
    TransmissionOptions,
    TransmissionLineSimulator,
    TransmissionLineSimulatorVowel,
)


def fixed_noise(shape, *args, device=None, dtype=None, **kwargs):
    return torch.sin(torch.arange(shape, device=device, dtype=dtype) * 0.37) * 0.1


class TorchSimulatorTests(unittest.TestCase):
    def test_original_plain_simulator_samples(self):
        sim = TransmissionLineSimulatorVowel(
            TransmissionOptions(SR=16000, TF=0.02, N=4, fft_size=0.004)
        )
        areas = torch.tensor([2.4, 1.7, 0.4, 0.9, 3.1])
        uin = torch.sin(torch.arange(320) * 0.3) * 0.2
        expected_frame = torch.tensor([
            -0.2462549210, -0.0184860229, -0.0539093725, -0.1764043123,
            -0.0123126134, -0.2202839106, -0.0092475899, -0.0015040785,
        ])
        torch.testing.assert_close(sim.step_frame(uin[:64], areas)[::8], expected_frame)
        expected_audio = torch.tensor([
            0.0, -0.2644174993, -0.0923485607, 0.1029579267,
            0.0800252035, 0.3709400892, 0.1602378339, 0.2151440382,
        ])
        audio = sim.rollout(uin, areas.repeat(sim.n_windows, 1))
        torch.testing.assert_close(audio[::40], expected_audio)

    def test_original_fricative_simulator_samples(self):
        sim = TransmissionLineSimulator(
            TransmissionOptions(SR=16000, TF=0.02, N=4, fft_size=0.004, Re_crit=0)
        )
        areas = torch.tensor([2.4, 1.7, 0.4, 0.9, 3.1])
        uin = torch.sin(torch.arange(320) * 0.3) * 0.2
        expected_frame = torch.tensor([
            -0.2460246068, -0.0184728110, -0.0537442474, -0.1764683053,
            -0.0120456142, -0.2203582316, -0.0091162578, -0.0015435635,
        ])
        expected_audio = torch.tensor([
            0.0, -0.4566414654, -0.2137631178, 0.1011450738,
            0.3846231699, 0.5480477214, 0.5403855443, 0.2451876551,
        ])
        with patch("difftlm.torch.tract.generate_fractal_noise_1d", fixed_noise):
            frame = sim.step_frame(uin[:64], areas)
            audio = sim.rollout(uin, areas.repeat(sim.n_windows, 1))
        torch.testing.assert_close(frame[::8], expected_frame)
        torch.testing.assert_close(audio[::40], expected_audio)

    def test_single_frame_gradients(self):
        for cls in (TransmissionLineSimulatorVowel, TransmissionLineSimulator):
            with self.subTest(simulator=cls.__name__):
                sim = cls(TransmissionOptions(SR=16000, N=4, fft_size=0.004, Re_crit=0))
                uin = (torch.sin(torch.arange(33, dtype=torch.float64) * 0.3) * 0.2)
                uin.requires_grad_()
                areas = torch.tensor(
                    [2.4, 1.7, 0.4, 0.9, 3.1], dtype=torch.float64,
                    requires_grad=True,
                )
                with patch("difftlm.torch.tract.generate_fractal_noise_1d", fixed_noise):
                    self.assertTrue(torch.autograd.gradcheck(
                        lambda x, a: sim.step_frame(x, a),
                        (uin, areas), fast_mode=True,
                    ))

    def test_pulse_to_audio_gradients_and_lengths(self):
        for cls in (TransmissionLineSimulatorVowel, TransmissionLineSimulator):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(simulator=cls.__name__, dtype=dtype):
                    sim = cls(TransmissionOptions(
                        SR=16000, TF=0.02, N=4, fft_size=65 / 16000
                    ))
                    pitch = torch.tensor([110., 140.], dtype=dtype, requires_grad=True)
                    rms = torch.tensor([0.7, 1.], dtype=dtype, requires_grad=True)
                    opening = torch.tensor([0.1, 0.2], dtype=dtype, requires_grad=True)
                    uin = lf_pulse(pitch, rms, opening, 16000, sim.NF)
                    areas = torch.full(
                        (sim.n_windows, sim.N + 1), 3., dtype=dtype, requires_grad=True
                    )
                    audio = sim.rollout(uin, areas)
                    self.assertEqual(audio.dtype, dtype)
                    self.assertEqual(audio.numel(),
                        sim.hop_length * (sim.n_windows - 1) + sim.frame_length)
                    self.assertTrue(torch.isfinite(audio).all())
                    audio.square().mean().backward()
                    for control in (pitch, rms, opening, areas):
                        self.assertTrue(torch.isfinite(control.grad).all())
                        self.assertGreater(control.grad.abs().max().item(), 0)
                    self.assertEqual(sim.step_frame(uin[:65], areas[0]).shape, (65,))

    def test_fricative_generator_reproducibility(self):
        sim = TransmissionLineSimulator(
            TransmissionOptions(SR=16000, TF=0.02, N=4, fft_size=0.004, Re_crit=0)
        )
        uin = torch.sin(torch.arange(sim.NF) * 0.3) * 0.2
        areas = torch.tensor([2.4, 1.7, 0.4, 0.9, 3.1]).repeat(sim.n_windows, 1)

        def synthesize(seed):
            return sim.rollout(
                uin, areas, generator=torch.Generator().manual_seed(seed)
            )

        torch.testing.assert_close(synthesize(7), synthesize(7), rtol=0, atol=0)
        self.assertFalse(torch.equal(synthesize(7), synthesize(8)))

    def test_re_crit_and_u_amp_ratio_scaling(self):
        uin = torch.sin(torch.arange(64) * 0.3) * 0.2
        areas = torch.tensor([2.4, 1.7, 0.4, 0.9, 3.1])
        sim_a = TransmissionLineSimulator(
            TransmissionOptions(SR=16000, TF=0.02, N=4, fft_size=0.004, Re_crit=3500.0, u_amp=400.0)
        )
        sim_b = TransmissionLineSimulator(
            TransmissionOptions(SR=16000, TF=0.02, N=4, fft_size=0.004, Re_crit=1750.0, u_amp=200.0)
        )
        with patch("difftlm.torch.tract.generate_fractal_noise_1d", fixed_noise):
            torch.testing.assert_close(sim_a.step_frame(uin, areas), sim_b.step_frame(uin, areas))

    def test_wall_options_take_effect(self):
        uin = torch.sin(torch.arange(64) * 0.3) * 0.2
        areas = torch.tensor([2.4, 1.7, 0.4, 0.9, 3.1])
        base = TransmissionLineSimulatorVowel(
            TransmissionOptions(SR=16000, TF=0.02, N=4, fft_size=0.004)
        )
        stiff = TransmissionLineSimulatorVowel(
            TransmissionOptions(SR=16000, TF=0.02, N=4, fft_size=0.004, Mw=4.0, Bw=1600.0, Kw=170000.0)
        )
        self.assertFalse(torch.allclose(base.step_frame(uin, areas), stiff.step_frame(uin, areas)))


class TorchGlottisTests(unittest.TestCase):
    def test_original_pulse_samples(self):
        expected = {
            False: [
                0.042113440546486, 0.001381058397225, -0.003757731831490,
                0.007252695624180, -0.011778486056224, 0.017271903675136,
                -0.023424934250129, 0.064003838882682,
            ],
            True: [
                2.024888092356261, -4.101129146674225e-9, -8.278339490386493e-7,
                -7.688785459682753e-6, -2.900075233202683e-6, -5.771326577237552e-6,
                -0.000513322367674, -3.139498457985632,
            ],
        }
        for derivative, samples in expected.items():
            with self.subTest(derivative=derivative):
                with patch("difftlm.torch.glottis.generate_fractal_noise_1d", fixed_noise):
                    actual = lf_pulse(
                        torch.tensor([100., 130., 90.], dtype=torch.float64),
                        torch.tensor([0.5, 0.9], dtype=torch.float64),
                        torch.tensor([0.1, 0.4], dtype=torch.float64),
                        1000, 64, get_derivative=derivative,
                    )
                torch.testing.assert_close(
                    actual[::8], torch.tensor(samples, dtype=torch.float64),
                    rtol=1e-10, atol=1e-12,
                )

    def test_control_gradients(self):
        controls = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                         for v in ([98., 131., 112.], [0.7, 1.], [0.1, 0.2]))
        with patch("difftlm.torch.glottis.generate_fractal_noise_1d", fixed_noise):
            self.assertTrue(torch.autograd.gradcheck(
                lambda f, r, o: lf_pulse(f, r, o, 1000, 33), controls, fast_mode=True
            ))

    def test_short_pulses_and_constant_controls(self):
        for n_samples in (1, 7, 8):
            pulse = lf_pulse([120.], [1.], [0.1], 16000, n_samples)
            self.assertEqual(pulse.shape, (n_samples,))
            self.assertTrue(torch.isfinite(pulse).all())


class TorchNoiseTests(unittest.TestCase):
    def test_original_fractal_noise_samples(self):
        # Feed both implementations uniform gradients from linspace(0.1, 0.9).
        expected = {
            False: [
                0., -0.185015869140625, -0.229296875, -0.216168212890625,
                -0.2, -0.194927978515625, -0.112109375, -0.039263916015625,
                0., -0.039263916015625, -0.112109375, -0.194927978515625,
                -0.2, -0.216168212890625, -0.229296875, -0.185015869140625,
            ],
            True: [
                0., -0.185015869140625, -0.229296875, -0.216168212890625,
                -0.2, -0.194927978515625, -0.112109375, -0.039263916015625,
                0., -0.016790771484375, 0.012109375, 0.080279541015625,
                0.2, 0.280816650390625, 0.329296875, 0.291070556640625,
            ],
        }

        def gradients(count, **kwargs):
            return torch.linspace(0.1, 0.9, count, device=kwargs['device'],
                                  dtype=kwargs['dtype'])

        for tileable, samples in expected.items():
            with self.subTest(tileable=tileable):
                with patch("difftlm.torch.perlin.torch.rand", gradients):
                    noise = generate_fractal_noise_1d(
                        32, 2, octaves=3, tileable=tileable, dtype=torch.float64
                    )
                torch.testing.assert_close(
                    noise[::2], torch.tensor(samples, dtype=torch.float64),
                    rtol=1e-12, atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
