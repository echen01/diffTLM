"""PyTorch transmission-line simulators ported from vocal-tts/tlm.py.

``step_frame`` simulates a single frame. ``rollout`` uses overlap-add.
Areas have shape ``(N + 1,)`` per frame, in cm^2, as in vocal-tts. This differs
from the area convention and options used by the JAX simulator in ``difftlm.jax.tract``.
"""

from dataclasses import dataclass
import math

import scipy.signal
import torch
import torch.nn.functional as F

from .perlin import generate_fractal_noise_1d


@dataclass
class TransmissionOptions:
    SR: float = 44100.0  # Sample rate (Hz)
    TF: float = 1.0  # Simulation duration (s)
    N: int = 20  # Number of spatial intervals; supply N + 1 areas
    fft_size: float = 0.025  # Analysis frame duration (s)

    u_amp: float = 400.0  # Volume velocity amplitude (cm^3/s)

    # Physical parameters in CGS units
    rho: float = 1.14e-3  # Density of air (g/cm^3)
    c: float = 3.43e4  # Speed of sound (cm/s)
    mu: float = 1.86e-4  # Dynamic viscosity of air (g/(cm*s))
    Mw: float = 2.1  # Wall mass (g/cm^2)
    Bw: float = 800.0  # Wall resistance (g/(cm^2*s))
    Kw: float = 84500.0  # Wall stiffness (g/(cm^2*s^2))
    Re_crit: float = 3500.0  # Critical Reynolds number for turbulence (dimensionless)


class TransmissionLineSimulatorVowel:
    """Vowel-only transmission-line model with differentiable tube areas."""

    _hop_ratio = 4
    _window_input = False

    def __init__(self, opt: TransmissionOptions | None = None):
        self.opt = opt if opt is not None else TransmissionOptions()
        self.N = self.opt.N

        self.calc_derived_parameters()
        self.a_window = torch.from_numpy(
            scipy.signal.windows.tukey(self.frame_length, alpha=0.25)
        )

    def calc_derived_parameters(self):
        self.SR = self.opt.SR
        self.TF = self.opt.TF

        self.NF = int(self.SR * self.TF)
        self.dt = 1 / self.SR

        self.frame_length = int(self.SR * self.opt.fft_size)
        self.hop_length = self.frame_length // self._hop_ratio

        if self.N < 1 or self.hop_length < 1:
            raise ValueError("N must be positive and the frame must span at least one hop")
        self.n_windows = max(0, (self.NF - self.frame_length) // self.hop_length + 1)

    @staticmethod
    def detect_constriction(Ai, threshold=0.2):
        return Ai < threshold

    def _transfer_matrices(self, uin, Ai, L):
        """Build the transfer matrix for each tube section and the lip impedance."""
        if uin.ndim != 1 or uin.numel() == 0 or Ai.shape != (self.N + 1,):
            raise ValueError("expected a nonempty 1-D input and N + 1 positive areas")

        rho = self.opt.rho
        mu = self.opt.mu
        c = self.opt.c

        circumference = torch.sqrt(Ai / math.pi) * 2 * math.pi
        li = L / self.N
        Sa = circumference * li  # Wall surface area of each section

        # Tube and wall elements, Eq. (1) of Birkholz & Jackel (2004)
        # -------------------------------------
        Li = rho * li / (2 * Ai)
        Ri = (Sa / (2 * Ai**2)) * (rho * mu * 0.5) ** 0.5
        Ci = Ai * li / (rho * c**2)

        Lwi = self.opt.Mw / Sa
        Rwi = self.opt.Bw / Sa
        Cwi = Sa / self.opt.Kw
        # -------------------------------------

        # Lip radiation, Eq. (2) of Birkholz & Jackel (2004)
        Lr = (8 * rho) / (3 * math.pi * torch.sqrt(math.pi * Ai[-1]))
        Rr = (128 * rho * c) / (9 * math.pi**2 * Ai[-1])

        freqs = torch.fft.rfftfreq(
            len(uin),
            1 / self.SR,
            device=uin.device,
            dtype=Ai.dtype,
        ).clamp(min=1e-8)
        omega = (2 * math.pi * freqs)[:, None]
        jomega = 1j * omega

        Ri_omega = Ri * torch.sqrt(omega)
        Za = Ri_omega + Li * jomega
        Zb1 = Rwi + Lwi * jomega + 1 / (jomega * Cwi)
        Zb2 = 1 / (jomega * Ci)
        Zb = Zb1 * Zb2 / (Zb1 + Zb2)

        Zr = Rr * (jomega[:, 0] * Lr) / (Rr + jomega[:, 0] * Lr)

        # Section transfer matrices: [frequency, section, 2, 2]
        Ks = torch.stack(
            [1 + Za / Zb, -2 * Za - Za**2 / Zb, -1 / Zb, 1 + Za / Zb], dim=-1
        ).reshape(len(freqs), self.N + 1, 2, 2)

        return Ks, Zr

    def step_frame(self, uin, Ai, L=15):
        """Return one pressure frame, with the same sample count as ``uin``."""
        Ks, Zr = self._transfer_matrices(uin, Ai, L)

        K = torch.eye(2, device=uin.device, dtype=Ks.dtype).expand(len(Ks), 2, 2)
        for i in reversed(range(self.N + 1)):
            K = K @ Ks[:, i]

        H_tract = Zr / (K[:, 0, 0] - K[:, 1, 0] * Zr)
        uin_fft = torch.fft.rfft(uin)
        pout = torch.fft.irfft(uin_fft * H_tract, n=len(uin))

        return pout

    def rollout(self, uin, Ais, *args, **kwargs):
        """Overlap-add one pressure frame per row of Ais (frames, N + 1).

        Extra arguments go directly to step_frame (``L``, and for the fricative
        model a keyword-only ``generator`` for reproducible noise shared across frames).
        """
        if uin.ndim != 1 or Ais.ndim != 2 or Ais.shape[1] != self.N + 1:
            raise ValueError("expected input (samples,) and areas (frames, N + 1)")

        frame_length = self.frame_length
        hop_length = self.hop_length
        num_frames = len(Ais)

        if num_frames < 1:
            raise ValueError("at least one area frame is required")

        out_len = hop_length * (num_frames - 1) + frame_length
        if out_len > len(uin):
            raise ValueError("input is too short for the supplied area frames")

        # The fricative model applies a Tukey window before padding for the FFT.
        uins = uin.unfold(0, frame_length, hop_length)[:num_frames]
        if self._window_input:
            analysis_window = self.a_window.to(device=uin.device, dtype=uin.dtype)
            uins = uins * analysis_window

            fft_size = 1 << (frame_length - 1).bit_length()
            uins = F.pad(uins, (0, fft_size - frame_length))

        def process_frame(uin_frame, Ai):
            return self.step_frame(uin_frame, Ai, *args, **kwargs)

        pout_frames = torch.vmap(process_frame, randomness="same")(uins, Ais)

        synthesis_window = torch.hann_window(
            frame_length, device=pout_frames.device, dtype=Ais.dtype
        )
        pout_frames = pout_frames[:, :frame_length] * synthesis_window

        pout = pout_frames.new_zeros(out_len)
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            pout[start:end] += pout_frames[i]

        return pout


class TransmissionLineSimulator(TransmissionLineSimulatorVowel):
    """Transmission-line model with differentiable turbulence injection."""

    _hop_ratio = 8
    _window_input = True

    def consonant_gate(self, u_c, noise, Ai):
        """Gate turbulence using the volume velocity and area at each constriction."""
        rho = self.opt.rho
        mu = self.opt.mu
        Re2 = 4 * rho**2 / (math.pi * mu**2) * u_c**2 / Ai[:, None]

        attn = torch.softmax(-Ai, dim=0)[:, None]
        gate = F.softplus(Re2 - (self.opt.Re_crit / self.opt.u_amp) ** 2)
        return 0.001 * noise * attn * gate

    def step_frame(self, uin, Ai, L=15, *, generator=None):
        """Simulate the pressure from glottal flow and turbulence in one frame."""
        Ks, Zr = self._transfer_matrices(uin, Ai, L)
        num_sections = Ks.shape[1]

        Ks_fwd = [Ks[:, 0]]
        Ks_bkwd = [Ks[:, -1]]

        for i in range(1, num_sections):
            Ks_fwd.append(Ks[:, i] @ Ks_fwd[-1])
            Ks_bkwd.append(Ks_bkwd[-1] @ Ks[:, num_sections - 1 - i])

        Ks_fwd = torch.stack(Ks_fwd, dim=1)
        Ks_bkwd = torch.stack(Ks_bkwd, dim=1)
        K_tract = Ks_fwd[:, -1]

        Z1 = -Ks_fwd[..., 0, 1] / Ks_fwd[..., 1, 1]
        Z2 = (Ks_bkwd[..., 1, 1] * Zr[:, None] - Ks_bkwd[..., 0, 1]) / (
            Ks_bkwd[..., 0, 0] - Ks_bkwd[..., 1, 0] * Zr[:, None]
        )

        # Transfer functions from each section to the lips
        H_fric = (
            Zr[:, None]
            * Z1
            / ((Ks_bkwd[..., 0, 0] - Ks_bkwd[..., 1, 0] * Zr[:, None]) * (Z1 + Z2))
        )

        # Transfer functions from the glottis to each section
        H_c = 1 / (Ks_fwd[..., 0, 0] - Ks_fwd[..., 1, 0] * Z2)
        H_tract = Zr / (K_tract[:, 0, 0] - K_tract[:, 1, 0] * Zr)

        H_fric, H_c = H_fric.T, H_c.T

        n_samples = len(uin)
        uin_fft = torch.fft.rfft(uin)

        # Volume velocity at each constriction
        u_c = torch.fft.irfft(H_c * uin_fft[None, :], n=n_samples)

        # Inject turbulence at the constrictions.
        n_octaves = 5
        res = max(1, self.frame_length // 8)
        noise_length = max(n_samples, res * 2 ** (n_octaves - 1))
        noise = generate_fractal_noise_1d(
            noise_length,
            res,
            octaves=n_octaves,
            persistence=0.5,
            device=uin.device,
            dtype=uin.dtype,
            generator=generator,
        )[:n_samples]

        u_n = self.consonant_gate(u_c, noise, Ai)
        u_n_fft = torch.fft.rfft(u_n)

        pout_fft = uin_fft * H_tract + (u_n_fft * H_fric).sum(dim=0)
        pout = torch.fft.irfft(pout_fft, n=n_samples)

        return pout
