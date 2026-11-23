from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import scipy.signal
from .perlin import generate_fractal_noise


@dataclass
class SimulatorOptions:
    SR: float = 44100.0  # Sample rate (Hz)
    L: float = 17.0  # Vocal tract length (cm)
    TF: float = 1.0  # Simulation duration (s)
    N: int = 32  # Number of tube sections

    u_amp: float = 400.0  # Volume velocity amplitude (cm^3/s)

    # Physical parameters in CGS units
    rho: float = 1.14e-3  # Air density (g/cm^3)
    c: float = 3.43e4  # Speed of sound (cm/s)
    mu: float = 1.86e-4  # Dynamic viscosity of air (g/(cm*s))
    Mw: float = 2.1  # Wall mass per unit area (g/cm^2)
    Bw: float = 800.0  # Wall resistance per unit area (g/(cm^2*s))
    Kw: float = 84500.0  # Wall stiffness per unit area (g/(cm^2*s^2))
    Re_crit: float = 3500.0  # Critical Reynolds number for turbulence (dimensionless)

    # FDS Parameters
    win_size: float = 0.025  # Window length (s)
    fft_size: int = 512  # FFT size for synthesis (samples)
    hop_ratio: float = 4.0  # Hop size ratio
    a_win_alpha: float = 0.25  # Tukey window shape parameter


class TransmissionLineSimulator:
    """
    Frequency Domain Vocal Simulator.

    Models acoustic propagation and turbulent noise generation using the frequency domain transmission matrix representation.
    """

    def __init__(self, opt: SimulatorOptions = SimulatorOptions()):
        self.opt = opt
        self.N = opt.N
        self.SR = opt.SR
        self.TF = opt.TF

        self.NF = int(opt.SR * opt.TF)
        self.dt = 1.0 / opt.SR

        self.lowpass = jax.vmap(self.lowpass_filter, in_axes=(0, None, None))

        self.frame_length = int(opt.SR * opt.win_size)
        self.hop_length = self.frame_length // int(opt.hop_ratio)
        self.n_windows = (self.NF - self.frame_length) // self.hop_length + 1

        n_octaves = 5
        # Resolve every octave with at least four samples per lattice cell.
        res = self.NF // (4 * 2 ** (n_octaves - 1))
        beta = 2

        self.generate_noise = partial(
            generate_fractal_noise,
            shape=self.NF,
            res=res,
            octaves=n_octaves,
            persistence=2 ** (-0.5 * beta),
            lacunarity=2,
            tileable=False,
        )
        self.generate_noise = jax.vmap(self.generate_noise)

        self.analysis_window = jnp.asarray(
            scipy.signal.windows.tukey(self.frame_length, alpha=self.opt.a_win_alpha)
        )
        self.synthesis_window = jnp.hanning(self.frame_length)

    @staticmethod
    def lowpass_filter(x, f_c=2000.0, sr=16000.0):
        """Direct-form lowpass filter"""
        alpha = 2.0 * jnp.pi * f_c / sr

        def step(y_prev, x_n):
            y = y_prev + (x_n - y_prev) * alpha
            return y, y

        y0 = jnp.array(0.0, dtype=x.dtype)
        _, y = jax.lax.scan(step, y0, x)
        return y

    def consonant_gate(self, u_c, noise, A):
        """
        u_c: volume velocity at constriction
        noise: noise source
        A: area of constriction

        A differentiable gate for consonants
        """
        u_lp = self.lowpass(u_c, 2000.0, self.SR)
        Re2 = (
            (4.0 * self.opt.rho**2) / (jnp.pi * self.opt.mu**2) * u_lp**2 / A[..., None]
        )

        attn = jax.nn.softmax(-A, axis=0)[..., None]
        gate = jax.nn.softplus(Re2 - (self.opt.Re_crit / self.opt.u_amp) ** 2)
        u_n = 1e-4 * noise * gate * attn

        return u_n

    @partial(jax.jit, static_argnames=["self"])
    def step_frame(self, uin, A, noise, L):
        """Simulates acoustic pressure output for a single frame segment."""
        rho = self.opt.rho
        mu = self.opt.mu
        c = self.opt.c

        # Compute boundary impedance from last area
        # Eq. (2) of Birkholz & Jackel (2004)
        # -------------------------------------
        Lr = (8.0 * rho) / (3.0 * jnp.pi * jnp.sqrt(jnp.pi * A[-1]))
        Rr = (128.0 * rho * c) / (9.0 * jnp.pi**2 * A[-1])
        # -------------------------------------

        A = A[:-1]  # Remove boundary area
        circumference = jnp.sqrt(A / jnp.pi) * 2.0 * jnp.pi
        li = L / self.N
        Sa = circumference * li

        # Eq. (1) of Birkholz & Jackel (2004)
        # -------------------------------------
        Li = rho * li / (2.0 * A)
        Ri = (Sa / (2.0 * A**2)) * jnp.sqrt(rho * mu * 0.5)
        Ci = A * li / (rho * c**2)

        Lwi = self.opt.Mw / Sa
        Rwi = self.opt.Bw / Sa
        Cwi = Sa / self.opt.Kw
        # -------------------------------------

        freqs = jnp.fft.rfftfreq(len(uin), 1.0 / self.SR).clip(min=1e-8)
        omega = 2.0 * jnp.pi * freqs[..., None]
        jomega = 1j * omega

        Ri_omega = Ri * jnp.sqrt(omega)

        Za = Ri_omega + Li * jomega
        Zb1 = Rwi + Lwi * jomega + 1.0 / (jomega * Cwi)
        Zb2 = 1.0 / (jomega * Ci)
        Zb = Zb1 * Zb2 / (Zb1 + Zb2)

        Zr = Rr * (jomega.squeeze() * Lr) / (Rr + jomega.squeeze() * Lr)

        # Build section transfer matrices Ks: [F, N, 2, 2]
        Ks = jnp.stack(
            [1.0 + Za / Zb, -2.0 * Za - Za**2 / Zb, -1.0 / Zb, 1.0 + Za / Zb], axis=-1
        ).reshape(len(omega), -1, 2, 2)

        num_sections = Ks.shape[1]

        Ks_fwd = [Ks[:, 0]]
        Ks_bkwd = [Ks[:, -1]]

        # This for loop is faster than jax.lax.associative_scan, likely because N is small.
        for i in range(1, num_sections):
            Ks_fwd.append(Ks[:, i] @ Ks_fwd[-1])
            Ks_bkwd.append(Ks_bkwd[-1] @ Ks[:, num_sections - 1 - i])

        Ks_fwd = jnp.stack(Ks_fwd, axis=1)
        Ks_bkwd = jnp.stack(Ks_bkwd, axis=1)

        K_tract = Ks_fwd[:, -1]

        Z1 = -Ks_fwd[..., 0, 1] / Ks_fwd[..., 1, 1]
        Z2 = (Ks_bkwd[..., 1, 1] * Zr[:, None] - Ks_bkwd[..., 0, 1]) / (
            Ks_bkwd[..., 0, 0] - Ks_bkwd[..., 1, 0] * Zr[:, None]
        )

        # Transfer functions from section n -> N, for n in 1 ... N
        H_fric = (
            Zr[:, None]
            * Z1
            / ((Ks_bkwd[..., 0, 0] - Ks_bkwd[..., 1, 0] * Zr[:, None]) * (Z1 + Z2))
        )

        # Transfer functions from section 0 -> n, for n in 1 ... N.
        H_c = 1.0 / (Ks_fwd[..., 0, 0] - Ks_fwd[..., 1, 0] * Z2)

        H_tract = Zr / (K_tract[:, 0, 0] - K_tract[:, 1, 0] * Zr)

        H_fric, H_c = H_fric.T, H_c.T

        uin_fft = jnp.fft.rfft(uin)

        # Inject noise
        u_c = H_c * uin_fft[None, ...]
        u_c = jnp.fft.irfft(u_c)
        u_n = self.consonant_gate(u_c, noise, A)
        u_n_fft = jnp.fft.rfft(u_n)
        # ---------------------------------------------------

        pout = jnp.fft.irfft(uin_fft * H_tract + (u_n_fft * H_fric).sum(axis=0))

        return pout

    @partial(jax.jit, static_argnames=["self"])
    def rollout(self, uin, As, L, key):
        """Performs overlap-add (OLA) synthesis across frames."""

        assert As.shape[0] == self.n_windows

        frame_length = self.frame_length

        noise = self.generate_noise(key=jax.random.split(key, self.N - 1))
        noise = noise.T

        starts = self.hop_length * jnp.arange(self.n_windows)
        frame_idx = jnp.arange(frame_length)
        indices = starts[:, None] + frame_idx[None, :]

        # Window input signals
        uins = uin[indices] * self.analysis_window[None, :]
        noises = noise[indices] * self.analysis_window[None, :, None]
        noises = noises.transpose(0, 2, 1)

        # Pad to FFT size
        padding = int(self.opt.fft_size - frame_length)
        uins = jnp.pad(uins, ((0, 0), (0, padding)))
        noises = jnp.pad(noises, ((0, 0), (0, 0), (0, padding)))

        @jax.vmap
        def process_frames(uin_frame, noise_frame, A):
            pout_frame = self.step_frame(uin_frame, A, noise_frame, L)
            return pout_frame[:frame_length] * self.synthesis_window

        pout_frames = process_frames(uins, noises, As)

        # scatter-add
        pout = jnp.zeros(self.NF)
        pout = pout.at[indices].add(pout_frames)

        return pout
