"""JAX multi-resolution spectrogram loss with cacheable target features."""

import jax
import jax.numpy as jnp
from jax.scipy import signal
from typing import List, Any
import librosa.filters as filters


def spectral_convergence_loss(x_mag: jax.Array, y_mag: jax.Array) -> jax.Array:
    return jnp.linalg.norm(y_mag - x_mag) / (jnp.linalg.norm(y_mag) + 1e-8)


def stft_magnitude_loss(
    x_mag: jax.Array,
    y_mag: jax.Array,
    log=True,
    log_eps=1e-8,
    log_fac=1.0,
    distance="L1",
) -> jax.Array:
    if log:
        x_mag = jnp.log10(log_fac * x_mag + log_eps)
        y_mag = jnp.log10(log_fac * y_mag + log_eps)

    match distance:
        case "L1":
            loss = jnp.abs(y_mag - x_mag)
        case "L2":
            loss = (x_mag - y_mag) ** 2
        case _:
            raise ValueError(f"Invalid distance: '{distance}'.")

    return loss.mean()


class STFTLoss:
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "hann",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: int = None,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        mag_distance: str = "L1",
    ):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = window
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.mag_distance = mag_distance

        self.phs_used = w_phs > 0.0

        self.fb = None
        if scale is not None:
            if scale == "mel":
                fb = filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
                self.fb = jnp.array(fb)
            elif scale == "chroma":
                fb = filters.chroma(sr=sample_rate, n_fft=fft_size, n_chroma=n_bins)
                self.fb = jnp.array(fb)
            else:
                raise ValueError(
                    f"Invalid scale: '{scale}'. Must be 'mel' or 'chroma'."
                )

    def stft(self, x: jax.Array) -> tuple[jax.Array, Any]:
        """Compute STFT magnitudes, adding epsilon only when taking logs."""
        window = jnp.hanning(self.win_length) if self.window == "hann" else self.window
        _, _, x_stft = signal.stft(
            x,
            window=window,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_size,
            nfft=self.fft_size,
            boundary="zeros",
            padded=True,
        )

        x_mag = jnp.abs(x_stft)

        if self.phs_used:
            x_phs = jnp.angle(x_stft)
        else:
            x_phs = None

        return x_mag, x_phs

    def features(self, x: jax.Array):
        """Return (magnitude, log magnitude, phase) for reuse across steps.

        Log magnitudes are deferred when scale invariance is enabled.
        Unused log magnitudes and phases are None.
        """
        if x.ndim == 1:
            x = x[jnp.newaxis, :]
        elif x.ndim == 3:
            B, C, T = x.shape
            x = x.reshape(B * C, T)

        x_mag, x_phs = self.stft(x)
        if self.fb is not None:
            x_mag = jnp.matmul(self.fb, x_mag)

        x_log_mag = (
            jnp.log10(x_mag + self.eps)
            if self.w_log_mag > 0.0 and not self.scale_invariance
            else None
        )
        return x_mag, x_log_mag, x_phs

    def compare(self, x_features, y_features) -> jax.Array:
        """Compare features produced by this loss's configuration."""
        x_mag, x_log_mag, x_phs = x_features
        y_mag, y_log_mag, y_phs = y_features

        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum(axis=(-2, -1), keepdims=True) / (
                (y_mag**2).sum(axis=(-2, -1), keepdims=True) + self.eps
            )
            y_mag = y_mag * alpha
            if self.w_log_mag > 0.0:
                x_log_mag = jnp.log10(x_mag + self.eps)
                y_log_mag = jnp.log10(y_mag + self.eps)

        # Compute individual spectral loss terms
        sc_mag_loss = (
            spectral_convergence_loss(x_mag, y_mag) if self.w_sc > 0.0 else 0.0
        )
        log_mag_loss = (
            stft_magnitude_loss(
                x_log_mag, y_log_mag, log=False, distance=self.mag_distance
            )
            if self.w_log_mag > 0.0
            else 0.0
        )
        lin_mag_loss = (
            stft_magnitude_loss(x_mag, y_mag, log=False, distance=self.mag_distance)
            if self.w_lin_mag > 0.0
            else 0.0
        )
        phs_loss = jnp.mean((x_phs - y_phs) ** 2) if self.phs_used else 0.0

        loss = (
            self.w_sc * sc_mag_loss
            + self.w_log_mag * log_mag_loss
            + self.w_lin_mag * lin_mag_loss
            + self.w_phs * phs_loss
        )

        return loss.mean()

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self.compare(self.features(x), self.features(y))


class MultiResolutionSTFTLoss:
    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [120, 240, 50],
        win_lengths: List[int] = [600, 1200, 240],
        window: str = "hann",
        w_sc: float = 0.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float = None,
        scale: str = None,
        n_bins: int = None,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        mag_distance: str = "L1",
    ):
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.stft_losses = []
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(
                STFTLoss(
                    fft_size=fs,
                    hop_size=ss,
                    win_length=wl,
                    window=window,
                    w_sc=w_sc,
                    w_log_mag=w_log_mag,
                    w_lin_mag=w_lin_mag,
                    w_phs=w_phs,
                    sample_rate=sample_rate,
                    scale=scale,
                    n_bins=n_bins,
                    scale_invariance=scale_invariance,
                    eps=eps,
                    mag_distance=mag_distance,
                )
            )

    def features(self, x: jax.Array):
        """Extract one feature tuple per resolution; cache this for fixed targets."""
        return tuple(f.features(x) for f in self.stft_losses)

    def compare(self, x_features, y_features) -> jax.Array:
        """Compare features produced by this loss's configuration."""
        return sum(
            f.compare(x, y)
            for f, x, y in zip(self.stft_losses, x_features, y_features)
        ) / len(self.stft_losses)

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self.compare(self.features(x), self.features(y))
