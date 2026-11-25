"""Differentiable PyTorch port of vocal-tts' time-varying LF glottal pulse.

The scalar LF parameter solvers follow VocalTractLab's LfPulse.cpp. Pulse
samples, interpolation, and aspiration noise stay in PyTorch.
"""

from functools import lru_cache
import math

import torch
import torch.nn.functional as F

from .perlin import generate_fractal_noise_1d


@lru_cache(maxsize=32)
def get_epsilon(ta, te):
    """Find epsilon using Newton iterations."""
    c = max(1 - te, 0.001)
    ta = min(max(ta, 0.0001), c - 0.00001)
    epsilon = 1 / ta
    for _ in range(40):
        h = 1 - math.exp(-epsilon * c) - epsilon * ta
        if abs(h) < 0.00001:
            break
        epsilon -= h / (c * math.exp(-epsilon * c) - ta)
    return epsilon


@lru_cache(maxsize=32)
def get_alpha(tp, te, ta, epsilon):
    w = math.pi / tp
    sin_wte = math.sin(w * te)
    cos_wte = math.cos(w * te)
    X = (
        sin_wte * math.exp(epsilon * te)
        * (-math.exp(-epsilon) / epsilon - math.exp(-epsilon))
    ) / (epsilon * ta)
    Y = (
        sin_wte * math.exp(epsilon * te)
        * (-math.exp(-epsilon * te) / epsilon - te * math.exp(-epsilon))
    ) / (epsilon * ta)

    def compute_h(a):
        return w / (w * w + a * a) + math.exp(a * te) * (
            (a * sin_wte - w * cos_wte) / (w * w + a * a) + X - Y
        )

    left, right = 0.0, 0.0
    for _ in range(20):
        right += 1
        if compute_h(right) < 0:
            break
    if compute_h(right) >= 0:
        return 0.0
    for _ in range(20):
        h_left, h_right = compute_h(left), compute_h(right)
        alpha = left - h_left * (right - left) / (h_right - h_left)
        h = compute_h(alpha)
        if abs(h) <= 0.00001:
            break
        if h < 0:
            right = alpha
        else:
            left = alpha
    return alpha


@lru_cache(maxsize=32)
def get_B(tp, alpha):
    w = math.pi / tp
    denominator = (
        math.exp(alpha * tp)
        * (alpha * math.sin(w * tp) - w * math.cos(w * tp)) + w
    ) / (w * w + alpha * alpha)
    return 1 / denominator


def interp_uniform(values, n_samples):
    """Interpolate a nonempty 1-D control sequence, including both endpoints."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("controls must be nonempty one-dimensional tensors")
    return F.interpolate(
        values[None, None], size=n_samples, mode="linear", align_corners=True
    )[0, 0]


def flutter_f0(f0, t):
    return f0 + 0.5 * (f0 / 100) * (
        torch.sin(2 * math.pi * 12.7 * t)
        + torch.sin(2 * math.pi * 7.1 * t)
        + torch.sin(2 * math.pi * 4.7 * t)
    )


def lf_pulse(
    f0s,
    rms,
    glottis_opening,
    sr: int,
    nf: int,
    get_derivative=False,
    OQ=0.5,
    SQ=3.0,
    TL=0.02,
    *,
    flutter=True,
    generator=None,
):
    """Synthesize ``nf`` samples from pitch, amplitude, and opening controls.

    Controls are nonempty 1-D sequences, interpolated independently. The output
    follows the pitch tensor's device and floating dtype and supports gradients
    through all three controls. OQ, SQ, and TL are scalar shape parameters.
    Flutter and the voiced/noise crossfade match vocal-tts' ``lf_pulse_np``;
    set ``flutter=False`` to use the supplied pitch without modulation.
    As in that implementation, derivative mode returns the unscaled LF
    derivative without aspiration noise or amplitude modulation.
    """
    nf = int(nf)
    if nf < 1 or sr <= 0:
        raise ValueError("nf and sr must be positive")
    if not 0 < OQ < 1 or SQ <= 0:
        raise ValueError("OQ must be between zero and one, and SQ must be positive")

    f0s = torch.as_tensor(f0s)
    if not f0s.is_floating_point():
        f0s = f0s.to(torch.get_default_dtype())
    rms = torch.as_tensor(rms, device=f0s.device, dtype=f0s.dtype)
    glottis_opening = torch.as_tensor(
        glottis_opening, device=f0s.device, dtype=f0s.dtype
    )
    f0s = interp_uniform(f0s, nf)
    rms = interp_uniform(rms, nf)
    glottis_opening = interp_uniform(glottis_opening, nf)
    if flutter:
        f0s = flutter_f0(
            f0s, torch.linspace(0, 1, nf, device=f0s.device, dtype=f0s.dtype)
        )
    t = torch.cumsum(f0s.clamp(0, sr / 2) / sr, dim=0).remainder(1)

    te = OQ
    tp = te * SQ / (1 + SQ)
    ta = min(max(TL, 0.01), 1 - te)
    epsilon = get_epsilon(ta, te)
    alpha = get_alpha(tp, te, ta, epsilon)
    B = get_B(tp, alpha)
    w = math.pi / tp
    c1 = B * math.exp(alpha * te) * math.sin(w * te) / (epsilon * ta)
    c2 = math.exp(-epsilon * (1 - te))
    # Avoid large exponentials in the inactive branch of torch.where.
    return_time = (t - te).clamp(min=0)

    if get_derivative:
        s_open = B * torch.exp(alpha * t) * torch.sin(w * t)
        s_return = c1 * (torch.exp(-epsilon * return_time) - c2)
        return torch.where(t < te, s_open, s_return)

    s_open = B * (
        torch.exp(alpha * t) * (alpha * torch.sin(w * t) - w * torch.cos(w * t)) + w
    ) / (w * w + alpha * alpha)
    u1_te = B * (
        math.exp(alpha * te) * (alpha * math.sin(w * te) - w * math.cos(w * te)) + w
    ) / (w * w + alpha * alpha)
    s_return = u1_te + c1 * (
        -torch.expm1(-epsilon * return_time) / epsilon - return_time * c2
    )
    s = torch.where(t < te, s_open, s_return)

    res = max(1, nf // 8)
    noise = generate_fractal_noise_1d(
        max(nf, res * 16), res, octaves=5, persistence=0.5,
        device=s.device, dtype=s.dtype, generator=generator,
    )[:nf]
    return (s * (1 - glottis_opening) + noise * glottis_opening) * rms
