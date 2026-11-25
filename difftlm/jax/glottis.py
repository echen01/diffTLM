import math
import jax
import jax.numpy as jnp
import numpy as np
from functools import lru_cache, partial
from .perlin import generate_fractal_noise

"""
LF Pulse implementation
Reference: 
https://github.com/TUD-STKS/VocalTractLabBackend-dev/blob/main/src/VocalTractLabBackend/LfPulse.cpp
"""


@lru_cache(maxsize=32)
def get_epsilon(ta, te):
    """
    Newton iterations to find epsilon
    """
    MAX_STEPS = 40
    MIN_TA = 0.0001
    MIN_C = 0.001
    TOLERANCE = 0.00001

    c = max(1 - te, MIN_C)
    ta = min(max(ta, MIN_TA), c - TOLERANCE)

    epsilon = 1 / ta

    for _ in range(MAX_STEPS):
        h = 1 - math.exp(-epsilon * c) - epsilon * ta
        if abs(h) < TOLERANCE:
            break

        h_derivative = c * math.exp(-epsilon * c) - ta
        epsilon = epsilon - h / h_derivative

    return epsilon


@lru_cache(maxsize=32)
def get_alpha(tp, te, ta, epsilon):
    w = np.pi / tp
    w2 = w * w
    sin_wte = np.sin(w * te)
    cos_wte = np.cos(w * te)
    X = (
        sin_wte
        * np.exp(epsilon * te)
        * (-np.exp(-epsilon) / epsilon - np.exp(-epsilon))
    ) / (epsilon * ta)
    Y = (
        sin_wte
        * np.exp(epsilon * te)
        * (-np.exp(-epsilon * te) / epsilon - te * np.exp(-epsilon))
    ) / (epsilon * ta)

    a = np.zeros(2)
    h = np.zeros(2)
    new_alpha = 0
    new_h = 0

    def compute_h(a):
        a2 = a * a

        return w / (w2 + a2) + np.exp(a * te) * (
            (a * sin_wte - w * cos_wte) / (w2 + a2) + X - Y
        )

    for _ in range(20):
        a[1] += 1
        h[1] = compute_h(a[1])
        if h[1] < 0:
            break

    if h[1] >= 0:
        return 0

    for _ in range(20):
        h[0] = compute_h(a[0])
        h[1] = compute_h(a[1])
        new_alpha = a[0] - (h[0] * (a[1] - a[0])) / (h[1] - h[0])
        new_h = compute_h(new_alpha)

        if np.abs(new_h) <= 0.00001:
            break

        if new_h < 0.0:
            a[1] = new_alpha
        else:
            a[0] = new_alpha

    return new_alpha


@lru_cache(maxsize=32)
def get_B(tp, alpha):
    w = np.pi / tp
    help = (np.exp(alpha * tp) * (alpha * np.sin(w * tp) - w * np.cos(w * tp)) + w) / (
        w * w + alpha * alpha
    )
    B = 1.0 / help
    return B


def interp_uniform(values, n_samples):
    n_controls = values.shape[0]
    position = jnp.linspace(0, n_controls - 1, n_samples)
    left = jnp.minimum(jnp.floor(position).astype(jnp.int32), n_controls - 2)
    weight = position - left
    return values[left] + weight * (values[left + 1] - values[left])


@partial(
    jax.jit,
    static_argnames=("sr", "nf", "get_derivative", "OQ", "SQ", "TL"),
)
def lf_pulse(
    f0s,
    rms,
    glottis_opening,
    key,
    sr: int,
    nf: int,
    get_derivative=False,
    OQ=0.5,  # Open Quotient
    SQ=3.0,  # Shape Quotient
    TL=0.02,  # Spectral tilt
):
    nf = int(nf)
    MIN_TA = 0.01

    T0 = 1.0
    te = OQ
    tp = (te * SQ) / (1.0 + SQ)
    ta = TL

    ta = min(max(ta, MIN_TA), T0 - te)

    epsilon = get_epsilon(ta, te)
    alpha = get_alpha(tp, te, ta, epsilon)
    B = get_B(tp, alpha)

    f0s = interp_uniform(f0s, nf)
    rms = interp_uniform(rms, nf)
    glottis_opening = interp_uniform(glottis_opening, nf)

    f0s = jnp.clip(f0s, 0.0, sr / 2.0)

    phi = jnp.cumsum(f0s / sr)
    t = jnp.mod(phi, 1.0)

    w = jnp.pi / tp

    if get_derivative:
        c1 = (B * jnp.exp(alpha * te) * jnp.sin(w * te)) / (epsilon * ta)
        c2 = jnp.exp(-epsilon * (T0 - te))

        s_open = B * jnp.exp(alpha * t) * jnp.sin(w * t)
        s_return = c1 * (jnp.exp(-epsilon * (t - te)) - c2)

        s = jnp.where(t < te, s_open, s_return)

        return s

    # Eq 4. of [FLL1995]
    u1_te = (
        B * (jnp.exp(alpha * te) * (alpha * jnp.sin(w * te) - w * jnp.cos(w * te)) + w)
    ) / (w * w + alpha * alpha)
    pre_factor = (B * jnp.exp(alpha * te) * jnp.sin(w * te) * jnp.exp(epsilon * te)) / (
        epsilon * ta
    )
    F2_te = pre_factor * (
        -jnp.exp(-epsilon * te) / epsilon - te * jnp.exp(-epsilon * T0)
    )

    s_open = (
        B * (jnp.exp(alpha * t) * (alpha * jnp.sin(w * t) - w * jnp.cos(w * t)) + w)
    ) / (w * w + alpha * alpha)
    s_return = (
        u1_te
        + pre_factor * (-jnp.exp(-epsilon * t) / epsilon - t * jnp.exp(-epsilon * T0))
        - F2_te
    )

    s = jnp.where(t < te, s_open, s_return)

    #noise = jax.random.normal(key, shape=s.shape) + 1

    res = nf // (2**3)
    n_octaves = 5
    beta = 2

    noise = generate_fractal_noise(
        nf,
        res,
        n_octaves,
        2 ** (-0.5 * beta),
        lacunarity=2,
        tileable=False,
        key=key
    )[:nf] + 1

    s = s + noise * glottis_opening
    s = s * rms

    return s
