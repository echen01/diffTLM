"""PyTorch port of the one-dimensional Perlin noise helpers in vocal-tts."""

import numpy as np
import torch


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_1d(
    shape,
    res,
    tileable=False,
    interpolant=interpolant,
    *,
    device=None,
    dtype=None,
    generator=None,
):
    """Sample gradient noise with a Torch generator or the MRI fit's NumPy RNG."""
    if shape < 1 or res < 1:
        raise ValueError("shape and res must be positive integers")
    dtype = dtype or torch.get_default_dtype()
    x = torch.arange(shape, device=device, dtype=dtype) * (res / shape)
    if isinstance(generator, np.random.RandomState):
        gradients = torch.as_tensor(2 * generator.random_sample(res + 1) - 1,
                                    device=device, dtype=dtype)
    else:
        gradients = 2 * torch.rand(res + 1, device=device, dtype=dtype, generator=generator) - 1
    cell = x.floor().long()
    t = x - cell
    if tileable:
        left, right = cell % res, (cell + 1) % res
    else:
        left, right = cell.clamp(0, res), (cell + 1).clamp(0, res)
    d0 = gradients[left] * t
    d1 = gradients[right] * (t - 1)
    return d0 + interpolant(t) * (d1 - d0)


def generate_fractal_noise_1d(
    shape,
    res,
    octaves=1,
    persistence=0.5,
    lacunarity=2,
    tileable=False,
    interpolant=interpolant,
    *,
    device=None,
    dtype=None,
    generator=None,
):
    """Sum octaves of Perlin noise, following vocal-tts' sampling convention."""
    if octaves < 1:
        raise ValueError("octaves must be positive")
    noise = torch.zeros(shape, device=device, dtype=dtype)
    for octave in range(octaves):
        noise = noise + persistence**octave * generate_perlin_noise_1d(
            shape,
            res * lacunarity**octave,
            tileable,
            interpolant,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    return noise
