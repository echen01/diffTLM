from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

"""
Reference:
https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py
"""


# Smoothstep
def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


@partial(
    jax.jit,
    static_argnames=("shape", "res", "octaves", "lacunarity", "tileable"),
)
def generate_fractal_noise(
    shape,
    res,
    octaves=1,
    persistence=0.5,
    lacunarity=2,
    tileable=False,
    key=None,
):
    """Fractal gradient noise with an independent random phase per octave.

    Use shape >= 4 * res * lacunarity**(octaves - 1) to resolve every octave.
    """
    resolutions = tuple(res * lacunarity**i for i in range(octaves))
    gradient_key, phase_key = jax.random.split(key)

    counts = tuple(r if tileable else r + 2 for r in resolutions)
    gradients = jax.random.uniform(
        gradient_key, (sum(counts),), minval=-1.0, maxval=1.0
    )
    phases = jax.random.uniform(phase_key, (octaves,))
    samples = jnp.arange(shape)
    noise = jnp.zeros(shape)
    offset = 0

    for i, (resolution, count) in enumerate(zip(resolutions, counts)):
        position = samples * (resolution / shape) + phases[i]
        cell = jnp.floor(position).astype(jnp.int32)
        t = position - cell
        if tileable:
            left = jnp.where(cell >= resolution, cell - resolution, cell)
            right = jnp.where(left + 1 >= resolution, 0, left + 1)
        else:
            left, right = cell, cell + 1

        d0 = gradients[offset + left] * t
        d1 = gradients[offset + right] * (t - 1)
        noise += persistence**i * (d0 + interpolant(t) * (d1 - d0))
        offset += count

    return noise
