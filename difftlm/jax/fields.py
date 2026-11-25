"""Fixed-resolution fields for time-varying vocal tract geometry."""

from flax import nnx
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


def _coordinates(N, T):
    """Normalized (time, position) coordinates for a fixed output grid."""
    if min(N, T) < 1:
        raise ValueError("Field dimensions must be positive.")
    ts, xs = jnp.meshgrid(jnp.linspace(0, 1, T), jnp.linspace(0, 1, N), indexing="ij")
    return jnp.stack([ts, xs], axis=-1)


class RFFMLP(nnx.Module):
    """Fixed Fourier features evaluated on an internal (T, N) coordinate grid.

    Scale is scalar or per-input (time, position), in cycles per unit.
    Tanh bounds hidden activations; leaky_relu reproduces the reference MLP.
    """

    def __init__(
        self,
        N,
        T,
        encoding_dim=128,
        hidden_dim=256,
        scale=16.0,
        *,
        activation="leaky_relu",
        rngs,
    ):
        if activation not in ("leaky_relu", "tanh"):
            raise ValueError(f"Unknown RFF activation: {activation!r}")
        self.activation = activation
        self.coords = nnx.Variable(_coordinates(N, T))
        self.B = nnx.Variable(
            jnp.asarray(scale)[..., None]
            * jax.random.normal(rngs.params(), (2, encoding_dim))
        )
        self.layers = nnx.List(
            [
                nnx.Linear(2 * encoding_dim, hidden_dim, rngs=rngs),
                nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
                nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            ]
        )
        self.output_linear = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self):
        phase = 2 * jnp.pi * (self.coords[...] @ self.B[...])
        x = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
        activate = jnp.tanh if self.activation == "tanh" else jax.nn.leaky_relu
        for layer in self.layers:
            x = activate(layer(x))
        return self.output_linear(x)[..., 0]


def resize_multiresolution_grid(grids, shape):
    """Sum bilinearly resized grids at a fixed (time, position) resolution.

    Grids may have a leading batch dimension. Align endpoints as in the original
    encoder's normalized-coordinate sampling, including when reducing resolution.
    """
    values = 0.0
    for grid in grids:
        nt, nx = grid.shape[-2:]
        coords = jnp.stack(jnp.meshgrid(
            jnp.linspace(0, nt - 1, shape[0]),
            jnp.linspace(0, nx - 1, shape[1]),
            indexing="ij",
        ))

        def resize(g):
            return map_coordinates(g, coords, order=1, mode="nearest")

        values = values + (resize(grid) if grid.ndim == 2 else jax.vmap(resize)(grid))
    return values


class Grid(nnx.Module):
    """A single trainable (T, N) array, initialized to zero."""

    def __init__(self, N, T):
        if min(N, T) < 1:
            raise ValueError("Grid dimensions must be positive.")
        self.grid = nnx.Param(jnp.zeros((T, N)))

    def __call__(self):
        return self.grid[...]


class MultiResolutionGrid(nnx.Module):
    """Learn or predict multiresolution grids and sum them at output size (T, N).

    Levels double in time and space, starting at (base_T, base_N). With
    in_features set, linear heads predict each grid from a feature batch.
    """

    def __init__(
        self, N, T, num_levels=4, *, base_N=2, base_T=32, in_features=None, rngs
    ):
        if min(N, T, base_N, base_T, num_levels) < 1:
            raise ValueError("Grid dimensions and num_levels must be positive.")
        self.output_shape = (T, N)
        self.grid_shapes = tuple(
            (base_T * 2**i, base_N * 2**i) for i in range(num_levels)
        )
        if in_features is None:
            self.heads = None
            self.grids = nnx.List([
                nnx.Param(0.1 * jax.random.normal(rngs.params(), shape))
                for shape in self.grid_shapes
            ])
        else:
            self.grids = None
            initializer = jax.nn.initializers.variance_scaling(
                0.25, "fan_in", "truncated_normal"
            )
            self.heads = nnx.List([
                nnx.Linear(in_features, nt * nx, kernel_init=initializer, rngs=rngs)
                for nt, nx in self.grid_shapes
            ])

    def __call__(self, features=None):
        if self.heads is None:
            if features is not None:
                raise ValueError("Set in_features to predict grids from features.")
            grids = [grid[...] for grid in self.grids]
        else:
            if features is None:
                raise ValueError("Encoder features are required for this grid.")
            grids = [
                0.1 * head(features).reshape(features.shape[0], nt, nx)
                for head, (nt, nx) in zip(self.heads, self.grid_shapes)
            ]
        return resize_multiresolution_grid(grids, self.output_shape)


def _uniform(low, high):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=low, maxval=high)

    return init


class FourierLayer(nnx.Module):
    """Learned sine filter from imitation_autoencoder/models/fields.py."""

    def __init__(self, in_features, out_features, weight_scale, *, rngs):
        self.linear = nnx.Linear(
            in_features, out_features, bias_init=_uniform(-jnp.pi, jnp.pi), rngs=rngs
        )
        self.linear.kernel[...] *= jnp.asarray(weight_scale)[..., None]

    def __call__(self, coords):
        return jnp.sin(self.linear(coords))


class FourierNet(nnx.Module):
    """Multiplicative sine-filter network evaluated on a fixed (T, N) grid.

    Adapted from references/imitation_autoencoder/models/fields.py. Each hidden
    linear transform is multiplied by a fresh learned sine filter of the same
    coordinates. Input scale controls angular frequencies (no RFF-style 2*pi).
    """

    def __init__(
        self,
        N,
        T,
        hidden_dim=256,
        n_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        bias=True,
        output_act=False,
        *,
        rngs,
    ):
        if hidden_dim < 1 or n_layers < 0 or weight_scale < 0:
            raise ValueError("Invalid hidden dimension, layer count, or weight scale.")
        self.coords = nnx.Variable(_coordinates(N, T))
        bound = jnp.sqrt(weight_scale / hidden_dim)
        self.linear = nnx.List([
            nnx.Linear(
                hidden_dim, hidden_dim, use_bias=bias,
                kernel_init=_uniform(-bound, bound), rngs=rngs,
            )
            for _ in range(n_layers)
        ])
        self.output_linear = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.output_act = output_act
        self.filters = nnx.List([
            FourierLayer(
                2, hidden_dim, jnp.asarray(input_scale) / jnp.sqrt(n_layers + 1),
                rngs=rngs,
            )
            for _ in range(n_layers + 1)
        ])

    def __call__(self):
        coords = self.coords[...]
        out = self.filters[0](coords)
        for layer, sine_filter in zip(self.linear, self.filters[1:]):
            out = sine_filter(coords) * layer(out)
        out = self.output_linear(out)[..., 0]
        return jnp.sin(out) if self.output_act else out
