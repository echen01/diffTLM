"""Neural network modules for vocal autoencoder simulation.

Implements Wav2Vec and Vision Transformer models using Flax NNX.
"""

import math
from typing import Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


def apply_rotary_pos_emb(x: jax.Array, freqs_cis: jax.Array) -> jax.Array:
    """Applies rotary position embeddings to query or key states."""
    x_complex = x.astype(jnp.float32).view(jnp.complex64)
    x_complex = x_complex.reshape(x.shape[:-1] + (-1,))
    freqs_cis = jnp.expand_dims(freqs_cis, axis=(0, 2))
    L = freqs_cis.shape[1]
    x_rotated = x_complex.at[:, -L:, :].multiply(freqs_cis)
    x_out = x_rotated.astype(x_complex.dtype).view(x.dtype)
    return x_out.reshape(x.shape)


def precompute_rope_freqs_1d(dim: int, maxlen: int, theta: float = 1e4) -> jax.Array:
    """Precomputes 1D rotary position embeddings."""
    freqs = 1.0 / (theta ** (jnp.arange(0.0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(maxlen)
    freqs = jnp.outer(t, freqs)
    real = jnp.cos(freqs)
    imag = jnp.sin(freqs)
    freqs_cis = jax.lax.complex(real, imag)
    return freqs_cis


def precompute_rope_freqs_2d(dim: int, maxlen: int, theta: float = 1e4) -> jax.Array:
    """Precomputes 2D rotary position embeddings."""
    dim = dim // 2
    T = int(maxlen**0.5)
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    positions = jnp.arange(T, dtype=jnp.float32)
    freqs_h = jnp.einsum("i,j->ij", positions, freqs)
    freqs_w = jnp.einsum("i,j->ij", positions, freqs)
    freqs = jnp.concatenate(
        [
            jnp.tile(freqs_h[:, None, :], (1, T, 1)),
            jnp.tile(freqs_w[None, :, :], (T, 1, 1)),
        ],
        axis=-1,
    )
    real = jnp.cos(freqs).reshape(maxlen, dim)
    imag = jnp.sin(freqs).reshape(maxlen, dim)
    return jax.lax.complex(real, imag)


class RoPEAttention(nnx.Module):
    """Multi-head attention layer with Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, n_heads: int, rngs: nnx.Rngs):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.wk = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.wv = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.wo = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

        self.q_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)
        self.k_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)

    def __call__(self, x: jax.Array, rope_freqs: jax.Array) -> jax.Array:
        B, L, _ = x.shape

        xq = self.wq(x).reshape(B, L, self.n_heads, self.head_dim)
        xk = self.wk(x).reshape(B, L, self.n_heads, self.head_dim)
        xv = self.wv(x).reshape(B, L, self.n_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = apply_rotary_pos_emb(xq, rope_freqs)
        xk = apply_rotary_pos_emb(xk, rope_freqs)

        attn = jax.nn.dot_product_attention(xq, xk, xv)
        attn = attn.reshape(B, L, self.dim)
        output = self.wo(attn)
        return output


class SwiGLUMLP(nnx.Module):
    """SwiGLU multi-layer perceptron block."""

    def __init__(self, dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.w1 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs)
        self.w2 = nnx.Linear(hidden_dim, dim, use_bias=False, rngs=rngs)
        self.w3 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nnx.Module):
    """Transformer block using RoPE attention and SwiGLU MLP."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 8 / 3,
        norm_eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.norm_eps = norm_eps
        mlp_hidden_dim = int(self.dim * mlp_ratio)

        self.attention = RoPEAttention(dim=dim, n_heads=n_heads, rngs=rngs)
        self.mlp = SwiGLUMLP(dim=dim, hidden_dim=mlp_hidden_dim, rngs=rngs)
        self.norm1 = nnx.RMSNorm(dim, epsilon=norm_eps, rngs=rngs)
        self.norm2 = nnx.RMSNorm(dim, epsilon=norm_eps, rngs=rngs)

        self.attn_scale = nnx.Param(jnp.zeros(dim))
        self.mlp_scale = nnx.Param(jnp.zeros(dim))

    def __call__(self, x: jax.Array, rope_freqs: jax.Array) -> jax.Array:
        x = x + self.attention(self.norm1(x), rope_freqs)
        x = x + self.mlp(self.norm2(x))
        return x


class BottleneckPatchEmbedder(nnx.Module):
    """Splits images into patches and projects them to a model dimension."""

    def __init__(
        self,
        input_size: tuple[int, int],
        patch_size: tuple[int, int],
        pca_channels: int,
        in_channels: int,
        dim: int,
        rngs: nnx.Rngs,
    ):
        self.proj1 = nnx.Conv(
            in_channels,
            pca_channels,
            kernel_size=patch_size,
            strides=patch_size,
            kernel_init=jax.nn.initializers.xavier_uniform(
                in_axis=(0, 1, 2), out_axis=-1
            ),
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        self.proj2 = nnx.Conv(
            pca_channels,
            dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_init=jax.nn.initializers.xavier_uniform(
                in_axis=(0, 1, 2), out_axis=-1
            ),
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        B, H, W, C = x.shape
        x = self.proj2(self.proj1(x))
        x = x.reshape(B, -1, x.shape[3])
        return x


class VisionTransformer(nnx.Module):
    """Vision Transformer module with rotary embeddings."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: int,
        patch_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_aux: int,
        n_registers: int,
        rngs: nnx.Rngs,
    ):
        self.patch_size = patch_size
        self.n_aux = n_aux
        self.n_registers = n_registers
        self.n_patches = (input_size // patch_size) ** 2

        @nnx.split_rngs(splits=n_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_transformer_block(rngs: nnx.Rngs):
            return TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                norm_eps=1e-6,
                rngs=rngs,
            )

        self.blocks = create_transformer_block(rngs)
        self.x_embedder = BottleneckPatchEmbedder(
            input_size=(input_size, input_size),
            patch_size=(patch_size, patch_size),
            pca_channels=128,
            in_channels=in_channels,
            dim=dim,
            rngs=rngs,
        )

        self.rope_freqs = precompute_rope_freqs_2d(dim // n_heads, self.n_patches)
        self.rope_freqs = nnx.Variable(self.rope_freqs)

        token_initializer = jax.nn.initializers.normal(stddev=1 / math.sqrt(dim))

        self.register_tokens = nnx.Param(
            token_initializer(key=rngs.params(), shape=(1, n_registers, dim))
        )
        self.aux_tokens = nnx.Param(
            token_initializer(key=rngs.params(), shape=(1, n_aux, dim))
        )

        self.total_tokens = n_aux + n_registers + self.n_patches

        self.fc = nnx.Linear(dim, out_channels, rngs=rngs)
        self.fc_aux = nnx.Linear(dim, out_channels, rngs=rngs)
        self.norm_out = nnx.RMSNorm(dim, rngs=rngs)
        self.norm_aux = nnx.RMSNorm(dim, rngs=rngs)

    @staticmethod
    def transformer(x, rope_freqs, layers):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def transformer(x, layer):
            x = layer(x, rope_freqs)
            return x

        return transformer(x, layers)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x = self.x_embedder(x)
        B, L, D = x.shape

        aux_tokens = jnp.broadcast_to(self.aux_tokens.value, (B, self.n_aux, D))
        register_tokens = jnp.broadcast_to(
            self.register_tokens.value, (B, self.n_registers, D)
        )

        x = jnp.concatenate([aux_tokens, register_tokens, x], axis=1)
        x = self.transformer(x, self.rope_freqs[:L], self.blocks)

        aux = x[:, : self.n_aux]
        aux = self.norm_aux(aux)
        aux = self.fc_aux(aux)

        x = x[:, self.n_aux + self.n_registers :]
        x = self.norm_out(x)
        x = self.fc(x)

        return x, aux


class Wav2VecBlock(nnx.Module):
    """1D Convolution block with RMS normalization and GELU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        rngs: nnx.Rngs,
    ):
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            kernel_init=jax.nn.initializers.kaiming_normal(),
            strides=stride,
            padding=padding,
            rngs=rngs,
        )

        self.norm = nnx.RMSNorm(
            num_features=out_channels, epsilon=1e-6, dtype=jnp.float32, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        x = self.norm(x)
        x = jax.nn.gelu(x)
        return x


class Wav2VecEncoder(nnx.Module):
    """Temporal convolution feature extractor followed by a transformer block."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 512,
        *,
        dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        n_aux: int = 2,
        aux_dim: int = 1,
        n_registers: int = 4,
        rngs: nnx.Rngs,
    ):
        self.n_registers = n_registers
        self.n_aux = n_aux
        self.aux_dim = 1

        self.block1 = Wav2VecBlock(in_channels, 512, kernel_size=10, stride=5, rngs=rngs)
        self.block2 = Wav2VecBlock(512, 512, kernel_size=3, stride=2, rngs=rngs)
        self.block3 = Wav2VecBlock(512, 512, kernel_size=3, stride=2, rngs=rngs)
        self.block4 = Wav2VecBlock(512, 512, kernel_size=3, stride=2, rngs=rngs)
        self.block5 = Wav2VecBlock(512, 512, kernel_size=3, stride=2, rngs=rngs)
        self.block6 = Wav2VecBlock(512, 512, kernel_size=2, stride=2, rngs=rngs)
        self.block7 = Wav2VecBlock(512, dim, kernel_size=2, stride=2, rngs=rngs)

        @nnx.split_rngs(splits=n_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_transformer_block(rngs: nnx.Rngs):
            return TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                norm_eps=1e-6,
                mlp_ratio=8 / 3,
                rngs=rngs,
            )

        self.blocks = create_transformer_block(rngs)
        self.rope_freqs = precompute_rope_freqs_1d(dim // n_heads, 150)
        self.rope_freqs = nnx.Variable(self.rope_freqs)

        token_initializer = jax.nn.initializers.normal(stddev=1 / math.sqrt(dim))

        self.register_tokens = nnx.Param(
            token_initializer(key=rngs.params(), shape=(1, n_registers, dim))
        )
        self.aux_tokens = nnx.Param(
            token_initializer(key=rngs.params(), shape=(1, n_aux, dim))
        )

        self.total_tokens = n_aux + n_registers + 150

        self.fc = nnx.Linear(dim, out_channels, rngs=rngs)
        self.fc_aux = nnx.List(
            [
                nnx.Linear(dim, aux_dim, rngs=rngs),
                nnx.Linear(dim, aux_dim, rngs=rngs),
            ]
        )
        self.norm_out = nnx.RMSNorm(dim, rngs=rngs)
        self.norm_aux = nnx.RMSNorm(dim, rngs=rngs)

    @staticmethod
    def transformer(x, rope_freqs, layers):
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def transformer(x, layer):
            x = layer(x, rope_freqs)
            return x

        return transformer(x, layers)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x = x[..., jnp.newaxis]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        B, L, D = x.shape

        aux_tokens = jnp.broadcast_to(self.aux_tokens.value, (B, self.n_aux, D))
        register_tokens = jnp.broadcast_to(
            self.register_tokens.value, (B, self.n_registers, D)
        )

        x = jnp.concatenate([aux_tokens, register_tokens, x], axis=1)
        x = self.transformer(x, self.rope_freqs[:L], self.blocks)

        aux = x[:, : self.n_aux]
        aux = self.norm_aux(aux)
        aux = jnp.concatenate(
            [self.fc_aux[i](aux[:, i : i + 1]) for i in range(self.n_aux)], axis=1
        )

        x = x[:, self.n_aux + self.n_registers :]
        x = nnx.leaky_relu(x)
        x = self.fc(x)

        return x, aux


class Wav2VecVocalEncoder(nnx.Module):
    """Vocal encoder that extracts area functions and glottal parameters from audio waveforms."""

    def __init__(
        self,
        in_channels: int,
        n_windows: int,
        N: int = 2,
        x_coords: int = 20,
        n_levels: int = 3,
        *,
        sigmoid_bias: float = 0.0,
        rngs: nnx.Rngs,
    ):
        self.N = N
        self.n_windows = n_windows
        self.n_levels = n_levels
        self.x_coords = x_coords
        self.Ns = [N * 2**i for i in range(self.n_levels)]
        self.Ns = np.cumsum(self.Ns).tolist()
        self.N_tot = self.Ns[-1]

        self.wav2vec = Wav2VecEncoder(
            in_channels,
            self.N_tot + 3,
            n_layers=12,
            dim=768,
            n_heads=12,
            n_registers=4,
            n_aux=2,
            aux_dim=1,
            rngs=rngs,
        )

        self.map_coordinates = jax.vmap(jnp.interp, in_axes=(None, None, 0))

    def __call__(
        self, x: jax.Array, key: jax.Array, image: jax.Array = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        grids, aux = self.wav2vec(x)

        tract_length = jax.nn.sigmoid(aux[:, 0, 0]) * 6 - 3 + 16
        s0 = jax.nn.sigmoid(aux[:, 1]) * 2 + 0.5

        glottal_grids = grids[:, :, -3:]
        grids = grids[:, :, :-3] * 0.1
        rms = jax.nn.sigmoid(glottal_grids[:, :, 0] + 1)

        semitone_shift = jax.nn.tanh(glottal_grids[:, :, 1] * 5) * 6
        f0s = 250 * 2 ** (semitone_shift / 12)
        glottis_opening = jax.nn.sigmoid(glottal_grids[:, :, 2] - 4)

        B, T, _ = grids.shape

        S = jnp.ones((B, T, self.x_coords + 1))

        x_range = jnp.linspace(0, 1, self.x_coords)
        grids = grids.reshape(B * T, self.N_tot)
        values = self.map_coordinates(
            x_range, jnp.linspace(0, 1, self.Ns[0]), grids[:, : self.Ns[0]]
        )

        for i in range(1, self.n_levels):
            values += self.map_coordinates(
                x_range,
                jnp.linspace(0, 1, self.Ns[i] - self.Ns[i - 1]),
                grids[:, self.Ns[i - 1] : self.Ns[i]],
            )

        values = values.reshape(S[..., 1:].shape)
        S = S.at[..., 1:].set(S[..., 1:] + values)
        S = nnx.softplus(S)

        S = S.transpose(0, 2, 1)
        S = S.reshape(-1, S.shape[-1])

        S = self.map_coordinates(
            jnp.linspace(0, 1, self.n_windows),
            jnp.linspace(0, 1, T),
            S,
        )

        S = S.reshape(B, self.x_coords + 1, self.n_windows)
        S = S.transpose(0, 2, 1)
        S = S * jnp.abs(s0[..., jnp.newaxis])
        S = S.clip(min=0.01, max=16)

        return S, glottis_opening, f0s, rms, s0, tract_length
