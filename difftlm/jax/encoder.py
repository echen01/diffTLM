"""Spectrogram-to-tract encoder adapted from vocal-warp/core/encoder.py."""

from flax import nnx
import jax

from .fields import MultiResolutionGrid


class ResNetBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, stride=1, *, rngs):
        initializer = jax.nn.initializers.variance_scaling(
            0.25, "fan_in", "truncated_normal"
        )
        self.norm1 = nnx.LayerNorm(in_channels, epsilon=1e-6, rngs=rngs)
        self.norm2 = nnx.LayerNorm(out_channels, epsilon=1e-6, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_channels, out_channels, (3, 3), strides=(stride, stride),
            padding="SAME", kernel_init=initializer, rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            out_channels, out_channels, (3, 3), padding="SAME",
            kernel_init=initializer, rngs=rngs,
        )
        if in_channels != out_channels or stride != 1:
            self.skip = nnx.Conv(
                in_channels, out_channels, (1, 1), strides=(stride, stride),
                kernel_init=initializer, rngs=rngs,
            )
        else:
            self.skip = None

    def __call__(self, x):
        h = self.conv1(jax.nn.leaky_relu(self.norm1(x)))
        h = self.conv2(jax.nn.leaky_relu(self.norm2(h)))
        return h + (x if self.skip is None else self.skip(x))


class ConvEncoder(nnx.Module):
    """Encode (batch, mel, time, 1) inputs into fixed-size (batch, T, N) fields."""

    def __init__(
        self, n_frames, N, T, num_levels=4, *, base_N=2, base_T=32, rngs
    ):
        initializer = jax.nn.initializers.variance_scaling(
            0.25, "fan_in", "truncated_normal"
        )
        self.conv_in = nnx.Conv(
            1, 64, (7, 7), strides=(2, 2), padding="SAME",
            kernel_init=initializer, rngs=rngs,
        )
        self.norm_in = nnx.LayerNorm(64, epsilon=1e-6, rngs=rngs)
        blocks = []
        channels = 64
        for out_channels in (64, 128, 128, 256, 256, 512, 512, 1024):
            stride = 1 if channels == out_channels else 2
            blocks.append(ResNetBlock(channels, out_channels, stride, rngs=rngs))
            channels = out_channels
        self.blocks = nnx.List(blocks)

        # The stem, pooling, and four strided blocks reduce time by 64.
        features = channels * ((n_frames + 63) // 64)
        self.grid = MultiResolutionGrid(
            N=N, T=T, num_levels=num_levels, base_N=base_N, base_T=base_T,
            in_features=features, rngs=rngs,
        )

    def __call__(self, spectrogram):
        x = jax.nn.leaky_relu(self.norm_in(self.conv_in(spectrogram)))
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for block in self.blocks:
            x = block(x)
        x = x.mean(axis=1)  # Pool frequency, preserving the encoded time axis.
        x = x.reshape(x.shape[0], -1)

        return self.grid(features=x)
