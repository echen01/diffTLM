"""Fit the Jude recording. Run from the repository directory:

    uv run python -m difftlm.fit_sample

Use --network rff for Fourier features or --network multiscale-grid for directly
learned multiresolution grids instead of the convolutional encoder. Use --network
grid for a single trainable array at the simulator's full resolution.
Use --network mfn for a multiplicative sine-filter network.
Use --network all to fit every representation into its own output subfolder.
RFF uses tanh hidden layers and separate time/position Fourier bandwidths;
--rff-scale controls bandwidth. --area-scale controls the area gain for every
representation.
"""

from pathlib import Path
import json
from time import perf_counter
from typing import Literal

from flax import nnx
import jax
import jax.numpy as jnp
import librosa
import numpy as np
import optax
import soundfile as sf
from swift_f0 import SwiftF0
from tqdm import tqdm
import tyro

from .tract import TransmissionLineSimulator, SimulatorOptions
from .encoder import ConvEncoder
from .fields import FourierNet, MultiResolutionGrid, RFFMLP, Grid
from .glottis import lf_pulse
from .loss import MultiResolutionSTFTLoss
from .viz import save_tract_visualizations


def field_to_cross_section(x, scale=0.35, bias=0.1):
    """Convert a field to positive areas, holding the glottis at its rest value.

    Bias sets the softplus input for a zero field, independently of its scale.
    Passing None uses bias=scale.
    """
    x = 1 + x
    x = x.at[:, 0].set(1.0)
    logits = x * scale
    if bias is not None:
        logits = logits + (bias - scale)
    return jnp.clip(jax.nn.softplus(logits) + 1e-4, max=16)


def fit_utterance(
    tgt, uin, sim, model, n_steps=250, area_scale=0.35, spectrogram=None,
    area_bias=0.1, L=17.5,
):
    """Fit model in place; pass a cached spectrogram when using ConvEncoder.

    All representations share the area scale and bias; bias sets the resting area.
    Return synthesized audio, areas, and loss history.
    """
    def areas(model):
        field = model() if spectrogram is None else model(spectrogram)
        field = field.reshape(sim.n_windows, sim.N)
        return field_to_cross_section(field, scale=area_scale, bias=area_bias)

    peak_lr = 1e-3 if spectrogram is None else 1e-4
    if isinstance(model, (Grid, MultiResolutionGrid)):
        peak_lr = 1e-1
    elif isinstance(model, RFFMLP):
        peak_lr = 1e-3
    elif isinstance(model, FourierNet):
        peak_lr = 5e-3
    lr = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=peak_lr,
        warmup_steps=min(50, n_steps - 1),
        decay_steps=n_steps,
    )
    optim = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    mel_loss = MultiResolutionSTFTLoss(
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[160, 80, 40],
        win_lengths=[800, 400, 160],
        sample_rate=sim.SR,
        scale="mel",
        n_bins=128,
        scale_invariance=False,
        w_sc=0.0,
        w_log_mag=1.0,
    )
    target_features = jax.jit(mel_loss.features)(tgt)

    def objective(model, key, target_features):
        pout = sim.rollout(uin, areas(model), L=L, key=key)
        return mel_loss.compare(mel_loss.features(pout), target_features)

    @nnx.jit
    def step(model, optim, n, target_features):
        loss, grads = nnx.value_and_grad(objective)(
            model,
            jax.random.key(n),
            target_features,
        )
        optim.update(model, grads)
        return loss

    history = []
    with tqdm(range(n_steps), desc="Fitting", unit="step") as progress:
        for n in progress:
            loss = step(model, optim, n, target_features)
            history.append(loss)
            progress.set_postfix(loss=f"{float(loss):.4f}")

    A = areas(model)
    pout = sim.rollout(uin, A, L=L, key=jax.random.key(0))
    return pout, A, jnp.stack(history)


ROOT = Path(__file__).resolve().parents[1]


def main(
    path: Path = ROOT / "assets/jude.wav",
    n_steps: int = 500,
    duration: float | None = None,
    output: Path = ROOT / "outputs/jude_fit",
    network: Literal["cnn", "rff", "grid", "multiscale-grid", "mfn", "all"] = "cnn",
    area_scale: float = 0.35,
    area_bias: float = 0.1,
    rff_scale: tuple[float, float] = (10.0, 2.0),
    rff_encoding_dim: int = 256,
    rff_hidden_dim: int = 512,
    rff_activation: Literal["leaky_relu", "tanh"] = "tanh",
):
    """Fit an audio file; optionally use just its first `duration` seconds.

    rff_scale sets Fourier bandwidths in (time, tract position), independently.
    area_scale controls field amplitude for every representation.
    area_bias sets the zero-field softplus input; 1 matches the reference.
    """
    sr = 32000
    L = 17.5
    tgt, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    tgt = tgt - tgt.mean()
    tgt = tgt / (np.max(np.abs(tgt)) + 1e-8)
    sim = TransmissionLineSimulator(
        SimulatorOptions(SR=sr, TF=len(tgt) / sr, fft_size=1024)
    )
    tgt = tgt[: sim.NF]

    print(
        f"Estimating pitch with SwiftF0 (audio duration: {len(tgt) / sr:.2f}s)...",
        flush=True,
    )
    start = perf_counter()
    # Avoid ONNX Runtime pinning workers to CPUs outside the allocated set.
    pitch = SwiftF0(threads=1).detect(tgt, sr, fmin=65, fmax=500)
    print(f"Pitch estimated in {perf_counter() - start:.2f}s.", flush=True)
    f0 = pitch.pitch_hz
    rms = librosa.feature.rms(y=tgt, frame_length=2048, hop_length=320)[0]
    rms = rms / (rms.max() + 1e-8)
    uin = lf_pulse(
        jnp.asarray(f0),
        jnp.asarray(rms),
        jnp.asarray(1 - pitch.confidence),
        key=jax.random.key(0),
        sr=sr,
        nf=sim.NF,
    )

    networks = (
        ("cnn", "rff", "grid", "multiscale-grid", "mfn")
        if network == "all" else (network,)
    )
    for network_name in networks:
        run_output = output / network_name if network == "all" else output
        spectrogram = None
        match network_name:
            case "cnn":
                mel = librosa.feature.melspectrogram(
                    y=tgt,
                    sr=sr,
                    n_fft=2048,
                    win_length=800,
                    hop_length=200,
                    n_mels=128,
                )
                log_mel = np.log10(np.maximum(mel, 1e-10))
                log_mel = np.maximum(log_mel, log_mel.max() - 8)
                spectrogram = jnp.asarray((log_mel + 4) / 4)[None, ..., None]
                model = ConvEncoder(
                    spectrogram.shape[2], N=sim.N, T=sim.n_windows, rngs=nnx.Rngs(0)
                )
            case "multiscale-grid":
                model = MultiResolutionGrid(N=sim.N, T=sim.n_windows, rngs=nnx.Rngs(0))
            case "grid":
                model = Grid(N=sim.N, T=sim.n_windows)
            case "mfn":
                model = FourierNet(N=sim.N, T=sim.n_windows, rngs=nnx.Rngs(0))
            case "rff":
                model = RFFMLP(
                    N=sim.N,
                    T=sim.n_windows,
                    encoding_dim=rff_encoding_dim,
                    hidden_dim=rff_hidden_dim,
                    scale=rff_scale,
                    activation=rff_activation,
                    rngs=nnx.Rngs(0),
                )
            case _:
                raise ValueError(f"Unknown network: {network_name!r}")
        print(f"Fitting {network_name} for {n_steps} steps...", flush=True)
        pout, A, losses = fit_utterance(
            jnp.asarray(tgt),
            uin,
            sim,
            model,
            n_steps,
            spectrogram=spectrogram,
            area_bias=area_bias,
            area_scale=area_scale,
            L=L,
        )

        run_output.mkdir(parents=True, exist_ok=True)
        config = {
            "network": network_name,
            "path": str(path.resolve()),
            "n_steps": n_steps,
            "sample_rate": sr,
            "duration": len(tgt) / sr,
            "seed": 0,
            "area_scale": area_scale,
            "area_bias": area_bias,
        }
        if network_name == "rff":
            config.update(
                fourier_scale=rff_scale,
                encoding_dim=rff_encoding_dim, hidden_dim=rff_hidden_dim,
                peak_lr=1e-3,
                activation=rff_activation,
            )
        (run_output / "config.json").write_text(json.dumps(config, indent=2))
        pout = np.asarray(pout)
        sf.write(run_output / "target.wav", tgt, sr, subtype="FLOAT")
        sf.write(run_output / "uin.wav", np.asarray(uin), sr, subtype="FLOAT")
        sf.write(run_output / "fitted.wav", pout / (np.max(np.abs(pout)) + 1e-8), sr)
        np.savez(
            run_output / "fit.npz",
            areas=np.asarray(A),
            losses=np.asarray(losses),
            f0=f0,
            rms=rms,
            uin=np.asarray(uin),
            pout=pout,
        )
        save_tract_visualizations(
            A, run_output,
            audio_path=run_output / "fitted.wav",
            target_audio=tgt,
            reconstructed_audio=pout,
            length_cm=L,
            hop_seconds=sim.hop_length / sim.SR,
            window_seconds=sim.frame_length / sim.SR,
        )
        print(f"Saved audio, fitted areas, and tube visualizations to {run_output}")
        if network == "all":
            # Release this model and its compiled programs before the next fit.
            del model, pout, A, losses
            jax.clear_caches()


if __name__ == "__main__":
    tyro.cli(main)
