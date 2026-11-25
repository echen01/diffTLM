"""Reconstruct midsagittal MRI from speech with a speaker-conditioned StyleGAN."""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re

from auraloss.freq import MultiResolutionSTFTLoss
import librosa
import numpy as np
import soundfile as sf
from swift_f0 import SwiftF0
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as recompute
from tqdm.auto import trange
import tyro

from .glottis import lf_pulse
from .mri import mri_to_area
from .mri_video import draw_area_overlay, extract_video_areas, open_mri_video
from .stylegan import load_stylegan, resolve_checkpoint, speaker_mean_w
from .tract import TransmissionLineSimulator, TransmissionOptions


@dataclass
class ReconstructOptions:
    """Reconstruct midsagittal MRI from speech with a speaker-conditioned StyleGAN."""

    path: Path = Path(__file__).resolve().parents[2] / "assets/jude.wav"
    """Input speech audio or video."""
    output: Path = Path("outputs/mri")
    """Output directory."""
    subject_id: int | None = None
    """One-based SPAN ID (inferred from subNNN, otherwise 1)."""
    checkpoint: Path | None = None
    """Path to a local .safetensors checkpoint (downloads from Hugging Face if omitted)."""
    start: float | None = None
    """Start time in seconds from the beginning of the file. Defaults to the detected audio onset."""
    duration: float | None = 3.0
    sample_rate: int = 16000
    n_steps: int = 400
    learning_rate: float = 0.02
    n_latents: int = 30
    batch_size: int = 128
    tract_length: float = 14.5
    area_scale: float = 1.0
    re_crit: float = 3500.0
    u_amp: float = 400.0
    mean_samples: int = 2048
    seed: int = 101
    device: str = "auto"
    """auto, cpu, cuda, or cuda:N."""
    fps: float = 82.28
    video_size: int = 256
    checkpoint_areas: bool = False
    """Recompute rendering during backward to save memory."""
    area_frame_stride: int = 1
    """Render fewer frames and interpolate their areas."""
    overlay_areas: bool = False
    """Also save mri_overlay.mp4."""


@dataclass
class Reconstruction:
    initial_w: torch.Tensor
    latents: torch.Tensor
    areas: torch.Tensor
    audio: torch.Tensor
    losses: list[float]
    export_loss: float
    sample_rate: int
    frame_length: int
    hop_length: int
    area_scale: float


def interpolate_latents(latents, n_frames):
    return F.interpolate(
        latents.permute(1, 2, 0), size=n_frames, mode="linear", align_corners=False
    ).permute(2, 0, 1)


def synthesize_areas(generator, latents, n_frames, batch_size=128, *, area_scale=1.0,
                     checkpoint_areas=False, area_frame_stride=1):
    """Render and measure one 33-section tube per acoustic frame."""
    def render(w):
        mri = generator.synthesis(w, noise_mode="random")[:, 0] * 0.5 + 0.5
        return mri_to_area(mri).clamp(0.01, 16) * area_scale

    n_render = math.ceil(n_frames / area_frame_stride)
    chunks = [
        recompute(render, w, use_reentrant=False, preserve_rng_state=True)
        if checkpoint_areas and torch.is_grad_enabled() else render(w)
        for w in interpolate_latents(latents, n_render).split(batch_size)
    ]
    areas = torch.cat(chunks)
    if n_render != n_frames:
        areas = F.interpolate(areas.T[None], size=n_frames, mode="linear", align_corners=False)[0].T
    return areas


def fit_mri(target, pulse, generator, simulator, opt: ReconstructOptions | None = None, *,
            subject_id=None, loss_fn=None, progress=True, noise_rng=None):
    """Fit temporal W+ controls to audio, starting every layer and time at the speaker mean."""
    opt = ReconstructOptions() if opt is None else opt
    subject_id = (opt.subject_id or 1) if subject_id is None else subject_id
    if generator.img_resolution != 64 or simulator.N != 32:
        raise ValueError("MRI reconstruction requires 64 × 64 images and 33 area samples")
    if not 1 <= subject_id <= generator.c_dim:
        raise ValueError(f"subject_id must be between 1 and {generator.c_dim}")

    n_frames = 1 + (len(target) - simulator.frame_length) // simulator.hop_length
    n_samples = simulator.frame_length + (n_frames - 1) * simulator.hop_length
    target = target[:n_samples].detach()
    pulse = pulse[:n_samples].detach().double()
    noise_rng = np.random.RandomState(opt.seed) if noise_rng is None else noise_rng

    generator.eval().requires_grad_(False)
    label = target.new_zeros(1, generator.c_dim)
    label[:, subject_id - 1] = 1
    initial = speaker_mean_w(generator, label, opt.seed, n_samples=opt.mean_samples)
    latents = torch.nn.Parameter(initial.repeat(opt.n_latents, 1, 1))

    if loss_fn is None:
        loss_fn = MultiResolutionSTFTLoss(
            scale="mel", n_bins=128, sample_rate=simulator.SR,
            w_sc=0.0, perceptual_weighting=True, scale_invariance=False,
        ).to(target.device)

    def simulate(areas):
        audio = simulator.rollout(pulse, areas, L=opt.tract_length, generator=noise_rng).to(target.dtype)
        loss = loss_fn(audio[None, None], target[None, None])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite acoustic loss")
        return audio, loss

    optimizer = torch.optim.Adam([latents], lr=opt.learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.OneCycleLR(optimizer, opt.learning_rate, opt.n_steps, pct_start=0.1)
        if opt.n_steps >= 20 else None
    )
    history = []
    devices = [target.device] if target.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(opt.seed)
        with trange(max(opt.n_steps, 1), desc="Reconstructing MRI", disable=not progress) as steps:
            for _ in steps:
                with torch.set_grad_enabled(opt.n_steps > 0):
                    areas = synthesize_areas(
                        generator, latents, n_frames, opt.batch_size,
                        area_scale=opt.area_scale, checkpoint_areas=opt.checkpoint_areas,
                        area_frame_stride=opt.area_frame_stride,
                    )
                    _, loss = simulate(areas)
                history.append(loss.item())
                if opt.n_steps:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if not torch.isfinite(latents.grad).all():
                        raise RuntimeError("nonfinite latent gradient")
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                steps.set_postfix(loss=f"{history[-1]:.4f}")
        with torch.no_grad():
            audio, export_loss = simulate(areas.detach())

    return Reconstruction(
        initial, latents.detach(), areas.detach(), audio, history, export_loss.item(),
        int(simulator.SR), simulator.frame_length, simulator.hop_length, opt.area_scale,
    )


def load_audio(path, start=None, duration=3.0, *, sample_rate=16000):
    """Decode and normalize mono speech. Explicit start times are relative to the input file."""
    samples, _ = librosa.load(path, sr=sample_rate, mono=True)
    if start is None:
        samples, bounds = librosa.effects.trim(samples, top_db=35)
        start = int(bounds[0]) / sample_rate
    else:
        samples = samples[round(start * sample_rate):]
    end = None if duration is None else round(duration * sample_rate)
    audio = samples[:end].astype(np.float64)
    if len(audio) < sample_rate // 10 or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-8:
        raise ValueError("select at least 0.1 seconds of finite, non-silent audio")
    return audio / np.max(np.abs(audio)), start


def estimate_excitation(audio, device, noise_rng, *, sample_rate=16000):
    pitch = SwiftF0(threads=1).detect(audio, sample_rate, fmin=50, fmax=500)
    f0, confidence = pitch.pitch_hz, pitch.confidence
    valid = np.flatnonzero((confidence > 0.9) & (f0 >= 50) & (f0 <= 500))
    f0 = np.interp(np.arange(len(f0)), valid, f0[valid]) if len(valid) else np.full(len(f0), 150.0)
    rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=512)[0]
    rms /= rms.max()
    controls = dict(f0=f0, rms=rms, confidence=confidence)
    pulse = lf_pulse(
        torch.from_numpy(f0).double(), torch.from_numpy(rms).double(),
        torch.from_numpy(1 - confidence).double(), sample_rate, len(audio), generator=noise_rng,
    )
    return pulse.to(device=device, dtype=torch.float32), controls


@torch.no_grad()
def save_reconstruction(result, generator, target, controls, output, *, fps=82.28, video_size=256,
                        seed=101, metadata=None, overlay_areas=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audio = result.audio.cpu().numpy()
    peak = float(np.max(np.abs(audio)))
    if not math.isfinite(peak) or peak == 0:
        raise RuntimeError("the simulator produced silent or nonfinite audio")

    sr = result.sample_rate
    sf.write(output / "reconstructed.wav", audio / peak, sr)
    sf.write(output / "target.wav", target[:len(audio)], sr)

    ws = interpolate_latents(result.latents, max(1, int(len(audio) / sr * fps)))
    devices = [ws.device] if ws.is_cuda else []
    video_areas = []
    with ExitStack() as stack:
        plain = stack.enter_context(open_mri_video(output, "mri.mp4", fps, 64, video_size=video_size))
        if overlay_areas:
            overlay = stack.enter_context(
                open_mri_video(output, "mri_overlay.mp4", fps, video_size, video_size=video_size)
            )
        stack.enter_context(torch.random.fork_rng(devices=devices))
        torch.manual_seed(seed)
        for w in ws.split(1):
            image = generator.synthesis(w, noise_mode="random") * 0.5 + 0.5
            geometry = extract_video_areas(image[:, 0], result.area_scale, overlay=overlay_areas)
            video_areas.append(geometry["areas"][0])
            frame = image[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            plain.send(frame)
            if overlay_areas:
                overlay.send(np.ascontiguousarray(draw_area_overlay(frame, geometry, 0, size=video_size)))

    np.savez_compressed(
        output / "reconstruction.npz",
        initial_w=result.initial_w.cpu().numpy(),
        latents=result.latents.cpu().numpy(),
        areas=result.areas.cpu().numpy(),
        audio=audio,
        losses=result.losses,
        video_areas=np.stack(video_areas),
        video_times=np.arange(len(ws)) / fps,
        area_times=(np.arange(len(result.areas)) * result.hop_length + result.frame_length / 2) / sr,
        **controls,
    )
    details = dict(
        metadata or {},
        sample_rate=sr,
        duration=len(audio) / sr,
        fps=fps,
        video_size=[video_size, video_size],
        overlay_areas=overlay_areas,
        initial_loss=result.losses[0],
        final_training_loss=result.losses[-1],
        export_loss=result.export_loss,
    )
    (output / "metadata.json").write_text(json.dumps(details, indent=2) + "\n")


def reconstruct_audio(opt: ReconstructOptions | None = None):
    opt = ReconstructOptions() if opt is None else opt
    subject_id = opt.subject_id
    if subject_id is None:
        match = re.search(r"sub(\d{3})", Path(opt.path).stem, re.IGNORECASE)
        subject_id = int(match[1]) if match else 1
    device = ("cuda" if torch.cuda.is_available() else "cpu") if opt.device == "auto" else opt.device

    checkpoint_path = resolve_checkpoint(opt.checkpoint)
    generator = load_stylegan(checkpoint_path, device=device)
    audio, start = load_audio(opt.path, opt.start, opt.duration, sample_rate=opt.sample_rate)
    noise_rng = np.random.RandomState(opt.seed)
    pulse, controls = estimate_excitation(audio, device, noise_rng, sample_rate=opt.sample_rate)
    simulator = TransmissionLineSimulator(TransmissionOptions(
        SR=opt.sample_rate, TF=len(audio) / opt.sample_rate,
        N=32, fft_size=0.1, Re_crit=opt.re_crit, u_amp=opt.u_amp,
    ))
    result = fit_mri(
        torch.as_tensor(audio, dtype=torch.float32, device=device), pulse, generator, simulator,
        opt, subject_id=subject_id, noise_rng=noise_rng,
    )
    controls["pulse"] = pulse[:len(result.audio)].cpu().numpy()
    metadata = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in asdict(opt).items()}
    metadata.update(checkpoint=str(checkpoint_path.resolve()), start=start, subject_id=subject_id)
    save_reconstruction(
        result, generator, audio, controls, opt.output,
        fps=opt.fps, video_size=opt.video_size, seed=opt.seed,
        overlay_areas=opt.overlay_areas, metadata=metadata,
    )
    print(f"Saved MRI reconstruction to {opt.output}")
    return result


if __name__ == "__main__":
    reconstruct_audio(tyro.cli(ReconstructOptions))
