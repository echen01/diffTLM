"""Load the MRI StyleGAN checkpoint with the bundled NVIDIA runtime."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from ..ext.stylegan.networks_stylegan2 import Generator


def default_checkpoint() -> Path:
    return Path(__file__).resolve().parents[2] / "weights" / "usc-span-stylegan2.safetensors"


def resolve_checkpoint(
    checkpoint: str | Path | None = None,
    *,
    repo_id: str = "emc348/usc-span-stylegan2",
    filename: str = "usc-span-stylegan2.safetensors",
) -> Path:
    """Return a local checkpoint path, downloading from Hugging Face Hub if omitted and not cached locally."""
    if checkpoint is not None:
        return Path(checkpoint)
    local = default_checkpoint()
    if local.is_file():
        return local
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


@torch.no_grad()
def speaker_mean_w(generator, label, seed, n_samples=2048):
    """Average W for one speaker and broadcast it across synthesis layers."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.randn(n_samples, generator.z_dim, device="cpu", generator=rng)
    samples = []
    for batch in draws.split(128):
        ws = generator.mapping(batch.to(label), label.expand(len(batch), -1))
        samples.append(ws[:, 0].contiguous())
    mean = torch.cat(samples).mean(0)
    return mean.expand(1, ws.shape[1], -1)


def save_stylegan(generator, checkpoint: str | Path):
    """Save a StyleGAN generator and its constructor kwargs to a safetensors file."""
    state_dict = {key: value.detach().cpu().contiguous() for key, value in generator.state_dict().items()}
    metadata = {"init_kwargs": json.dumps(dict(generator.init_kwargs))}
    save_file(state_dict, Path(checkpoint), metadata=metadata)


def load_stylegan(
    checkpoint: str | Path | None = None,
    device="cpu",
    *,
    repo_id: str = "emc348/usc-span-stylegan2",
    filename: str = "usc-span-stylegan2.safetensors",
):
    """Load the StyleGAN generator from a safetensors checkpoint for inference.

    When ``checkpoint`` is omitted, uses ``weights/usc-span-stylegan2.safetensors`` if
    present and otherwise downloads it from Hugging Face Hub.

    The returned module is in evaluation mode with frozen weights. Gradients
    through latent inputs remain available. CUDA operators compile lazily on
    first GPU use; CPU inference uses the bundled PyTorch reference operators.
    """
    checkpoint = resolve_checkpoint(checkpoint, repo_id=repo_id, filename=filename)
    with safe_open(checkpoint, framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
        if "init_kwargs" not in metadata:
            raise ValueError("checkpoint metadata must include 'init_kwargs'")
        init_kwargs = json.loads(metadata["init_kwargs"])
        state_dict = {key: stream.get_tensor(key) for key in stream.keys()}
    generator = Generator(**init_kwargs)
    generator.load_state_dict(state_dict, strict=True)
    return generator.eval().requires_grad_(False).to(device)
