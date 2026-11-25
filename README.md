# Reconstructing the Vocal Tract with Differentiable Acoustic Simulation


![image](assets/method.jpg)

**Reconstructing the Vocal Tract with Differentiable Acoustic Simulation** \
Eric Ming Chen, Jin Woo Lee, Vincent Sitzmann \
NeurIPS 2026 Spotlight Paper \
https://arxiv.org/abs/2609.36737 

*The vocal tract is the region of the human body responsible for filtering one's voice to create speech. In this paper, we present a differentiable and GPU accelerated acoustic simulator for the vocal tract. The differentiable simulator synthesizes speech by propagating sound along an acoustic tube model of the vocal tract, and via its gradients, can solve the inverse problem: reconstructing the shape of the vocal tract solely from the sound it produces. Although the inverse mapping between geometry and sound is notoriously non-convex, we discover that gradient descent succeeds with three technical contributions: (1) we design a frequency domain formulation of the vocal tract's fluid dynamics that is 70x more GPU parallelizable than finite differences in time, (2) we integrate a differentiable model for turbulence to synthesize consonants, and (3) similar to prior work in implicit neural representations (INRs) and neural fields, we find that parameterizing the geometry with a neural network accelerates convergence and escapes local minima that trap discrete representations. Because the simulator is differentiable, it is readily integrated with other deep learning pipelines to enable novel linguistics and medical imaging applications. (1) We demonstrate self-supervised autoencoding of vocal tract shapes across 11 languages, and (2) we couple our simulator with a generative model of MRI (magnetic resonance imaging) images to reconstruct one's moving vocal tract from only their speech without paired data.*

### Environment

To sync the environment, run:

```bash
uv sync
```

The simulator is implemented in both JAX (`difftlm.jax`) and PyTorch (`difftlm.torch`).

### Fitting an audio sample (JAX)

To fit `assets/jude.wav` with different neural field representations, run:

```bash
uv run python -m difftlm.jax.fit_sample --network all --output outputs/jude
```

The script first estimates the glottal pulse using SwiftF0, then optimizes the tube geometry to match the target audio. 500 training steps are used by default.

Choose a representation with `--network`:

| Value | Representation |
| --- | --- |
| `cnn` | Convolutional encoder with a multiresolution grid (default) |
| `rff` | MLP with random Fourier features |
| `grid` | A single directly optimized grid |
| `multiscale-grid` | Directly optimized multiresolution grids |
| `mfn` | Multiplicative Fourier filter network |
| `all` | Run all five representations sequentially |

To fit your own recording, or run a shorter experiment:

```bash
uv run python -m difftlm.jax.fit_sample \
    --path /path/to/audio.wav \
    --network rff \
    --duration 2.0 \
    --n-steps 200 \
    --output outputs/custom_rff
```

`--duration` selects the first N seconds; omit it to use the full recording.

### Reconstructing MRI from speech (PyTorch)

Reconstruct a midsagittal MRI video from speech using the pretrained StyleGAN2 checkpoint ([`emc348/usc-span-stylegan2`](https://huggingface.co/emc348/usc-span-stylegan2)):

```bash
uv run python -m difftlm.torch.reconstruct \
    --path assets/jude.wav \
    --subject-id 1 \
    --output outputs/jude_mri
```

The checkpoint (`usc-span-stylegan2.safetensors`) is downloaded automatically from Hugging Face Hub on first run and cached locally. You can also place it at `weights/usc-span-stylegan2.safetensors` or pass a local path with `--checkpoint`.

This initializes from the speaker's mean latent and optimizes temporal StyleGAN latents through the differentiable area extractor and PyTorch vocal tract simulator (400 steps by default). The output folder contains `mri.mp4` (with reconstructed audio), `target.wav`, `reconstructed.wav`, `reconstruction.npz`, and `metadata.json`.

Common options (run with `--help` for all options):

| Flag | Description |
| --- | --- |
| `--subject-id` | Speaker conditioning ID (1–75; inferred from `subNNN` filenames, otherwise 1) |
| `--checkpoint` | Local `.safetensors` checkpoint path (downloads from Hugging Face if omitted) |
| `--start`, `--duration` | Audio interval in seconds (defaults to 3 seconds starting at the detected audio onset) |
| `--n-steps` | Optimization steps (default: 400) |
| `--checkpoint-areas` | Recompute frames during backward to reduce GPU memory (pair with e.g. `--batch-size 8`) |
| `--overlay-areas` | Also save `mri_overlay.mp4` with the extracted airway rays |
| `--device` | `auto` (default), `cuda`, or `cpu` |

For a quick smoke test, add `--duration 0.2 --n-steps 2 --batch-size 4`. GPU runs compile StyleGAN's CUDA kernels on first use (requiring `nvcc`, a C++ compiler, and `ninja`); CPU runs fall back to PyTorch ops.

The bundled StyleGAN code in `difftlm/ext/stylegan/` retains its original [StyleGAN license](difftlm/ext/stylegan/LICENSE.txt).

### Tips and Tricks
Learning rate and area scale are the two most important parameters to scale when fitting a new sample. Optimal learning rates vary significantly depending on the neural field architecture used. 


### To Dos
- [ ] Upload autoencoder weights and code


### Citation
```
@inproceedings{Chen2026DiffTLM,
  author = {Chen, Eric Ming and Lee, Jin Woo and Sitzmann, Vincent},
  title = {Reconstructing the Vocal Tract with Differentiable Acoustic Simulation},
  booktitle = {Proc. NeurIPS},
  year = {2026}
}
```
