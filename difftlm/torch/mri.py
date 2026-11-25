"""Differentiable fixed-ray area extraction for aligned 64 × 64 midsagittal MRI."""

import math

import torch
import torch.nn.functional as F


def measure_rays(images):
    """Return 33 area proxies, their soft air samples, and the ray coordinates."""
    if images.shape[-2:] != (64, 64):
        raise ValueError("the MRI area extractor requires 64 x 64 grayscale images")

    # vocal-tts constructs these coordinates on the CPU in float32.
    angles = torch.cat([torch.linspace(0, math.pi / 2, 20), torch.full((13,), math.pi / 2)])
    offsets = torch.cat([torch.zeros(20), torch.linspace(0, 16, 13)])
    distances = torch.linspace(6, 24, 100)
    x = 32 + distances * angles.cos()[:, None] - offsets[:, None]
    y = 40 - distances * angles.sin()[:, None]
    rays = torch.stack([x, y], dim=-1).to(images)
    grid = (rays - 32) / 32

    air = torch.sigmoid(((1 - images) - 0.88) * 30).reshape(-1, 1, 64, 64)
    density = F.grid_sample(air, grid[None].expand(len(air), -1, -1, -1), align_corners=False)
    weights = torch.sigmoid((density[:, 0] - 0.8) * 30)
    weights = weights.reshape(*images.shape[:-2], 33, 100)
    return weights.sum(-1) * 0.1, weights, rays - 0.5


def mri_to_area(images):
    """Measure 33 glottis-to-lips area proxies, preserving leading batch dimensions."""
    areas, _, _ = measure_rays(images)
    return areas
