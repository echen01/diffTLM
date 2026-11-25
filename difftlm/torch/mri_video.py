"""Area extraction and lightweight overlays for MRI video frames."""

from contextlib import closing, contextmanager

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch

from .mri import measure_rays


@contextmanager
def open_mri_video(output, name, fps, input_size, video_size=256):
    """Encode MRI frames at display resolution with the reconstructed audio."""
    scale = f"scale={video_size}:{video_size}:flags=bilinear"
    with closing(imageio_ffmpeg.write_frames(
        str(output / name), (input_size, input_size), fps=fps, codec="libx264",
        pix_fmt_out="yuv420p", audio_path=str(output / "reconstructed.wav"),
        audio_codec="aac", output_params=["-vf", scale, "-shortest"],
    )) as writer:
        next(writer)
        yield writer


@torch.no_grad()
def extract_video_areas(images, area_scale=1.0, *, overlay=False):
    areas, weights, points = measure_rays(images)
    geometry = dict(areas=(areas.clamp(.01, 16) * area_scale).cpu().numpy())
    if overlay:
        geometry.update(weights=weights.cpu().numpy(), points=points.cpu().numpy())
    return geometry


def draw_area_overlay(frame, geometry, index, *, size=256):
    """Draw actual air samples or fitted walls on an enlarged RGB MRI frame."""
    scale = size / frame.shape[1]
    base = Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR).convert("RGBA")
    overlay = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(overlay)

    def coordinates(points):
        # Match pixel centers when upscaling the original 64x64 MRI.
        return [tuple(p) for p in ((points + .5) * scale - .5)]

    points = geometry["points"]
    weights = geometry["weights"][index]
    for ray in points:
        draw.line(coordinates(ray[[0, -1]]), fill=(255, 255, 255, 32), width=1)
    base = Image.alpha_composite(base, overlay)
    pixel = np.rint((points + .5) * scale - .5).astype(int).reshape(-1, 2)
    valid = ((pixel >= 0) & (pixel < size)).all(-1)
    mask = np.zeros((size, size), dtype=np.uint8)
    np.maximum.at(mask, (pixel[valid, 1], pixel[valid, 0]),
                  np.rint(weights.reshape(-1)[valid] * 235).astype(np.uint8))
    overlay = Image.new("RGBA", base.size, (245, 151, 30, 0))
    overlay.putalpha(Image.fromarray(mask).filter(ImageFilter.MaxFilter(3)))

    base = Image.alpha_composite(base, overlay)
    text_layer = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(text_layer)
    font = ImageFont.load_default(size=12)
    areas = geometry["areas"][index]
    label = f"Area proxy: min {areas.min():.2f}  max {areas.max():.2f}"
    draw.rectangle((0, size-21, size, size), fill=(0, 0, 0, 185))
    draw.text((5, size-18), label, font=font, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(base, text_layer).convert("RGB"))
