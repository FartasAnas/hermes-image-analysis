#!/usr/bin/env python3
"""
Shared image loading utilities for the Hermes Image Analysis pipeline.

Provides safe image loading with proper alpha-channel handling.
Used by: analyze_image.py, llava_engine.py, pixel_analysis.py
"""

from PIL import Image


def load_image_safely(image_path):
    """
    Load an image, handling LA/RGBA alpha channels properly.

    For LA/RGBA images, the actual visual content may be in the alpha channel.
    We composite onto a white background to preserve what the human eye sees.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL.Image in RGB mode.
    """
    img = Image.open(image_path)

    if img.mode in ("LA", "PA"):
        # Luminance+Alpha: alpha channel IS the visible content
        # Composite alpha onto white background
        background = Image.new("L", img.size, 255)
        alpha = img.getchannel("A")
        background.paste(alpha, mask=alpha)
        return background.convert("RGB")

    elif img.mode in ("RGBA", "RGBa"):
        # Composite onto white background to preserve transparency
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img)
        return background

    elif img.mode == "P":
        # Palette mode — convert to RGBA first
        return img.convert("RGBA").convert("RGB")

    else:
        return img.convert("RGB")
