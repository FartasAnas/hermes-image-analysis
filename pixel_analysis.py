#!/usr/bin/env python3
"""
Pixel-Level Image Analysis Engine
==================================
Fills BLIP's blind spots by analyzing image pixels directly:
  - Dominant color extraction (exact hex values)
  - Color diversity / palette richness
  - Radial symmetry (zoom/streak/motion effect detection)
  - Brightness distribution
  - Contrast analysis

Optimized: NumPy vectorized HSV conversion, smart downsampling, tuned thresholds.
"""
import numpy as np
from PIL import Image


def _rgb_to_hsv_vectorized(rgb_array):
    """
    Vectorized RGB → HSV conversion using NumPy.
    Replaces the old pure-Python colorsys loop — 10-50× faster.
    
    Args:
        rgb_array: numpy array of shape (N, 3) with values 0-255
    Returns:
        hsv_array: numpy array of shape (N, 3) with H in [0,360], S,V in [0,1]
    """
    r = rgb_array[:, 0].astype(np.float32) / 255.0
    g = rgb_array[:, 1].astype(np.float32) / 255.0
    b = rgb_array[:, 2].astype(np.float32) / 255.0

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Hue
    h = np.zeros_like(max_c)
    mask = delta > 1e-6

    # Red is max
    r_max = (max_c == r) & mask
    h[r_max] = 60 * (((g[r_max] - b[r_max]) / delta[r_max]) % 6)

    # Green is max
    g_max = (max_c == g) & mask
    h[g_max] = 60 * (((b[g_max] - r[g_max]) / delta[g_max]) + 2)

    # Blue is max
    b_max = (max_c == b) & mask
    h[b_max] = 60 * (((r[b_max] - g[b_max]) / delta[b_max]) + 4)

    # Saturation
    s = np.zeros_like(max_c)
    s[mask] = delta[mask] / max_c[mask]

    # Value
    v = max_c

    return np.column_stack([h, s, v])


def _color_name_from_hsv(h, s, v):
    """Classify a single HSV triplet into a human-readable name."""
    if v < 0.15:
        return "near-black"
    if v > 0.90 and s < 0.1:
        return "near-white"
    if s < 0.15:
        return "gray"
    if h < 15 or h > 345:
        return "red"
    if h < 45:
        return "orange"
    if h < 70:
        return "yellow/gold"
    if h < 170:
        return "green"
    if h < 260:
        return "blue"
    if h < 290:
        return "purple/violet"
    if h < 330:
        return "pink/magenta"
    return "red"


def analyze_pixels(image_path):
    """
    Comprehensive pixel-level analysis.
    Returns dict with color, composition, and motion metrics.
    """
    img = Image.open(image_path)

    # Handle alpha channels using shared utility
    from image_utils import load_image_safely
    img = load_image_safely(image_path)

    # Resize for performance (max 512px on longest side)
    w, h = img.size
    if max(w, h) > 512:
        scale = 512 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape

    results = {}

    # ═══════════════════════════════════════════════════════════
    # 1. DOMINANT COLORS (histogram sampling — fast, accurate)
    # ═══════════════════════════════════════════════════════════
    pixels = arr.reshape(-1, 3)

    # Sample pixels (every Nth pixel for large images)
    sample_step = max(1, len(pixels) // 8000)
    sampled = pixels[::sample_step]

    # Histogram binning for dominant colors
    n_bins = 12
    bin_size = 256 // n_bins
    bins = (sampled // bin_size).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)

    # Convert to 1D bin index
    bin_indices = bins[:, 0] * n_bins * n_bins + bins[:, 1] * n_bins + bins[:, 2]
    unique_bins, counts = np.unique(bin_indices, return_counts=True)

    # Get top colors by bin count
    top_indices = np.argsort(-counts)[:8]
    total_count = counts.sum()

    dominant_colors = []
    for idx in top_indices:
        # Decode bin index back to RGB
        b_val = unique_bins[idx] % n_bins
        g_val = (unique_bins[idx] // n_bins) % n_bins
        r_val = unique_bins[idx] // (n_bins * n_bins)

        r = int(r_val * bin_size + bin_size // 2)
        g = int(g_val * bin_size + bin_size // 2)
        b = int(b_val * bin_size + bin_size // 2)

        pct = counts[idx] / total_count * 100

        # HSV for color naming
        hsv = _rgb_to_hsv_vectorized(np.array([[r, g, b]]))[0]
        hue, sat, val = float(hsv[0]), float(hsv[1]), float(hsv[2])
        name = _color_name_from_hsv(hue, sat, val)

        if pct > 1.5:  # Only include significant colors
            dominant_colors.append({
                "name": name,
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "percentage": round(pct, 1),
                "brightness": round(val, 2),
            })

    results["dominant_colors"] = dominant_colors[:6]

    # ═══════════════════════════════════════════════════════════
    # 2. COLOR DIVERSITY / SATURATION / BRIGHTNESS (vectorized)
    # ═══════════════════════════════════════════════════════════
    # Sample every 20th pixel for HSV computation
    sample_step_hsv = max(1, len(pixels) // 4000)
    sampled_pixels = pixels[::sample_step_hsv]

    # Vectorized HSV conversion
    hsv_pixels = _rgb_to_hsv_vectorized(sampled_pixels)
    avg_saturation = float(np.mean(hsv_pixels[:, 1]))
    avg_brightness = float(np.mean(hsv_pixels[:, 2]))

    # Unique color estimate (downsampled to avoid np.unique on full array)
    downsampled = pixels[::max(1, len(pixels) // 10000)]
    unique_colors = len(np.unique(downsampled.astype(np.int32).reshape(-1, 3), axis=0))

    results["color_diversity"] = {
        "unique_colors_est": unique_colors,
        "diversity_ratio": round(unique_colors / min(len(downsampled), 10000) * 100, 1),
        "avg_saturation": round(avg_saturation, 3),
        "avg_brightness": round(avg_brightness, 3),
    }

    # Vibrancy assessment
    if avg_saturation > 0.4:
        results["vibrancy"] = "highly vibrant and saturated"
    elif avg_saturation > 0.2:
        results["vibrancy"] = "moderately colorful"
    elif avg_saturation > 0.08:
        results["vibrancy"] = "subdued and muted"
    else:
        results["vibrancy"] = "near-monochrome or desaturated"

    # ═══════════════════════════════════════════════════════════
    # 3. BRIGHTNESS ANALYSIS (vectorized)
    # ═══════════════════════════════════════════════════════════
    brightness_flat = np.mean(pixels, axis=1)
    brightness_2d = brightness_flat.reshape(h, w)

    dark_pct = float(np.mean(brightness_flat < 40) * 100)
    bright_pct = float(np.mean(brightness_flat > 200) * 100)
    mid_pct = 100 - dark_pct - bright_pct

    results["brightness"] = {
        "mean": round(float(np.mean(brightness_flat)), 1),
        "dark_pct": round(dark_pct, 1),
        "bright_pct": round(bright_pct, 1),
        "mid_pct": round(mid_pct, 1),
    }

    if dark_pct > 60:
        results["brightness_desc"] = "predominantly dark with a deep background"
    elif bright_pct > 60:
        results["brightness_desc"] = "predominantly bright and well-lit"
    elif dark_pct > 30 and bright_pct > 10:
        results["brightness_desc"] = "high contrast with bright elements against a dark background"
    else:
        results["brightness_desc"] = "balanced exposure with mixed light levels"

    # ═══════════════════════════════════════════════════════════
    # 4. RADIAL SYMMETRY (zoom/streak/motion effect detection)
    # ═══════════════════════════════════════════════════════════
    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    norm_dist = dist_from_center / max(max_dist, 1)

    ring_brightness = []
    for ring_radius in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mask = (norm_dist > ring_radius - 0.05) & (norm_dist < ring_radius + 0.05)
        if mask.sum() > 0:
            ring_brightness.append(float(np.mean(brightness_2d[mask])))

    radial_variance = float(np.std(ring_brightness)) if len(ring_brightness) > 1 else 0

    # Angular patterns (streaks)
    angles = np.arctan2(y_coords - center_y, x_coords - center_x)
    angle_bins = 24
    angle_brightness = []
    for i in range(angle_bins):
        angle_min = -np.pi + i * 2 * np.pi / angle_bins
        angle_max = angle_min + 2 * np.pi / angle_bins
        mask = (angles > angle_min) & (angles <= angle_max)
        if mask.sum() > 0:
            angle_brightness.append(float(np.mean(brightness_2d[mask])))

    angular_variance = float(np.std(angle_brightness)) if angle_brightness else 0

    if radial_variance > 15 and angular_variance > 8:
        results["motion_effect"] = "strong radial zoom or streak effect emanating from center"
    elif radial_variance > 8:
        results["motion_effect"] = "noticeable radial pattern suggesting motion or light streaking"
    elif angular_variance > 10:
        results["motion_effect"] = "angular variations suggesting directional light or motion"

    # ═══════════════════════════════════════════════════════════
    # 5. CONTRAST ANALYSIS
    # ═══════════════════════════════════════════════════════════
    brightness_std = float(np.std(brightness_flat))
    results["contrast"] = {
        "std_dev": round(brightness_std, 1),
        "level": ("very high" if brightness_std > 80 else
                  "high" if brightness_std > 50 else
                  "moderate" if brightness_std > 25 else
                  "low")
    }

    return results


def pixel_analysis_to_text(analysis):
    """Convert pixel analysis results to natural language text."""
    parts = []

    if analysis.get("dominant_colors"):
        dc = analysis["dominant_colors"]
        color_list = [f"{c['name']} ({c['hex']}, {c['percentage']:.0f}%)"
                     for c in dc[:5]]
        parts.append(f"Dominant colors: {', '.join(color_list)}.")

    if analysis.get("vibrancy"):
        parts.append(f"The image is {analysis['vibrancy']}.")

    if analysis.get("brightness_desc"):
        parts.append(f"Its exposure is {analysis['brightness_desc']}.")

    if analysis.get("motion_effect"):
        parts.append(f"There is a {analysis['motion_effect']}.")

    if analysis.get("contrast"):
        parts.append(f"Contrast is {analysis['contrast']['level']} (σ={analysis['contrast']['std_dev']}).")

    return " ".join(parts)


if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) > 1:
        result = analyze_pixels(sys.argv[1])
        print(json.dumps(result, indent=2))
        print("\n--- Natural Language ---")
        print(pixel_analysis_to_text(result))
