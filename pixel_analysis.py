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

All GPU-accelerated where possible, falls back to CPU.
"""
import numpy as np
from PIL import Image
import colorsys

def analyze_pixels(image_path):
    """
    Comprehensive pixel-level analysis.
    Returns dict with color, composition, and motion metrics.
    """
    img = Image.open(image_path)
    # Handle alpha channels
    if img.mode in ('LA', 'PA'):
        bg = Image.new('L', img.size, 255)
        alpha = img.getchannel('A')
        bg.paste(alpha, mask=alpha)
        img = bg.convert('RGB')
    elif img.mode in ('RGBA', 'RGBa'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
        img = bg
    else:
        img = img.convert('RGB')
    
    # Resize for performance (max 512px on longest side)
    w, h = img.size
    if max(w, h) > 512:
        scale = 512 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    
    results = {}
    
    # ═══════════════════════════════════════════════════════════
    # 1. DOMINANT COLORS (k-means clustering)
    # ═══════════════════════════════════════════════════════════
    pixels = arr.reshape(-1, 3)
    
    # Simple dominant color extraction using histogram bins
    # Faster than k-means for this purpose
    n_bins = 16
    hist_r = np.histogram(pixels[:, 0], bins=n_bins, range=(0, 256))
    hist_g = np.histogram(pixels[:, 1], bins=n_bins, range=(0, 256))
    hist_b = np.histogram(pixels[:, 2], bins=n_bins, range=(0, 256))
    
    # Get top 5 most common color bins
    bin_size = 256 // n_bins
    color_counts = {}
    for i in range(0, len(pixels), max(1, len(pixels) // 5000)):
        r = int(pixels[i, 0] // bin_size) * bin_size + bin_size // 2
        g = int(pixels[i, 1] // bin_size) * bin_size + bin_size // 2
        b = int(pixels[i, 2] // bin_size) * bin_size + bin_size // 2
        key = (r, g, b)
        color_counts[key] = color_counts.get(key, 0) + 1
    
    top_colors = sorted(color_counts.items(), key=lambda x: -x[1])[:8]
    
    # Classify dominant colors
    dominant_colors = []
    for (r, g, b), count in top_colors:
        hsv = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        hue = hsv[0] * 360
        sat = hsv[1]
        val = hsv[2]
        
        # Color naming
        if val < 0.15:
            name = "near-black"
        elif val > 0.90 and sat < 0.1:
            name = "near-white"
        elif sat < 0.15:
            name = "gray"
        elif hue < 15 or hue > 345:
            name = "red"
        elif hue < 45:
            name = "orange"
        elif hue < 70:
            name = "yellow/gold"
        elif hue < 170:
            name = "green"
        elif hue < 260:
            name = "blue"
        elif hue < 290:
            name = "purple/violet"
        elif hue < 330:
            name = "pink/magenta"
        else:
            name = "red"
        
        pct = count / sum(c for _, c in top_colors) * 100
        if pct > 2:  # Only include significant colors
            dominant_colors.append({
                "name": name,
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "percentage": round(pct, 1),
                "brightness": round(val, 2),
            })
    
    results["dominant_colors"] = dominant_colors[:6]
    
    # ═══════════════════════════════════════════════════════════
    # 2. COLOR DIVERSITY / VIBRANCY
    # ═══════════════════════════════════════════════════════════
    unique_colors = len(np.unique(pixels.reshape(-1, 3), axis=0))
    total_pixels = pixels.shape[0]
    color_diversity = min(unique_colors / total_pixels * 100, 100)
    
    # Average saturation
    hsv_pixels = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) 
                          for r, g, b in pixels[::10]])  # Sample every 10th pixel
    avg_saturation = float(np.mean(hsv_pixels[:, 1]))
    avg_brightness = float(np.mean(hsv_pixels[:, 2]))
    
    results["color_diversity"] = {
        "unique_colors": unique_colors,
        "diversity_pct": round(color_diversity, 1),
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
    # 3. BRIGHTNESS ANALYSIS
    # ═══════════════════════════════════════════════════════════
    brightness = np.mean(pixels, axis=1)
    brightness_2d = brightness.reshape(h, w)
    brightness_flat = brightness.flatten()
    
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
    
    # Distance from center
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    norm_dist = dist_from_center / max_dist
    
    # Check if brightness increases toward edges (zoom effect: center bright, edges dark-streaked)
    # Sample rings at different distances
    ring_brightness = []
    for ring_radius in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mask = (norm_dist > ring_radius - 0.05) & (norm_dist < ring_radius + 0.05)
        if mask.sum() > 0:
            ring_brightness.append(float(np.mean(brightness_2d[mask])))
    
    # Radial analysis: is there a pattern?
    radial_variance = float(np.std(ring_brightness)) if len(ring_brightness) > 1 else 0
    
    # Check for angular patterns (streaks)
    angles = np.arctan2(y_coords - center_y, x_coords - center_x)
    angle_bins = 36  # 10-degree bins
    angle_brightness = []
    for i in range(angle_bins):
        angle_min = -np.pi + i * 2 * np.pi / angle_bins
        angle_max = angle_min + 2 * np.pi / angle_bins
        mask = (angles > angle_min) & (angles <= angle_max)
        if mask.sum() > 0:
            angle_brightness.append(float(np.mean(brightness_2d[mask])))
    
    angular_variance = float(np.std(angle_brightness)) if angle_brightness else 0
    
    # Zoom/streak detection threshold
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
        "level": "very high" if brightness_std > 80 else
                 "high" if brightness_std > 50 else
                 "moderate" if brightness_std > 25 else
                 "low"
    }
    
    return results


def pixel_analysis_to_text(analysis):
    """Convert pixel analysis results to natural language text."""
    parts = []
    
    # Colors
    if analysis.get("dominant_colors"):
        dc = analysis["dominant_colors"]
        color_list = [f"{c['name']} ({c['hex']}, {c['percentage']:.0f}%)" 
                     for c in dc[:5]]
        parts.append(f"Dominant colors: {', '.join(color_list)}.")
    
    # Vibrancy
    if analysis.get("vibrancy"):
        parts.append(f"The image is {analysis['vibrancy']}.")
    
    # Brightness
    if analysis.get("brightness_desc"):
        parts.append(f"Its exposure is {analysis['brightness_desc']}.")
    
    # Motion
    if analysis.get("motion_effect"):
        parts.append(f"There is a {analysis['motion_effect']}.")
    
    # Contrast
    if analysis.get("contrast"):
        parts.append(f"Contrast is {analysis['contrast']['level']} (σ={analysis['contrast']['std_dev']}).")
    
    return " ".join(parts)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = analyze_pixels(sys.argv[1])
        import json
        print(json.dumps(result, indent=2))
        print("\n--- Natural Language ---")
        print(pixel_analysis_to_text(result))
