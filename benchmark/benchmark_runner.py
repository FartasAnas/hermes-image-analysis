#!/usr/bin/env python3
"""
Hermes Image Analysis — Benchmark Runner
=========================================
Runs the full pipeline on test images and records:
  - Latency (vision, OCR, total)
  - OCR confidence
  - Classification output
  - Cross-engine flags
  - Resource metadata

Usage:
  python benchmark_runner.py                    # Run with built-in test images
  python benchmark_runner.py --images <dir>     # Run on custom image directory
  python benchmark_runner.py --engine blip      # Force BLIP
  python benchmark_runner.py --engine llava     # Force LLaVA
  python benchmark_runner.py --compare          # Compare BLIP vs LLaVA
"""

import os
import sys
import json
import time
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def generate_test_images(output_dir, count=4):
    """Generate simple synthetic test images for benchmarking."""
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # 1. Solid color image (red gradient)
    img = Image.new("RGB", (400, 300), (200, 50, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 350, 250], fill=(50, 50, 200))
    draw.text((150, 140), "Test Image", fill=(255, 255, 255))
    path = os.path.join(output_dir, "test_solid.png")
    img.save(path)
    paths.append(path)

    # 2. Dark image
    img = Image.new("RGB", (400, 300), (10, 10, 10))
    draw = ImageDraw.Draw(img)
    draw.text((100, 140), "Dark Mode", fill=(200, 200, 200))
    path = os.path.join(output_dir, "test_dark.png")
    img.save(path)
    paths.append(path)

    # 3. Colorful image (horizontal stripes)
    img = Image.new("RGB", (400, 300))
    colors = [
        (255, 0, 0), (255, 165, 0), (255, 255, 0),
        (0, 255, 0), (0, 0, 255), (128, 0, 128),
    ]
    for i, color in enumerate(colors):
        y_start = i * 50
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, y_start, 400, y_start + 50], fill=color)
    path = os.path.join(output_dir, "test_colorful.png")
    img.save(path)
    paths.append(path)

    # 4. Small image
    img = Image.new("RGB", (50, 50), (100, 200, 100))
    path = os.path.join(output_dir, "test_tiny.png")
    img.save(path)
    paths.append(path)

    return paths


def run_single_benchmark(image_path, engine="blip"):
    """Run full pipeline on one image. Returns metrics dict."""
    from analyze_image import run_vision, run_doctr, analyze_metadata
    from pixel_analysis import analyze_pixels
    from describe_engine import build_state

    t_start = time.time()

    # Metadata
    meta = analyze_metadata(image_path)

    # Vision
    t_vision_start = time.time()
    try:
        vision = run_vision(image_path, engine=engine)
        vision_time = vision.get("time_seconds", 0)
        caption = vision.get("caption", "")
        vram = vision.get("vram_gb", 0)
    except Exception as e:
        vision_time = -1
        caption = f"ERROR: {e}"
        vram = 0
        vision = None

    # OCR
    t_ocr_start = time.time()
    try:
        ocr = run_doctr(image_path, force_cpu=False)
        ocr_time = ocr.get("time_seconds", 0)
        ocr_words = ocr.get("word_count", 0)
        ocr_conf = ocr.get("avg_confidence", 0)
    except Exception as e:
        ocr_time = -1
        ocr_words = 0
        ocr_conf = 0
        ocr = None

    # Pixel
    try:
        pixel = analyze_pixels(image_path)
    except Exception:
        pixel = None

    # Build unified state
    state = build_state(
        meta=meta,
        vision_result=vision,
        ocr_result=ocr,
        pixel_result=pixel,
        engine_used=engine,
    )

    total_time = round(time.time() - t_start, 2)

    return {
        "image": os.path.basename(image_path),
        "dimensions": meta.get("dimensions", "?"),
        "file_kb": meta.get("file_size_kb", 0),
        "engine": engine,
        "vision_time_s": vision_time,
        "ocr_time_s": ocr_time,
        "total_time_s": total_time,
        "vram_gb": vram,
        "ocr_words": ocr_words,
        "ocr_confidence": ocr_conf,
        "caption": caption[:200] if caption else "",
        "is_tech": state.get("flags", {}).get("technical_screenshot", False),
        "vibrancy": pixel.get("vibrancy", "") if pixel else "",
        "contrast": pixel.get("contrast", {}).get("level", "") if pixel else "",
    }


def run_benchmark_suite(image_paths, engines=None):
    """Run benchmarks across multiple images and engines."""
    if engines is None:
        engines = ["blip"]

    results = []
    for path in image_paths:
        for engine in engines:
            print(f"  Benchmarking {os.path.basename(path)} [{engine}]...", end=" ", flush=True)
            result = run_single_benchmark(path, engine)
            results.append(result)
            status = "✓" if result["vision_time_s"] > 0 else "✗"
            print(f"{status} {result['total_time_s']}s total")

    return results


def print_summary(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)

    blip_results = [r for r in results if r["engine"] == "blip"]
    llava_results = [r for r in results if r["engine"] == "llava"]

    if blip_results:
        avg_vision = sum(r["vision_time_s"] for r in blip_results if r["vision_time_s"] > 0)
        n = max(1, len([r for r in blip_results if r["vision_time_s"] > 0]))
        avg_total = sum(r["total_time_s"] for r in blip_results) / max(1, len(blip_results))
        print(f"\n  BLIP ({len(blip_results)} images):")
        print(f"    Avg vision time:  {avg_vision/n:.2f}s")
        print(f"    Avg total time:   {avg_total:.2f}s")

    if llava_results:
        avg_vision = sum(r["vision_time_s"] for r in llava_results if r["vision_time_s"] > 0)
        n = max(1, len([r for r in llava_results if r["vision_time_s"] > 0]))
        avg_total = sum(r["total_time_s"] for r in llava_results) / max(1, len(llava_results))
        avg_vram = sum(r["vram_gb"] for r in llava_results) / max(1, len(llava_results))
        print(f"\n  LLaVA ({len(llava_results)} images):")
        print(f"    Avg vision time:  {avg_vision/n:.2f}s")
        print(f"    Avg total time:   {avg_total:.2f}s")
        print(f"    Avg VRAM usage:   {avg_vram:.2f} GB")

    print(f"\n  {'─' * 60}")
    print(f"  {'Image':<25} {'Engine':<8} {'Vision':<8} {'OCR':<8} {'Total':<8} {'VRAM':<8}")
    print(f"  {'─' * 60}")
    for r in results:
        v_time = f"{r['vision_time_s']:.2f}s" if r['vision_time_s'] > 0 else "FAIL"
        o_time = f"{r['ocr_time_s']:.2f}s" if r['ocr_time_s'] > 0 else "FAIL"
        t_time = f"{r['total_time_s']:.2f}s"
        vram = f"{r['vram_gb']:.1f}GB"
        print(f"  {r['image']:<25} {r['engine']:<8} {v_time:<8} {o_time:<8} {t_time:<8} {vram:<8}")

    print(f"  {'─' * 60}")
    print()


def save_results(results, output_path):
    """Save benchmark results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hermes Image Analysis Benchmark Runner")
    parser.add_argument("--images", help="Directory of images to benchmark")
    parser.add_argument("--engine", default="blip", choices=["blip", "llava", "both"])
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--compare", action="store_true", help="Compare BLIP vs LLaVA")
    args = parser.parse_args()

    # Generate or load images
    if args.images:
        image_dir = args.images
        image_paths = sorted(
            str(p) for p in Path(image_dir).glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
        )
        if not image_paths:
            print(f"No images found in {image_dir}")
            sys.exit(1)
        print(f"Found {len(image_paths)} images in {image_dir}")
    else:
        temp_dir = os.path.join(REPO_ROOT, "benchmark", "test_images")
        image_paths = generate_test_images(temp_dir)
        print(f"Generated {len(image_paths)} test images in {temp_dir}")

    # Determine engines
    engines = ["blip"]
    if args.compare or args.engine == "both":
        engines = ["blip", "llava"]
    elif args.engine == "llava":
        engines = ["llava"]

    # Run benchmarks
    print(f"\nRunning benchmarks with {len(engines)} engine(s) on {len(image_paths)} image(s)...\n")
    results = run_benchmark_suite(image_paths, engines)

    # Print summary
    print_summary(results)

    # Save
    if args.output:
        save_results(results, args.output)
    else:
        default_output = os.path.join(REPO_ROOT, "benchmark", "results.json")
        save_results(results, default_output)
