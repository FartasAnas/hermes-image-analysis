#!/usr/bin/env python3
"""
Engine Configuration — BLIP vs LLaVA Selection
==============================================
Auto-detects GPU and recommends the best vision engine.
Writes user preference to a config file so Hermes can ask
the user once and remember their choice.

Usage (from skill/hermes):
  from engine_config import get_engine_choice
  engine = get_engine_choice()
  # Returns 'llava' or 'blip' based on user preference + GPU availability

Config file: <repo_root>/engine_preference.json
"""
import os, sys, json, torch

CONFIG_FILENAME = "engine_preference.json"

def _get_config_path():
    """Find config file — same directory as this module or repo root."""
    # Try repo root first
    repo_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(repo_root, CONFIG_FILENAME)
    return config_path

def _has_gpu():
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

def _get_gpu_info():
    """Get GPU name and VRAM if available."""
    if not _has_gpu():
        return None, 0
    try:
        import torch
        name = torch.cuda.get_device_name(0) or "NVIDIA GPU"
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        return name, round(vram_gb, 1)
    except:
        return "NVIDIA GPU", 0

def detect_best_engine():
    """
    Detect the best engine based on system capabilities.
    Returns (recommended_engine, reason_string).
    """
    has_gpu = _has_gpu()
    
    if has_gpu:
        return "llava", (
            "GPU detected. LLaVA-1.5-7B is recommended for rich, detailed descriptions "
            "(~4GB VRAM, ~14s inference). BLIP is available as a faster alternative "
            "(0.8s, short captions, works on CPU too)."
        )
    else:
        return "blip", (
            "No GPU detected. BLIP is recommended — fast (0.8s), works on CPU, "
            "short captions. LLaVA requires a GPU with 6GB+ VRAM."
        )

def read_engine_config():
    """Read user's engine preference from config file."""
    config_path = _get_config_path()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return data.get('engine', 'auto')
        except:
            pass
    return 'auto'

def write_engine_config(engine):
    """Write user's engine preference to config file."""
    config_path = _get_config_path()
    data = {
        'engine': engine,
        'description': 'Vision engine preference for image analysis pipeline.',
        'options': {
            'llava': 'LLaVA-1.5-7B (4-bit GPU) — rich multi-paragraph descriptions, ~4GB VRAM',
            'blip': 'BLIP-base — fast short captions, works on CPU, ~1GB model',
            'auto': 'Auto-detect based on GPU availability'
        }
    }
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_engine_choice(force=None):
    """
    Get the engine to use. Resolution order:
    1. force= parameter (command-line override)
    2. Config file (user preference)
    3. Auto-detect (GPU-based recommendation)
    
    Returns 'llava' or 'blip'.
    """
    if force and force in ('llava', 'blip'):
        return force
    
    config_choice = read_engine_config()
    if config_choice in ('llava', 'blip'):
        return config_choice
    
    # Auto-detect
    recommended, reason = detect_best_engine()
    return recommended

def print_engine_recommendation():
    """Print a user-friendly engine recommendation (for Hermes to display)."""
    recommended, reason = detect_best_engine()
    has_gpu = _has_gpu()
    
    print("\n" + "=" * 60)
    print("  🖥️  VISION ENGINE SELECTION")
    print("=" * 60)
    
    if has_gpu:
        print(f"  ✅ CUDA GPU detected")
        print()
        print(f"  📌 Recommended: LLaVA-1.5-7B (4-bit)")
        print(f"     • Rich, detailed multi-paragraph descriptions")
        print(f"     • Can identify colors, objects, spatial relationships, motion")
        print(f"     • ~4GB VRAM usage, ~14s per image")
        print()
        print(f"  ⚡ Alternative: BLIP-base")
        print(f"     • Fast, short captions (5-15 words)")
        print(f"     • Works on CPU, ~1GB model, 0.8s per image")
        print(f"     • Good for: quick scanning, OCR-focused workloads, low-resource machines")
    else:
        print(f"  ⚠️  No CUDA GPU detected")
        print()
        print(f"  📌 Recommended: BLIP-base")
        print(f"     • Fast, short captions (5-15 words)")
        print(f"     • Works on CPU, ~1GB model, 0.8s per image")
        print()
        print(f"  ⚡ LLaVA requires GPU — not available on this machine")
    
    print()
    print(f"  💡 To select an engine:")
    print(f"     python analyze_image.py <image> --engine llava")
    print(f"     python analyze_image.py <image> --engine blip")
    print(f"     python analyze_image.py <image> --engine auto  (default)")
    print(f"  💾 Config file: engine_preference.json")
    print("=" * 60)

if __name__ == '__main__':
    print_engine_recommendation()
