#!/usr/bin/env python3
"""
Engine Configuration — BLIP vs LLaVA Selection with Interactive Prompting
=========================================================================
Auto-detects GPU and recommends the best vision engine.
Prompts the user ONCE for their preference ("Short or Detailed?") then
persists it to engine_preference.json for all future sessions.

Usage (from skill/hermes):
  from engine_config import get_engine_choice, prompt_for_engine
  engine = prompt_for_engine()  # Interactive: prompts once, remembers forever
  engine = get_engine_choice()  # Non-interactive: config → auto-detect

Config file: <repo_root>/engine_preference.json
"""
import os
import sys
import json


CONFIG_FILENAME = "engine_preference.json"


def _get_config_path():
    """Find config file in repo root."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(repo_root, CONFIG_FILENAME)


def _has_gpu():
    """Check if CUDA GPU is available. Lazy import to avoid loading torch unnecessarily."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
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
    except Exception:
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
    """
    Read user's engine preference from config file.
    Returns 'blip', 'llava', or 'auto' (no preference saved).
    """
    config_path = _get_config_path()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            engine = data.get('engine', 'auto')
            if engine in ('blip', 'llava'):
                return engine
        except (json.JSONDecodeError, IOError):
            pass
    return 'auto'


def write_engine_config(engine, source='user'):
    """
    Write user's engine preference to config file.

    Args:
        engine: 'blip' or 'llava'
        source: 'user' (interactive prompt), 'cli' (--engine flag), 'auto' (GPU detect)
    """
    import datetime
    config_path = _get_config_path()

    data = {
        'engine': engine,
        'source': source,
        'saved_at': datetime.datetime.now().isoformat(),
        'description': 'Vision engine preference for image analysis pipeline.',
        'options': {
            'blip': 'BLIP-base — fast short captions (0.8s), works on CPU/GPU, ~1GB model',
            'llava': 'LLaVA-1.5-7B (4-bit GPU) — rich multi-paragraph descriptions, ~4GB VRAM',
        },
        'override_hint': 'Use --engine blip|llava to override this preference temporarily.',
        'reset_hint': 'Delete this file or set engine to "auto" to be re-prompted.',
    }
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=2)


def prompt_user_for_engine():
    """
    INTERACTIVE: Prompt the user to choose between short (BLIP) and detailed (LLaVA).

    This is the main entry point for Phase 1 — it asks ONCE and remembers forever.
    Called when no saved preference exists in engine_preference.json.

    Returns:
        'blip' or 'llava'
    """
    has_gpu = _has_gpu()
    gpu_name, vram_gb = _get_gpu_info()

    print()
    print("=" * 64)
    print("  \U0001f916 Hermes Image Analysis — Engine Selection")
    print("=" * 64)
    print()

    if has_gpu:
        print(f"  \u2705 GPU detected: {gpu_name} ({vram_gb} GB VRAM)")
    else:
        print(f"  \u26a0\ufe0f  No GPU detected — running on CPU")

    print()
    print("  Would you like a short or detailed description?")
    print()
    print("    [1] \u26a1 SHORT  — BLIP-base (fast, 0.8s, short captions)")
    print("        \u2514 Works on CPU and GPU. Good for quick scanning.")
    print()

    if has_gpu:
        print("    [2] \U0001f50d DETAILED — LLaVA-1.5-7B (rich, ~14s, multi-paragraph)")
        print("        \u2514 GPU only. Identifies colors, objects, spatial relationships, mood.")
        print()
        print("  \U0001f4a1 Your preference will be saved and used for all future images.")
        print("     Use --engine blip or --engine llava to override per-run.")
        print()

        while True:
            try:
                choice = input("  Select [1-2] (default: 1 for short): ").strip()
                if choice == '' or choice == '1':
                    engine = 'blip'
                    break
                elif choice == '2':
                    engine = 'llava'
                    break
                else:
                    print("  Please enter 1 or 2.")
            except (KeyboardInterrupt, EOFError):
                print("\n  Using default: BLIP (short descriptions)")
                engine = 'blip'
                break
    else:
        print("    [2] DETAILED — LLaVA-1.5-7B \u274c UNAVAILABLE (needs GPU)")
        print()
        print("  \u2139\ufe0f  Only BLIP is available on this machine (no GPU detected).")
        print("     Using BLIP for fast short captions.")
        engine = 'blip'

    # Persist the preference
    write_engine_config(engine, source='user')

    engine_label = "\u26a1 BLIP (short)" if engine == "blip" else "\U0001f50d LLaVA (detailed)"
    print(f"\n  \u2705 Preference saved: {engine_label}")
    print(f"     Config: {_get_config_path()}")
    print(f"     This preference will be used for ALL future image analysis.")
    print(f"{'=' * 64}")
    print()

    return engine


def get_engine_choice(force=None):
    """
    Get the engine to use. Resolution order:
    1. force= parameter (command-line override)
    2. Config file (user preference from engine_preference.json)
    3. Auto-detect (GPU-based recommendation)

    This is the NON-interactive path — does NOT prompt the user.
    Use prompt_for_engine() for the interactive first-run experience.

    Returns 'blip' or 'llava'.
    """
    if force and force in ('llava', 'blip'):
        return force

    config_choice = read_engine_config()
    if config_choice in ('blip', 'llava'):
        return config_choice

    # Auto-detect
    recommended, _reason = detect_best_engine()
    return recommended


def get_or_prompt_engine(force=None, interactive=True):
    """
    Get engine with optional interactive prompting on first run.

    This is the PRIMARY entry point for Phase 1 behavior:
    1. If --engine flag provided → use it (no prompt, no save)
    2. If saved preference exists → use it (no prompt)
    3. If no preference and interactive=True → PROMPT user, save, return
    4. If no preference and interactive=False → auto-detect

    Args:
        force: CLI override ('blip', 'llava', or None)
        interactive: If True and no saved preference, prompt the user.

    Returns:
        'blip' or 'llava'
    """
    # CLI override takes priority — always
    if force and force in ('blip', 'llava'):
        return force

    # Check for saved preference
    saved = read_engine_config()
    if saved in ('blip', 'llava'):
        return saved

    # No saved preference — prompt or auto-detect
    if interactive:
        return prompt_user_for_engine()

    # Non-interactive fallback
    recommended, _reason = detect_best_engine()
    return recommended


def reset_engine_preference():
    """
    Delete the saved preference so the user is re-prompted on next run.
    """
    config_path = _get_config_path()
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"  \u267b\ufe0f  Engine preference reset. You will be prompted on next run.")
        return True
    return False


def print_engine_recommendation():
    """Print a user-friendly engine recommendation (for --show-engines)."""
    recommended, reason = detect_best_engine()
    has_gpu = _has_gpu()

    print("\n" + "=" * 60)
    print("  \U0001f5a5\ufe0f  VISION ENGINE SELECTION")
    print("=" * 60)

    if has_gpu:
        gpu_name, vram_gb = _get_gpu_info()
        print(f"  \u2705 CUDA GPU detected: {gpu_name} ({vram_gb} GB VRAM)")
        print()
        print(f"  \U0001f4cc Recommended: LLaVA-1.5-7B (4-bit)")
        print(f"     \u2022 Rich, detailed multi-paragraph descriptions")
        print(f"     \u2022 Can identify colors, objects, spatial relationships, motion")
        print(f"     \u2022 ~4GB VRAM usage, ~14s per image")
        print()
        print(f"  \u26a1 Alternative: BLIP-base")
        print(f"     \u2022 Fast, short captions (5-15 words)")
        print(f"     \u2022 Works on CPU, ~1GB model, 0.8s per image")
        print(f"     \u2022 Good for: quick scanning, OCR-focused workloads, low-resource machines")
    else:
        print(f"  \u26a0\ufe0f  No CUDA GPU detected")
        print()
        print(f"  \U0001f4cc Recommended: BLIP-base")
        print(f"     \u2022 Fast, short captions (5-15 words)")
        print(f"     \u2022 Works on CPU, ~1GB model, 0.8s per image")
        print()
        print(f"  \u26a1 LLaVA requires GPU — not available on this machine")

    # Show saved preference if any
    saved = read_engine_config()
    if saved in ('blip', 'llava'):
        config_path = _get_config_path()
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            saved_at = data.get('saved_at', 'unknown date')
            source = data.get('source', 'unknown')
        except Exception:
            saved_at = 'unknown date'
            source = 'unknown'
        print()
        saved_label = "\u26a1 BLIP (short)" if saved == "blip" else "\U0001f50d LLaVA (detailed)"
        print(f"  \U0001f4be Saved preference: {saved_label}")
        print(f"     Set on: {saved_at} (via: {source})")

    print()
    print(f"  \U0001f4a1 To select an engine:")
    print(f"     python analyze_image.py <image> --engine llava")
    print(f"     python analyze_image.py <image> --engine blip")
    print(f"     python analyze_image.py <image>              (uses saved preference)")
    print(f"  \U0001f4be Config file: engine_preference.json")
    print(f"  \u267b\ufe0f  Reset: delete engine_preference.json to be re-prompted")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Engine configuration tool')
    parser.add_argument('--show', action='store_true', help='Show engine recommendation')
    parser.add_argument('--reset', action='store_true', help='Reset saved preference')
    parser.add_argument('--prompt', action='store_true', help='Force re-prompt for engine choice')
    args = parser.parse_args()

    if args.reset:
        reset_engine_preference()
    elif args.prompt:
        prompt_user_for_engine()
    else:
        print_engine_recommendation()
