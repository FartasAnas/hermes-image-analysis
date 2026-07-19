#!/usr/bin/env python3
"""
Shared configuration for Hermes Image Analysis tools.
Handles:
  - Dynamic drive/path detection (no hardcoded E: drive)
  - GPU availability detection
  - Environment variable setup for cache directories

Usage:
  from hermes_config import get_storage_drive, setup_environment, gpu_available
  drive = get_storage_drive()  # e.g., 'E:' or '/mnt/data'
"""

import os, sys, platform

# ═══════════════════════════════════════════════════════════════
# DRIVE / STORAGE PATH DETECTION
# ═══════════════════════════════════════════════════════════════

def _get_available_drives_windows():
    """Get list of available drive letters on Windows (excluding C:)."""
    import string
    available = []
    for letter in string.ascii_uppercase[2:]:  # D: through Z:
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            available.append(f"{letter}:")
    return available

def _get_available_paths_unix():
    """Get list of large mount points on Linux/Mac."""
    candidates = ['/mnt', '/media', '/Volumes', os.path.expanduser('~')]
    available = []
    for path in candidates:
        if os.path.exists(path) and os.path.isdir(path):
            available.append(path)
    return available

def _prompt_user_for_drive(available):
    """Prompt user to select a storage drive."""
    print("\n" + "=" * 60)
    print("  Hermes Image Analysis — Storage Configuration")
    print("=" * 60)
    print(f"\n  Available storage locations:")
    for i, drive in enumerate(available):
        print(f"    [{i+1}] {drive}")
    print(f"    [{len(available)+1}] Custom path")
    
    while True:
        try:
            choice = input(f"\n  Select [1-{len(available)+1}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                return available[idx]
            elif idx == len(available):
                custom = input("  Enter custom path: ").strip()
                if os.path.exists(custom):
                    return custom
                print(f"  Path '{custom}' does not exist.")
            else:
                print(f"  Invalid choice.")
        except (ValueError, KeyboardInterrupt):
            print("\n  Using home directory as fallback.")
            return os.path.expanduser('~')

def get_storage_drive(auto=True):
    """
    Returns the storage drive/path to use.
    
    On Windows: prefers D:, then E:, then prompts.
    On Unix: prefers /mnt/data, then /media, then ~/hermes_tools.
    
    Set auto=False to always prompt the user.
    """
    system = platform.system()
    
    if system == 'Windows':
        available = _get_available_drives_windows()
        
        if not available:
            print("Warning: No non-C: drives found. Using C: as fallback.")
            return "C:"
        
        # Auto-select: prefer E:, then D:, then largest
        if auto:
            for preferred in ['E:', 'D:', 'F:', 'G:']:
                if preferred in available:
                    return preferred
            return available[0]
        else:
            return _prompt_user_for_drive(available)
    
    else:  # Linux / Mac
        home = os.path.expanduser('~')
        available = _get_available_paths_unix()
        
        if auto:
            for path in available:
                if path != home and os.path.exists(path):
                    # Check if writable
                    test_path = os.path.join(path, '.hermes_test')
                    try:
                        with open(test_path, 'w') as f:
                            f.write('test')
                        os.remove(test_path)
                        return path
                    except:
                        pass
            # Fallback to home
            hermes_dir = os.path.join(home, 'hermes_tools')
            os.makedirs(hermes_dir, exist_ok=True)
            return hermes_dir
        else:
            return _prompt_user_for_drive(available + [home])

def get_base_path(drive=None):
    """
    Returns the base path: {drive}/hermes_tools/
    On Windows: E:/hermes_tools/
    On Unix: /mnt/data/hermes_tools/ or ~/hermes_tools/
    """
    if drive is None:
        drive = get_storage_drive()
    
    system = platform.system()
    if system == 'Windows':
        # Windows: D:/hermes_tools/
        if ':' in drive:
            return f"{drive}/hermes_tools/"
        else:
            return f"{drive}/hermes_tools/"
    else:
        # Unix: /path/hermes_tools/
        if drive.endswith('/'):
            return f"{drive}hermes_tools/"
        else:
            return f"{drive}/hermes_tools/"

def get_cache_dir(drive=None):
    """Returns the cache directory path."""
    return os.path.join(get_base_path(drive), 'cache')

def get_temp_dir(drive=None):
    """Returns the temp directory path."""
    return os.path.join(get_base_path(drive), 'temp')

def get_scripts_dir(drive=None):
    """Returns the scripts directory path."""
    return os.path.join(get_base_path(drive), 'scripts')

def get_config_dir(drive=None):
    """Returns the config directory path."""
    return os.path.join(get_base_path(drive), 'config')

# ═══════════════════════════════════════════════════════════════
# GPU DETECTION
# ═══════════════════════════════════════════════════════════════

def gpu_available():
    """
    Check if a CUDA-capable GPU is available.
    Returns: (has_gpu: bool, device: str, gpu_name: str)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except:
                gpu_name = "Unknown NVIDIA GPU"
            return True, device, gpu_name
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return True, "mps", "Apple Silicon GPU"
        
        return False, "cpu", "CPU only"
    except ImportError:
        return False, "cpu", "torch not installed"

def gpu_info_str():
    """Return a human-readable GPU status string."""
    has_gpu, device, name = gpu_available()
    if has_gpu:
        return f"✅ GPU available: {name} (device: {device})"
    return f"⚠️  No GPU detected — using CPU"

# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════

def setup_environment(drive=None):
    """
    Set all required environment variables for the pipeline.
    Creates directories if needed.
    Returns dict of paths.
    """
    if drive is None:
        drive = get_storage_drive()
    
    base = get_base_path(drive)
    cache = get_cache_dir(drive)
    temp = get_temp_dir(drive)
    
    # Convert Windows paths to proper format for env vars
    if platform.system() == 'Windows':
        # E:/hermes_tools/ format
        hf_home = f"{base}.hf".replace('\\', '/')
        cache_base = cache.replace('\\', '/')
    else:
        hf_home = os.path.join(base, '.hf')
        cache_base = cache
    
    # Set env vars
    os.environ.setdefault('TMP', temp)
    os.environ.setdefault('TEMP', temp)
    os.environ.setdefault('TMPDIR', temp)
    os.environ.setdefault('HF_HOME', hf_home)
    os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(hf_home, 'hub'))
    os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(hf_home, 'hub'))
    os.environ.setdefault('DOCTR_CACHE_DIR', os.path.join(cache_base, 'doctr'))
    os.environ.setdefault('EASYOCR_MODULE_PATH', os.path.join(cache_base, 'easyocr'))
    os.environ.setdefault('XDG_CACHE_HOME', cache_base)
    os.environ.setdefault('TORCH_HOME', os.path.join(cache_base, 'torch'))
    
    # Create directories
    for d in [cache, temp, 
              os.path.join(cache, 'doctr'),
              os.path.join(cache, 'easyocr'),
              os.path.join(cache, 'torch'),
              hf_home]:
        os.makedirs(d, exist_ok=True)
    
    return {
        'drive': drive,
        'base': base,
        'cache': cache,
        'temp': temp,
        'hf_home': hf_home,
        'gpu': gpu_available(),
    }

# ═══════════════════════════════════════════════════════════════
# AUTO-CONFIGURE ON IMPORT
# ═══════════════════════════════════════════════════════════════

_CONFIG = None

def get_config():
    """Get or create the global config (lazy init)."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = setup_environment()
    return _CONFIG

# ═══════════════════════════════════════════════════════════════
# MAIN (if run directly, shows config)
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    drive = get_storage_drive(auto=False)  # Prompt user
    config = setup_environment(drive)
    
    print("\n" + "=" * 60)
    print("  Hermes Image Analysis — Configuration")
    print("=" * 60)
    print(f"\n  Storage Drive:   {drive}")
    print(f"  Base Path:       {config['base']}")
    print(f"  Cache Dir:       {config['cache']}")
    print(f"  Temp Dir:        {config['temp']}")
    print(f"  HF Cache:        {config['hf_home']}")
    print(f"\n  GPU Status:      {gpu_info_str()}")
    print(f"\n  Environment variables set:")
    for key in ['HF_HOME', 'DOCTR_CACHE_DIR', 'EASYOCR_MODULE_PATH', 
                'XDG_CACHE_HOME', 'TORCH_HOME', 'TMP', 'TEMP']:
        print(f"    {key}={os.environ.get(key, 'NOT SET')}")
    print(f"\n{'=' * 60}")
