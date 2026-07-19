#!/usr/bin/env python3
"""
Engine Configuration — BLIP vs LLaVA Selection with Scope-Aware Prompting
==========================================================================
Auto-detects GPU, prompts user for engine + retention scope, and persists
accordingly. Three scope levels:

  photo     — Ephemeral: this image only, no persistence, ask again next time
  session   — Volatile: saved to .hermes_session, resets when terminal closes
  permanent — Persistent: saved to engine_preference.json, never ask again

Resolution order: CLI override → session state → permanent config → prompt → auto-detect

Usage:
  from engine_config import get_or_prompt_engine
  engine = get_or_prompt_engine()                 # full resolution
  engine = get_or_prompt_engine(force='llava')    # CLI override
  engine = get_or_prompt_engine(interactive=False) # no prompt, just resolve
"""
import os
import sys
import json

# ── Config filenames (relative to repo root) ──

PERMANENT_CONFIG = "engine_preference.json"
SESSION_STATE_FILE = ".hermes_session"


def _get_repo_root():
    """Absolute path to the repo root (where this file lives)."""
    return os.path.dirname(os.path.abspath(__file__))


def _get_permanent_config_path():
    return os.path.join(_get_repo_root(), PERMANENT_CONFIG)


def _get_session_state_path():
    return os.path.join(_get_repo_root(), SESSION_STATE_FILE)


# ═══════════════════════════════════════════════════════════════
# GPU DETECTION (lazy torch import)
# ═══════════════════════════════════════════════════════════════

def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _get_gpu_info():
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
    """Returns (recommended_engine, reason_string)."""
    if _has_gpu():
        return "llava", (
            "GPU detected. LLaVA-1.5-7B is recommended for rich, detailed descriptions "
            "(~4GB VRAM, ~14s inference). BLIP is available as a faster alternative "
            "(0.8s, short captions, works on CPU too)."
        )
    return "blip", (
        "No GPU detected. BLIP is recommended — fast (0.8s), works on CPU, "
        "short captions. LLaVA requires a GPU with 6GB+ VRAM."
    )


# ═══════════════════════════════════════════════════════════════
# PERMANENT CONFIG (engine_preference.json)
# ═══════════════════════════════════════════════════════════════

def read_permanent_config():
    """Returns 'blip', 'llava', or 'auto' (no permanent pref saved)."""
    path = _get_permanent_config_path()
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            engine = data.get('engine', 'auto')
            if engine in ('blip', 'llava'):
                return engine
        except (json.JSONDecodeError, IOError):
            pass
    return 'auto'


def write_permanent_config(engine, source='user'):
    """Persist engine choice to engine_preference.json."""
    import datetime
    path = _get_permanent_config_path()
    data = {
        'engine': engine,
        'source': source,
        'saved_at': datetime.datetime.now().isoformat(),
        'scope': 'permanent',
        'description': 'Vision engine preference for image analysis pipeline.',
        'options': {
            'blip': 'BLIP-base — fast short captions (0.8s), works on CPU/GPU',
            'llava': 'LLaVA-1.5-7B (4-bit GPU) — rich multi-paragraph descriptions',
        },
        'override_hint': 'Use --engine blip|llava to override this preference.',
        'reset_hint': 'Run --reset-preference to be re-prompted.',
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def reset_permanent_config():
    """Delete engine_preference.json so user is re-prompted."""
    path = _get_permanent_config_path()
    if os.path.exists(path):
        os.remove(path)
        print("  \u267b\ufe0f  Permanent preference reset. You will be prompted on next run.")
        return True
    return False


# ═══════════════════════════════════════════════════════════════
# SESSION STATE (.hermes_session — volatile, resets with terminal)
# ═══════════════════════════════════════════════════════════════

def read_session_state():
    """
    Read session-scoped preference from .hermes_session.
    Validates that the PID in the file is still alive — if the terminal
    was closed and reopened, the old PID won't exist, so the session
    state is treated as expired.

    Returns 'blip', 'llava', or None (no active session state).
    """
    path = _get_session_state_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        engine = data.get('engine', '')
        pid = data.get('pid', 0)
        if engine not in ('blip', 'llava'):
            return None
        # Check if the PID is still alive
        if _pid_is_alive(pid):
            return engine
        # PID is dead — clean up stale session file
        os.remove(path)
    except (json.JSONDecodeError, IOError, OSError):
        pass
    return None


def write_session_state(engine):
    """
    Save engine choice to .hermes_session with current PID.
    The session is valid as long as this terminal process is alive.
    """
    import datetime
    path = _get_session_state_path()
    data = {
        'engine': engine,
        'pid': os.getpid(),
        'saved_at': datetime.datetime.now().isoformat(),
        'scope': 'session',
        'description': 'Session-scoped engine preference. Expires when terminal closes.',
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def clear_session_state():
    """Remove .hermes_session to end the session preference early."""
    path = _get_session_state_path()
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def _pid_is_alive(pid):
    """Check if a process with the given PID is still running (cross-platform)."""
    if pid <= 0:
        return False
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            SYNCHRONIZE = 0x00100000
            PROCESS_QUERY_INFORMATION = 0x0400
            handle = ctypes.windll.kernel32.OpenProcess(
                SYNCHRONIZE | PROCESS_QUERY_INFORMATION, False, pid
            )
            if handle == 0:
                return False
            # Process exists — check exit code
            exit_code = wintypes.DWORD()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return exit_code.value == 259  # STILL_ACTIVE
        else:
            # Unix: signal 0 checks existence
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError, Exception):
        return False


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE PROMPT — 6 options, 3 scopes
# ═══════════════════════════════════════════════════════════════

def prompt_user_for_engine():
    """
    INTERACTIVE: Prompt user for engine + retention scope.

    Menu:
      [1] SHORT  — This photo only    [2] DETAILED — This photo only
      [3] SHORT  — This session only  [4] DETAILED — This session only
      [5] SHORT  — Permanently        [6] DETAILED — Permanently

    Returns:
        (engine, scope) tuple: engine is 'blip'|'llava', scope is 'photo'|'session'|'permanent'
    """
    has_gpu = _has_gpu()
    gpu_name, vram_gb = _get_gpu_info()

    print()
    print("=" * 60)
    print("  \U0001f916 Hermes Image Analysis — Engine Selection")
    print("=" * 60)
    print()

    if has_gpu:
        print(f"  \u2705 GPU detected: {gpu_name} ({vram_gb} GB VRAM)")
    else:
        print(f"  \u26a0\ufe0f  No GPU detected — running on CPU")

    print()
    print("  Select your description preference and duration:")
    print()

    # Options 1-2: photo only
    print("  [1] \u26a1 SHORT (BLIP-base) — Just for this photo")
    if has_gpu:
        print("  [2] \U0001f50d DETAILED (LLaVA-1.5-7B) — Just for this photo")
    else:
        print("  [2] \U0001f50d DETAILED (LLaVA-1.5-7B) — Just for this photo  \u274c (GPU needed)")
    print()

    # Options 3-4: session
    print("  [3] \u26a1 SHORT (BLIP-base) — For this session only")
    if has_gpu:
        print("  [4] \U0001f50d DETAILED (LLaVA-1.5-7B) — For this session only")
    else:
        print("  [4] \U0001f50d DETAILED (LLaVA-1.5-7B) — For this session only  \u274c (GPU needed)")
    print()

    # Options 5-6: permanent
    print("  [5] \u26a1 SHORT (BLIP-base) — Permanently (Save & don't ask again)")
    if has_gpu:
        print("  [6] \U0001f50d DETAILED (LLaVA-1.5-7B) — Permanently (Save & don't ask again)")
    else:
        print("  [6] \U0001f50d DETAILED (LLaVA-1.5-7B) — Permanently (Save & don't ask again)  \u274c (GPU needed)")
    print()

    print("  \U0001f4a1 Global Override: You can always use '--engine blip' or '--engine llava'")
    print("     to bypass this per-run.")
    print()

    valid_choices = {'1', '2', '3', '4', '5', '6'}
    if not has_gpu:
        valid_choices = {'1', '3', '5'}

    while True:
        try:
            choice = input("  Select [1-6] (default: 1): ").strip()
            if choice == '':
                choice = '1'
            if choice not in valid_choices:
                if not has_gpu and choice in ('2', '4', '6'):
                    print("  \u274c LLaVA requires a GPU — only BLIP (SHORT) options are available.")
                else:
                    print(f"  Please enter a number between 1 and 6.")
                continue

            # Map choice to (engine, scope)
            mapping = {
                '1': ('blip', 'photo'),
                '2': ('llava', 'photo'),
                '3': ('blip', 'session'),
                '4': ('llava', 'session'),
                '5': ('blip', 'permanent'),
                '6': ('llava', 'permanent'),
            }
            engine, scope = mapping[choice]
            break

        except (KeyboardInterrupt, EOFError):
            print("\n  Using default: BLIP (short) — this photo only")
            engine, scope = 'blip', 'photo'
            break

    # ── Persist according to scope ──
    if scope == 'permanent':
        write_permanent_config(engine, source='user')
    elif scope == 'session':
        write_session_state(engine)
    # scope == 'photo': no persistence

    # ── Confirmation message ──
    engine_display = "\u26a1 BLIP (short)" if engine == "blip" else "\U0001f50d LLaVA (detailed)"
    scope_labels = {
        'photo': "\U0001f4f7 This photo only (will ask again next image)",
        'session': "\U0001f4bb This session only (resets when terminal closes)",
        'permanent': "\U0001f4be Permanently saved (won't ask again)",
    }
    print(f"\n  \u2705 Selected: {engine_display}")
    print(f"     Scope: {scope_labels[scope]}")
    if scope == 'permanent':
        print(f"     Config: {_get_permanent_config_path()}")
    elif scope == 'session':
        print(f"     Session: {_get_session_state_path()}")
    print(f"{'=' * 60}")
    print()

    return engine, scope


# ═══════════════════════════════════════════════════════════════
# ENGINE RESOLUTION — main entry points
# ═══════════════════════════════════════════════════════════════

def resolve_engine(force=None):
    """
    Resolve which engine to use WITHOUT prompting.
    Resolution order:
      1. force= parameter (CLI --engine override)
      2. Session state (.hermes_session — still alive?)
      3. Permanent config (engine_preference.json)
      4. Auto-detect (GPU-based fallback)

    Returns 'blip' or 'llava'.
    """
    # 1. CLI override
    if force and force in ('blip', 'llava'):
        return force

    # 2. Session state
    session_engine = read_session_state()
    if session_engine in ('blip', 'llava'):
        return session_engine

    # 3. Permanent config
    permanent_engine = read_permanent_config()
    if permanent_engine in ('blip', 'llava'):
        return permanent_engine

    # 4. Auto-detect
    recommended, _reason = detect_best_engine()
    return recommended


def _is_interactive():
    """
    True if the user can actually respond to prompts.

    Checks (in order):
      1. HERMES_NON_INTERACTIVE=1 env var → False (agent override)
      2. Python's -i flag / PYTHONINSPECT → True
      3. sys.__stdin__.isatty() → True/False (original fd, before redirects)
      4. sys.stdin.isatty() → True/False (current fd, may be piped)

    Returns False for: pipes, redirects, agent tool calls, cron, subprocess w/out PTY.
    Returns True for: real terminal, interactive python, PTY sessions.
    """
    # Explicit agent override
    if os.environ.get('HERMES_NON_INTERACTIVE', '').strip() in ('1', 'true', 'yes'):
        return False

    # Python was started with -i (interactive) or PYTHONINSPECT is set
    if hasattr(sys, 'ps1') or os.environ.get('PYTHONINSPECT'):
        return True

    # Original stdin fd (before any potential redirection by subprocess)
    try:
        if sys.__stdin__.isatty():
            return True
    except Exception:
        pass

    # Current stdin fd
    try:
        if sys.stdin.isatty():
            return True
    except Exception:
        pass

    return False


def get_or_prompt_engine(force=None, interactive=True):
    """
    Primary entry point — resolve engine, prompting interactively if needed.

    Resolution order:
      1. force= parameter (CLI --engine override) → return immediately
      2. Permanent config exists? → return silently (user chose "don't ask again")
      3. Session state alive? → return silently (user chose "this session")
      4. interactive=True AND no permanent config? → PROMPT with 6 options
      5. interactive=False → auto-detect

    Args:
        force: CLI override ('blip', 'llava', or None)
        interactive: If True, prompt when no saved preference exists.

    Returns:
        'blip' or 'llava'
    """
    # CLI override always wins
    if force and force in ('blip', 'llava'):
        return force

    # Permanent config exists → silent, don't prompt
    # (User explicitly chose "Permanently — don't ask again")
    permanent_engine = read_permanent_config()
    if permanent_engine in ('blip', 'llava'):
        return permanent_engine

    # Session state alive → silent, don't prompt
    # (User chose "This session only" and the terminal is still open)
    session_engine = read_session_state()
    if session_engine in ('blip', 'llava'):
        return session_engine

    # No saved preference of any kind
    if interactive and _is_interactive():
        engine, scope = prompt_user_for_engine()
        return engine

    # Non-interactive fallback (no TTY, --no-prompt, or interactive=False)
    if interactive and not _is_interactive():
        print("  ℹ️  Non-interactive mode (no TTY). Use --engine blip|llava or save a"
              " permanent preference to skip this warning.", file=sys.stderr)
    recommended, _reason = detect_best_engine()
    return recommended


# ═══════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE ALIASES
# ═══════════════════════════════════════════════════════════════

def read_engine_config():
    """Alias for read_permanent_config — backward compatibility."""
    return read_permanent_config()


def write_engine_config(engine, source='user'):
    """Alias for write_permanent_config — backward compatibility."""
    write_permanent_config(engine, source)


def get_engine_choice(force=None):
    """Alias for resolve_engine — backward compatibility."""
    return resolve_engine(force)


def reset_engine_preference():
    """Reset both permanent AND session state."""
    result = reset_permanent_config()
    if clear_session_state():
        result = True
    return result


# ═══════════════════════════════════════════════════════════════
# DISPLAY / INFO
# ═══════════════════════════════════════════════════════════════

def print_engine_recommendation():
    """Print a user-friendly engine recommendation (for --show-engines)."""
    has_gpu = _has_gpu()
    gpu_name, vram_gb = _get_gpu_info()

    print("\n" + "=" * 60)
    print("  \U0001f5a5\ufe0f  VISION ENGINE SELECTION")
    print("=" * 60)

    if has_gpu:
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
    else:
        print(f"  \u26a0\ufe0f  No CUDA GPU detected")
        print()
        print(f"  \U0001f4cc Recommended: BLIP-base")
        print(f"     \u2022 Fast, short captions (5-15 words)")
        print(f"     \u2022 Works on CPU, ~1GB model, 0.8s per image")

    # Show permanent config
    perm = read_permanent_config()
    if perm in ('blip', 'llava'):
        path = _get_permanent_config_path()
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            saved_at = data.get('saved_at', 'unknown')
        except Exception:
            saved_at = 'unknown'
        label = "\u26a1 BLIP (short)" if perm == "blip" else "\U0001f50d LLaVA (detailed)"
        print()
        print(f"  \U0001f4be Permanent preference: {label}")
        print(f"     Saved: {saved_at}")

    # Show session state
    sess = read_session_state()
    if sess in ('blip', 'llava'):
        label = "\u26a1 BLIP (short)" if sess == "blip" else "\U0001f50d LLaVA (detailed)"
        print(f"  \U0001f4bb Session preference: {label} (active)")

    print()
    print(f"  \U0001f4a1 Quick commands:")
    print(f"     python analyze_image.py <image> --engine llava")
    print(f"     python analyze_image.py <image> --engine blip")
    print(f"     python analyze_image.py <image>              (uses saved preferences)")
    print(f"     python analyze_image.py --reset-preference   (clear all)")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# CLI (for direct use: python engine_config.py --show / --reset / --prompt)
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Engine configuration tool')
    parser.add_argument('--show', action='store_true', help='Show engine recommendation')
    parser.add_argument('--reset', action='store_true', help='Reset all saved preferences')
    parser.add_argument('--prompt', action='store_true', help='Force interactive prompt')
    args = parser.parse_args()

    if args.reset:
        reset_engine_preference()
    elif args.prompt:
        engine, scope = prompt_user_for_engine()
        print(f"Result: {engine} ({scope})")
    else:
        print_engine_recommendation()
