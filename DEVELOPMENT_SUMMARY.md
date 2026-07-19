# Hermes Image Analysis Pipeline — Complete Development Summary

**Repository:** [FartasAnas/hermes-image-analysis](https://github.com/FartasAnas/hermes-image-analysis)  
**Branch:** `main` (HEAD: `aa1f6e5`)  
**Tested on:** Ryzen 7 5700X, 16GB RAM, NVIDIA RTX 3060 12GB VRAM, CUDA 13.2  
**Constraint:** 100% local, zero cloud APIs, all paths on `E:\hermes_tools\`

---

## Timeline of Changes

### v3.1 — Interactive Engine Prompting + Deep Optimization
**Commit:** `c8a58c2`

**New Features:**
- First-run interactive prompt: *"Would you like a short or detailed description?"*
- Engine preference persisted to `engine_preference.json` for future sessions
- `--no-prompt` and `--reset-preference` CLI flags

**Performance Optimizations (8 fixes):**

| Fix | File | What |
|-----|------|------|
| Inverted-index keyword matching | `max_classifier.py` | O(W) replaces O(D×L×K) triple loop; 19,449 indexed words |
| NumPy vectorized HSV | `pixel_analysis.py` | 10-50× faster than `colorsys` loop |
| DocTR model caching | `analyze_image.py` | Module-level singleton avoids ~1.7s reload per image |
| BLIP model caching | `analyze_image.py` | Module-level singleton avoids ~0.5s reload per image |
| LLaVA VRAM cleanup | `llava_engine.py` | `torch.cuda.empty_cache()` between describe() calls |
| Smart downsampling | `pixel_analysis.py` | Avoids `np.unique` on full pixel arrays |
| Lazy keyword count | `max_classifier.py` | 10.5M mega-keywords not expanded at import |
| NumPy metadata | `analyze_image.py` | Replaces `list(getdata())` for 100× memory reduction |

**Bug Fixes:**
- `textwrap` NameError (used before import)
- 136-line duplicate `generate_detailed_description()` removed
- Module-level `import torch` removed from `engine_config.py`
- `ZeroDivisionError` guard in `analyze_metadata()`
- Graceful error handling for corrupt images

**Documentation:** Merged `BENCHMARK.md` + `COMPLETE_SUMMARY.md` into single `README.md`

---

### v3.2 — Granular Engine Scope: Photo / Session / Permanent
**Commit:** `298aafd`

**New Features:**
- 6-option interactive menu with three retention scopes
- `.hermes_session` file with PID liveness check for session-scoped preferences
- `HERMES_NON_INTERACTIVE=1` env var + `sys.__stdin__.isatty()` detection

**Menu structure:**
```
[1][2] — Just for this photo (ephemeral)
[3][4] — For this session only (.hermes_session)
[5][6] — Permanently (engine_preference.json)
```

**Resolution chain:** CLI flag → permanent config → session state → prompt → auto-detect

**Session PID auto-expiry:** Stale `.hermes_session` files auto-cleaned when original process is dead (cross-platform: `OpenProcess`/`GetExitCodeProcess` on Windows, `os.kill(pid, 0)` on Unix)

---

### v3.3 — Unified Synthesis Engine + Cross-Engine Intelligence
**Commit:** `b92a9f0`

**New Features:**

| Component | Purpose |
|-----------|---------|
| `build_state()` | Merges vision, OCR, pixel, classification into single JSON state object |
| `synthesize()` | Produces one flowing natural-language paragraph from all engines |
| `detect_technical_screenshot()` | 50+ signal patterns (curl, bash, git, npm, SQL, paths, error patterns) |
| `_suppress_hallucinations()` | Filters LLaVA false-positives ("person", "cell phone", "scenery") on terminal shots |
| `debug_dump()` | Raw multi-section output behind `--debug` flag |
| `generate_detailed_description()` | Backward-compat wrapper retained |

**Output contrast:**

*Before (v3.2):*
```
── LLaVA Description ──
  [vision caption]
── Pixel Analysis ──
  [color data]
── Structured Summary ──
  [classifier output]
── Text found (DocTR) ──
  [OCR words with confidence bars]
```

*After (v3.3):*
```
── UNIFIED ANALYSIS ──
  A dark terminal screenshot showing output from curl/python.
  The terminal output includes: "curl api.github.com | python..."
  With near-black palette; moderately colorful. 1632x918, 1825KB.
```

**Cross-engine intelligence matrix:**
```
OCR contains curl/bash/git/python...?
├─ YES → flag "technical_screenshot"
│        Suppress vision hallucinations about people/scenery
│        Replace with terminal framing
└─ NO  → Normal fusion: vision + OCR + pixel + classification
```

**Behavioral changes:**
- Default output: single unified paragraph
- Old multi-section report only shown with `--debug`
- OCR text <5 words with low confidence not forced into output

---

### Review Pass — Claude-Fable-5 Deep Audit
**Commit:** `aa1f6e5`

**6 fixes applied:**

| File | Fix |
|------|-----|
| `analyze_image.py` | Removed unused `from PIL import Image` at module level |
| `analyze_image.py` | Removed unused `read_engine_config` import |
| `describe_engine.py` | TECH_SIGNALS: stripped trailing spaces from `"uv "`, `"git "`, `"node "` |
| `describe_engine.py` | `_suppress_hallucinations`: set-based cmd_hints with `.strip()` for precise matching |
| `describe_engine.py` | `synthesize()`: conditional `"With"` / `"The image has"` prefix avoids awkward phrasing |
| `describe_engine.py` | Metadata line simplified to `dimensions, sizeKB` (removed raw brightness bracket) |
| `engine_config.py` | Removed dead comment block in `prompt_user_for_engine()` |

---

## Architecture (Current State)

```
analyze_image.py (Main Pipeline)
│
├── 🧠 Vision Engine
│   ├── BLIP-base          → fast 0.8s, CPU/GPU, short captions
│   └── LLaVA-1.5-7B (4-bit) → rich 14s, GPU only, multi-paragraph
│
├── 📝 OCR Engines
│   ├── DocTR (primary)      → 1.7s, 92% confidence
│   └── EasyOCR (backup)     → 2-16s, multilingual
│
├── 📊 Analysis
│   ├── MAX Classifier       → 34 dims, 11.5M keywords, inverted index
│   ├── Pixel Analysis       → NumPy-vectorized HSV, motion detection
│   └── Synthesis Engine     → Unified JSON state → single paragraph
│
└── ⚙️ Configuration
    ├── engine_config.py     → GPU detection, 6-option scope prompt, TTY guard
    ├── hermes_config.py     → Dynamic drive detection, env var setup
    └── llava_engine.py      → LLaVA singleton with VRAM management
```

---

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `analyze_image.py` | ~404 | Main pipeline — engine selection, vision, OCR, unified synthesis |
| `describe_engine.py` | ~413 | Synthesis engine, tech detection, hallucination suppression |
| `engine_config.py` | ~568 | GPU detection, 6-option scope prompt, TTY guard, session state |
| `llava_engine.py` | ~250 | LLaVA-1.5-7B engine with VRAM management |
| `max_classifier.py` | ~323 | 34-dim classifier with inverted index |
| `pixel_analysis.py` | ~270 | NumPy-vectorized color, motion, contrast analysis |
| `hermes_config.py` | ~278 | Dynamic drive detection, GPU setup, env vars |
| `keyword_generator.py` | ~1315 | 37K dimension-specific keyword definitions |
| `skill/SKILL.md` | ~88 | Hermes Agent skill — mandatory interactive prompt rule |
| `README.md` | ~470 | Comprehensive documentation |

---

## Key Design Decisions

1. **100% local** — No cloud APIs anywhere. BLIP + LLaVA + DocTR all run on RTX 3060.
2. **Inverted index** — 19,449 indexed words replace O(D×L×K) triple-nested keyword loop for O(W) lookups.
3. **Scope-aware persistence** — Three levels (photo/session/permanent) with PID-based session expiry.
4. **TTY guard** — `_is_interactive()` prevents agent-mode hangs; respects `HERMES_NON_INTERACTIVE=1`.
5. **Hallucination suppression** — Cross-engine intelligence: OCR terminal signals override vision false-positives.
6. **Model caching** — DocTR, BLIP, LLaVA all use module-level singletons.
7. **Unified output** — Single flowing paragraph replaces fragmented multi-section reports.
8. **`--debug` flag** — Raw engine outputs available on demand without cluttering default output.
