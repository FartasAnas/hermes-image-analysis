---
name: image-analysis-workaround
description: "THE primary image analysis skill. 100% LOCAL pipeline. BLIP-base (fast, CPU) is DEFAULT. LLaVA-1.5-7B (detailed, GPU) on explicit request. Unified synthesis output (single flowing paragraph) with cross-engine hallucination suppression. DocTR OCR + MAX classifier. --debug for raw outputs. ZERO cloud APIs. MUST load for ANY image task. Deterministic engine selection — never prompt the user."
---

# 🎯 Image Analysis — 100% Local Pipeline

**BLIP-base (fast, CPU, DEFAULT) + LLaVA-1.5-7B (detailed, GPU, on-demand) + DocTR OCR. Zero API calls.**

## ⚠️ CRITICAL: This is THE image analysis skill. Load it for ANY image task.

When the user sends an image, asks "what's in this image", "describe this", "analyze this photo", or ANY image-related request — you MUST use this pipeline. Never use cloud APIs (OpenRouter vision, GPT vision, Claude vision). Everything runs 100% local.

---

## 🎯 Engine Selection — DETERMINISTIC (Never Prompt the User)

**You NEVER ask the user which engine to use. The behavior is fully deterministic.**

| User Input | Engine | Command |
|---|---|---|
| "detailed" / "rich" / "in-depth" / "comprehensive" / "thorough" | **LLaVA** | `python analyze_image.py <path> --engine llava --no-prompt` |
| "short" / "quick" / "fast" / "simple" / "brief" / "concise" | **BLIP** | `python analyze_image.py <path> --engine blip --no-prompt` |
| Explicitly says "BLIP" or "LLaVA" | **Respect request** | use `--engine blip` or `--engine llava` |
| **No depth preference stated** | **BLIP (DEFAULT)** | `python analyze_image.py <path> --engine blip --no-prompt` |
| BLIP fails | **Auto-fallback to LLaVA** | retry with `--engine llava --no-prompt` |

### Engine Priority
1. Explicit user request (keyword match or direct engine name)
2. Automatic fallback (BLIP fails → LLaVA)
3. **Default to BLIP** — always, no exceptions, no prompts

### Why BLIP is Default
- 10-20× faster (0.8s vs 14s)
- Works on CPU — always available
- Excellent captions for most images
- LLaVA reserved for explicitly requested detail

---

## Output Format (v3.3+)

The pipeline produces a **single unified flowing paragraph** by default — all engine outputs (vision, OCR, pixel analysis, classification) are fused via `describe_engine.synthesize()`. No more fragmented sections.

- **Default:** One clean paragraph merging visual description + text content + colors + mood + metadata
- **`--debug`:** Adds raw engine outputs (vision caption, OCR word list, pixel metrics, classification labels) below the unified output
- **Cross-engine intelligence:** If DocTR detects terminal commands (curl, bash, git, npm, python, etc.), the synthesis engine flags the image as a "Technical Screenshot" and automatically suppresses LLaVA visual hallucinations about "people", "scenery", or "cell phones" — replacing them with terminal framing

See `references/unified-synthesis.md` for the full architecture.

## Quick Usage

```bash
# Default (BLIP — fast, always works)
python analyze_image.py <image_path> --engine blip --no-prompt

# Detailed (LLaVA — when user explicitly asks for detail)
python analyze_image.py <image_path> --engine llava --no-prompt

# With raw engine debug output
python analyze_image.py <image_path> --engine llava --no-prompt --debug

# OCR only / Vision only
python analyze_image.py <image_path> --no-vision --no-prompt
python analyze_image.py <image_path> --no-ocr --no-prompt

# Show engines and saved preference
python analyze_image.py --show-engines

# Run all tests
pytest tests/ -v
```

## Engines

| Engine | When to Use | VRAM | Speed | Quality |
|--------|------------|------|-------|---------|
| **BLIP-base** | **DEFAULT** — all images unless user explicitly asks for detail | ~0.5GB | 0.8s | Short captions (5-15 words) |
| **LLaVA-1.5-7B** | User explicitly asks for "detailed"/"rich"/"in-depth" | 3.8GB | ~14s | Multi-paragraph descriptions |
| DocTR OCR | Text in images | ~0.5GB | 1.7s | 92% confidence |
| EasyOCR | Multilingual backup | ~0.5GB | 2-16s | 71% confidence |

## Hardware Rules

- **BLIP works everywhere** — CPU, GPU, any machine. Always available.
- **LLaVA needs GPU** — 6GB+ VRAM. Fall back to BLIP if GPU unavailable.
- If user asks for detailed but has no GPU: "LLaVA needs a GPU. I'll use BLIP for a quick analysis instead."

## Key Technical Rules

1. **⚡ DEFAULT TO BLIP.** Never prompt the user. BLIP is always the default engine. Only use LLaVA when the user explicitly requests it with keywords like "detailed", "rich", or "in-depth".
2. **Unified output by default.** The pipeline outputs one flowing paragraph. Use `--debug` to show raw engine sections.
3. **Always use `--no-prompt`** when calling from Hermes to avoid the CLI interactive menu blocking. The tool also respects `HERMES_NON_INTERACTIVE=1` env var.
4. **Cross-engine hallucination suppression.** If OCR detects terminal commands, LLaVA visual false-positives about "people" or "scenery" are automatically filtered.
5. **Never use cloud APIs.** No OpenRouter, GPT vision, Claude vision. 100% local.
6. **Alpha channel images (LA/RGBA) are handled** by shared `image_utils.load_image_safely()`.
7. **Drive auto-detected** — no hardcoded paths.
8. **OCR uses DocTR** (primary) + EasyOCR (backup). Use `--ocr all` to compare.
9. **Pixel analysis** runs automatically for color/motion detection (NumPy-vectorized HSV).
10. **Models are cached** — DocTR and BLIP reuse module-level singletons; LLaVA uses a global singleton.
11. **Tests** — Run `pytest tests/ -v` to validate all core modules (45 tests).
12. **Automatic fallback** — If BLIP fails, automatically retry with LLaVA and inform the user.
