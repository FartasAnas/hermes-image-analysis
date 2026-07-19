---
name: image-analysis-workaround
description: "THE primary image analysis skill. 100% LOCAL pipeline. DEFAULT: BLIP-base (fast, CPU). Use LLaVA-1.5-7B GPU ONLY when user asks for 'detailed', 'rich', 'in-depth' description. DocTR OCR + MAX classifier. ZERO cloud APIs. MUST load this skill for ANY image-related task."
---

# 🎯 Image Analysis — 100% Local Pipeline (DEFAULT: BLIP)

**BLIP-base (fast, CPU, default) + LLaVA-1.5-7B (detailed, GPU, on-demand) + DocTR OCR. Zero API calls.**

## ⚠️ CRITICAL: This is THE image analysis skill. Load it for ANY image task.

When the user sends an image, asks "what's in this image", "describe this", "analyze this photo", or ANY image-related request — you MUST use this pipeline. Never use cloud APIs (OpenRouter vision, GPT vision, Claude vision). Everything runs 100% local.

## 🎯 Engine Selection — Interactive First-Run Prompt

**On the first run**, the pipeline prompts the user interactively:
*"Would you like a short or detailed description?"*

- **[1] SHORT → BLIP-base** (0.8s, fast captions, works on CPU/GPU)
- **[2] DETAILED → LLaVA-1.5-7B** (~14s, rich multi-paragraph, GPU only)

**The choice is saved** to `engine_preference.json` and used for ALL future images — no re-prompting. Use `--engine llava|blip` to override per-run. Run `--reset-preference` to be re-prompted next time.

## Decision Flow (for Hermes)

```
User sends an image:
  └─ Load this skill (MANDATORY)
       └─ Check: does user say "detailed" / "rich" / "in-depth" trigger words?
            ├─ YES → python analyze_image.py <img> --engine llava --no-prompt
            └─ NO  → Check: does engine_preference.json exist?
                 ├─ YES → python analyze_image.py <img> --no-prompt  (uses saved pref)
                 └─ NO  → python analyze_image.py <img>  (interactive prompt)
                          Hermes: "Would you like a short or detailed description?"
                          Save answer for all future runs.
```

## Quick Usage

```bash
# Interactive: prompts for engine on first run, then remembers
python analyze_image.py <image_path>

# Non-interactive: uses saved preference (no prompt)
python analyze_image.py <image_path> --no-prompt

# Force specific engine (overrides saved preference)
python analyze_image.py <image_path> --engine blip
python analyze_image.py <image_path> --engine llava

# Forget preference and re-prompt next time
python analyze_image.py --reset-preference

# OCR only / Vision only
python analyze_image.py <image_path> --no-vision
python analyze_image.py <image_path> --no-ocr

# Show engines and saved preference
python analyze_image.py --show-engines
```

## Engines

| Engine | When to Use | VRAM | Speed | Quality |
|--------|------------|------|-------|---------|
| **BLIP-base** (DEFAULT) | Always — unless user asks for detail | ~0.5GB | 0.8s | Short captions (5-15 words) |
| **LLaVA-1.5-7B** (ON-DEMAND) | User asks for "detailed"/"rich" description | 3.8GB | ~14s | Multi-paragraph descriptions |
| DocTR OCR | Text in images | ~0.5GB | 1.7s | 92% confidence |
| EasyOCR | Multilingual backup | ~0.5GB | 2-16s | 71% confidence |

## Hardware Rules

- **BLIP works everywhere** — CPU, GPU, any machine. Always available.
- **LLaVA needs GPU** — 6GB+ VRAM. Fall back to BLIP if GPU unavailable.
- If user asks for detailed but has no GPU: "LLaVA needs a GPU. I'll use BLIP for a quick analysis instead."

## Key Technical Rules

1. **BLIP is the DEFAULT.** Only use LLaVA when user explicitly asks for detailed/rich/in-depth.
2. **Interactive first-run.** On very first use, prompt the user with "short or detailed?" and persist the answer.
3. **Saved preference.** Check `engine_preference.json` before running. No need to ask every time.
4. **Trigger words still work.** If user says "detailed" in message, use `--engine llava` regardless of saved pref.
5. **Never use cloud APIs.** No OpenRouter, GPT vision, Claude vision. 100% local.
6. **Alpha channel images (LA/RGBA) are handled** by `_load_image_safely()`.
7. **Drive auto-detected** — no hardcoded paths.
8. **OCR uses DocTR** (primary) + EasyOCR (backup). Use `--ocr all` to compare.
9. **Pixel analysis** runs automatically for color/motion detection.
