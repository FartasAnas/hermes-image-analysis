---
name: image-analysis-workaround
description: "THE primary image analysis skill. 100% LOCAL pipeline. DEFAULT: BLIP-base (fast, CPU). Use LLaVA-1.5-7B GPU ONLY when user asks for 'detailed', 'rich', 'in-depth' description. DocTR OCR + MAX classifier. ZERO cloud APIs. MUST load this skill for ANY image-related task."
---

# 🎯 Image Analysis — 100% Local Pipeline (DEFAULT: BLIP)

**BLIP-base (fast, CPU, default) + LLaVA-1.5-7B (detailed, GPU, on-demand) + DocTR OCR. Zero API calls.**

## ⚠️ CRITICAL: This is THE image analysis skill. Load it for ANY image task.

When the user sends an image, asks "what's in this image", "describe this", "analyze this photo", or ANY image-related request — you MUST use this pipeline. Never use cloud APIs (OpenRouter vision, GPT vision, Claude vision). Everything runs 100% local.

## 🎯 Engine Selection — DEFAULT is BLIP

**BLIP-base is the DEFAULT engine.** It's fast (0.8s), works on CPU and GPU, and produces short accurate captions. Use BLIP for:
- Quick "what's in this image?" queries
- OCR-focused analysis
- Any image task unless the user explicitly asks for detail

**LLaVA-1.5-7B is ON-DEMAND only.** Use LLaVA ONLY when the user uses trigger words:
- "detailed description", "rich description", "in-depth analysis"
- "describe in detail", "analyze thoroughly", "comprehensive description"
- "what do you see in detail", "full analysis"
- Or explicitly says "use LLaVA" / "use the detailed engine"

## Quick Usage

```bash
# DEFAULT — fast BLIP analysis
python analyze_image.py <image_path>

# Detailed — LLaVA (only when user asks for rich detail)
python analyze_image.py <image_path> --engine llava

# OCR only
python analyze_image.py <image_path> --no-vision

# Show available engines
python analyze_image.py --show-engines
```

## Decision Flow (for Hermes)

```
User sends an image:
  └─ Does user say "detailed" / "rich" / "in-depth" / "describe in detail"?
       ├─ YES → Use LLaVA: python analyze_image.py <img> --engine llava
       └─ NO  → Use BLIP (default): python analyze_image.py <img>
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
2. **Follow the decision flow above.** Check for trigger words before choosing engine.
3. **Never use cloud APIs.** No OpenRouter, GPT vision, Claude vision. 100% local.
4. **Alpha channel images (LA/RGBA) are handled** by `_load_image_safely()`.
5. **Drive auto-detected** — no hardcoded paths.
6. **OCR uses DocTR** (primary) + EasyOCR (backup). Use `--ocr all` to compare.
7. **Pixel analysis** runs automatically for color/motion detection.
