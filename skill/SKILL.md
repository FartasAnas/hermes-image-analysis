---
name: image-analysis-workaround
description: "THE primary image analysis skill. 100% LOCAL pipeline. BLIP-base (fast, CPU) + LLaVA-1.5-7B (detailed, GPU). DocTR OCR + MAX classifier. ZERO cloud APIs. MUST load this skill for ANY image-related task. MANDATORY: always ask user 'Would you like a detailed, rich, or in-depth description?' before executing — never auto-select an engine."
---

# 🎯 Image Analysis — 100% Local Pipeline

**BLIP-base (fast, CPU) + LLaVA-1.5-7B (detailed, GPU, on-demand) + DocTR OCR. Zero API calls.**

## ⚠️ CRITICAL: This is THE image analysis skill. Load it for ANY image task.

When the user sends an image, asks "what's in this image", "describe this", "analyze this photo", or ANY image-related request — you MUST use this pipeline. Never use cloud APIs (OpenRouter vision, GPT vision, Claude vision). Everything runs 100% local.

---

## 🚨 MANDATORY INTERACTIVE SELECTION (Highest Priority)

**You MUST NOT auto-execute the analysis pipeline** when a user submits an image without explicitly stating their depth preference. You are strictly forbidden from running the script with default configurations or background assumptions.

### The Mandatory Question

Before executing ANY background tool or local python script, you MUST immediately reply with this exact question:

> "Would you like a **detailed**, **rich**, or **in-depth** description for this analysis?"

### Routing Based on User Response

Wait for the user's explicit response, then:

| User Response | Engine | Command |
|--------------|--------|---------|
| Detailed / rich / in-depth / comprehensive | **LLaVA** | `python analyze_image.py <path> --engine llava --no-prompt` |
| Short / fast / simple / quick / basic | **BLIP** | `python analyze_image.py <path> --engine blip --no-prompt` |

### Exception Override

The ONLY exception: if the user includes a depth keyword in their very first prompt (e.g., *"give me a detailed analysis of this image"*), you may execute immediately without asking.

**Depth keywords that trigger immediate execution:**
- `detailed`, `rich`, `in-depth`, `comprehensive`, `thorough`, `full` → LLaVA
- `short`, `quick`, `fast`, `simple`, `brief`, `basic` → BLIP

If NO depth keyword is present in the user's message, you **MUST** pause and ask the mandatory question. Do NOT guess. Do NOT use saved preferences. Do NOT auto-detect.

---

## Quick Usage

```bash
# Detailed (LLaVA — when user explicitly asks for detail)
python analyze_image.py <image_path> --engine llava --no-prompt

# Short (BLIP — when user explicitly asks for quick)
python analyze_image.py <image_path> --engine blip --no-prompt

# OCR only / Vision only
python analyze_image.py <image_path> --no-vision --no-prompt
python analyze_image.py <image_path> --no-ocr --no-prompt

# Show engines and saved preference
python analyze_image.py --show-engines
```

## Engines

| Engine | When to Use | VRAM | Speed | Quality |
|--------|------------|------|-------|---------|
| **BLIP-base** | User asks for "short"/"quick"/"fast" | ~0.5GB | 0.8s | Short captions (5-15 words) |
| **LLaVA-1.5-7B** | User asks for "detailed"/"rich"/"in-depth" | 3.8GB | ~14s | Multi-paragraph descriptions |
| DocTR OCR | Text in images | ~0.5GB | 1.7s | 92% confidence |
| EasyOCR | Multilingual backup | ~0.5GB | 2-16s | 71% confidence |

## Hardware Rules

- **BLIP works everywhere** — CPU, GPU, any machine. Always available.
- **LLaVA needs GPU** — 6GB+ VRAM. Fall back to BLIP if GPU unavailable.
- If user asks for detailed but has no GPU: "LLaVA needs a GPU. I'll use BLIP for a quick analysis instead."

## Key Technical Rules

1. **🚨 MANDATORY PROMPT FIRST.** Never auto-execute. Always ask the user if they want detailed/rich/in-depth unless they already stated it.
2. **Never use cloud APIs.** No OpenRouter, GPT vision, Claude vision. 100% local.
3. **Always use `--no-prompt`** when calling from Hermes to avoid the CLI interactive menu blocking.
4. **Alpha channel images (LA/RGBA) are handled** by `_load_image_safely()`.
5. **Drive auto-detected** — no hardcoded paths.
6. **OCR uses DocTR** (primary) + EasyOCR (backup). Use `--ocr all` to compare.
7. **Pixel analysis** runs automatically for color/motion detection.
8. **Models are cached** — DocTR and BLIP reuse module-level singletons; LLaVA uses a global singleton.
