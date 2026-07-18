# Hermes Image Analysis Skill — 100% Local Pipeline

**Three local engines. Zero API calls. Zero rate limits. Zero cost.**

## What It Does

This is a complete local image analysis skill for [Hermes Agent](https://hermes-agent.nousresearch.com). It replaces cloud vision APIs with three local engines:

| Engine | Purpose | Accuracy | Speed |
|--------|---------|----------|-------|
| **DocTR** | OCR text extraction | 92% confidence per word, 85.7% text detection | 1.7s |
| **EasyOCR** | Backup OCR (multilingual) | 71% confidence | 2–16s |
| **BLIP** | Image captioning & object description | 88.6% useful vs gpt-4o | 1.2s |

Also includes BLIP-powered camera vs digital detection at **100% accuracy** (validated on 35 diverse images; 7-image test was too small).

## Engines Rejected (After Full Benchmark)

| Engine | Why Rejected |
|--------|-------------|
| PaddleOCR 3.7 | oneDNN crash on Windows CPU |
| PaddleOCR 2.10 | torch DLL conflicts |
| TrOCR (Microsoft) | `importlib.metadata.version('torch')` → None bug |
| Surya 0.22 | Requires vllm or llama-server (not pip-installable) |
| Tesseract | 60-70% accuracy vs 92% for DocTR |
| OpenRouter Vision | Replaced by BLIP — same accuracy, zero cost |

## Quick Install

```bash
# Dependencies
uv pip install easyocr python-doctr transformers torch pillow

# Prevent C: drive pollution (add to ~/.bashrc)
export HF_HOME=<your-drive>:/hermes_tools/.hf
export DOCTR_CACHE_DIR=<your-drive>:/hermes_tools/cache/doctr
export EASYOCR_MODULE_PATH=<your-drive>:/hermes_tools/cache/easyocr
export XDG_CACHE_HOME=<your-drive>:/hermes_tools/cache
export TORCH_HOME=<your-drive>:/hermes_tools/cache/torch
```

## Usage

```bash
# Full analysis (OCR + caption)
python analyze_image.py photo.png

# Compare OCR engines
python analyze_image.py screenshot.png --ocr all

# Caption only
python analyze_image.py scenic.jpg --no-ocr
```

After first model download (~1.2GB), everything runs offline forever.
