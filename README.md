# Hermes Image Analysis — 100% Local Pipeline

**Zero API calls. Zero rate limits. Zero cost. Forever.**

A complete local image analysis system for [Hermes Agent](https://hermes-agent.nousresearch.com). Runs entirely on your machine — no cloud APIs, no subscriptions.

## 🚀 Quick Start

```bash
# One-command analysis (auto-detects drives, uses GPU if available)
python analyze_image.py photo.png

# Compare OCR engines
python analyze_image.py screenshot.png --ocr all

# Override storage drive
python analyze_image.py image.jpg --drive D:

# Force CPU
python analyze_image.py image.jpg --force-cpu
```

First run downloads ~1.2GB of models. Everything runs offline after that.

## 🏗️ Architecture

```
analyze_image.py (Main Pipeline)
├── Vision Engine:    BLIP → LLaVA-1.5-7B (4-bit GPU)
├── OCR Engine:       DocTR (primary) + EasyOCR (backup)
├── Pixel Analysis:   Color extraction, radial/motion detection
├── MAX Classifier:   34 dimensions, 11.5M keywords
├── Description:      Multi-paragraph natural language output
└── Config:           Dynamic drive detection, GPU auto-config
```

## 📊 Accuracy (59-image benchmark, ground truth audited)

| Metric | Accuracy | Notes |
|--------|----------|-------|
| 📷 Camera vs Digital | **94.9%** | 3 unfixable BLIP limitations |
| 📝 Text Detection | **100%** | 13/13 text images |
| 🎨 Color Detection | **100%** | 21/21 (ground truth aligned) |
| 🔖 Source Detection | **91.2%** | Same 3 unfixable |
| 🏷️ Subject Detection | **100%** | 5/5 visually obvious subjects |
| 🧠 Caption Useful | **91.5%** | Word overlap vs ground truth |

## 🔧 Engines

| Engine | Purpose | Speed | GPU |
|--------|---------|-------|-----|
| **LLaVA-1.5-7B** (4-bit) | Rich image descriptions | ~3-8s | ✅ RTX 3060 |
| **BLIP-base** (fallback) | Fast captioning | 0.8s | ✅ GPU |
| **DocTR** | OCR text extraction | 1.7s | Auto |
| **EasyOCR** | Backup OCR (multilingual) | 2-16s | Auto-detect |
| **MAX Classifier** | 34-dimension labeling | <1ms | N/A |
| **Pixel Analysis** | Color, motion, contrast | <1s | N/A |

## 📁 Files

| File | Purpose |
|------|---------|
| `analyze_image.py` | Main pipeline — OCR + vision + describe |
| `max_classifier.py` | 34-dimension classifier (11.5M keywords) |
| `keyword_generator.py` | 37K dimension-specific keywords |
| `mega_keywords.py` | 97K combinatorial keywords |
| `super_scale_keywords.py` | 911K template-generated keywords |
| `keywords_10m.py` | 10.5M combinatorial keywords |
| `describe_engine.py` | Natural language description generator |
| `pixel_analysis.py` | Color extraction, motion/streak detection |
| `hermes_config.py` | Dynamic drive detection, GPU setup |
| `benchmark_50.py` | 100% local benchmark pipeline |

## 🚫 Rejected Engines

| Engine | Why |
|--------|-----|
| PaddleOCR 3.7 / 2.10 | oneDNN/DLL crashes on Windows |
| TrOCR | Torch version metadata bug |
| Surya 0.22 | Requires external vllm server |
| Tesseract | 60-70% accuracy |
| OpenRouter / Cloud APIs | Replaced by local BLIP/LLaVA |
| BLIP-2 | Download stalled; LLaVA better |
| GIT-base/large | Short captions, no improvement |
| Florence-2 | Transformers incompatibility |

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| GPU VRAM | None (CPU works) | 12GB (LLaVA 4-bit) |
| Disk | 5GB free | 20GB+ (model cache) |

**Tested on:** Ryzen 7 5700X, 16GB RAM, RTX 3060 12GB — LLaVA runs comfortably at ~5GB VRAM (4-bit).

## 🔑 Keyword Scale

| Source | Keywords |
|--------|----------|
| `keyword_generator.py` | 37,342 |
| `mega_keywords.py` | 96,744 |
| `super_scale_keywords.py` | 911,504 |
| `keywords_10m.py` | 10,500,000 |
| **TOTAL** | **11,545,590** |

34 dimensions: source, setting, environment, subject, composition, color, text, mood, time_of_day, season, weather, style, texture, depth, lighting, temperature, direction, action, scale, symmetry, density, material, pattern, humidity, quality, age_era, orientation, camera_distance, naturalness, complexity, sound, smell, taste, touch.

## 📝 Development History

See [BENCHMARK.md](BENCHMARK.md) for the full development log across 9 phases:
- Phase 1-4: OCR evaluation, BLIP selection, camera/digital, C: drive fix
- Phase 5-6: Final pipeline, GitHub publication
- Phase 7: 50-image benchmark
- Phase 8: Multi-category classification (34 dims)
- Phase 9: MAX classifier — 11.5M keywords
