#!/usr/bin/env python3
"""
Image Analysis — 100% LOCAL Pipeline (Zero API Calls)
=====================================================
OCR:      DocTR (primary) + EasyOCR (backup)
Vision:   BLIP image captioning
=====================================================

Usage:
  python analyze_image.py <image_path>                # Full analysis (OCR + caption)
  python analyze_image.py <image_path> --ocr all       # Compare DocTR vs EasyOCR
  python analyze_image.py <image_path> --no-vision     # OCR only
  python analyze_image.py <image_path> --no-ocr        # Vision caption only

Requirements (all local, no internet after first model download):
  - DocTR:  ~100MB models (auto-downloaded on first run)
  - EasyOCR: ~100MB models (auto-downloaded on first run)
  - BLIP:    ~1GB model  (auto-downloaded on first run)

E: drive only. Zero C: references.
"""
import sys, os, time, argparse

os.environ['TMP'] = r'E:\hermes_tools\temp'
os.environ['TEMP'] = r'E:\hermes_tools\temp'

from PIL import Image

# ═══════════════════════════════════════════════════════════
# ENGINE 1: DocTR OCR — PyTorch, db_resnet50 + crnn_vgg16_bn
# ═══════════════════════════════════════════════════════════
def run_doctr(image_path):
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    t0 = time.time()
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    elapsed = time.time() - t0
    words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    t = word.value.strip()
                    if t and len(t) > 1:
                        words.append({"text": t, "confidence": round(word.confidence, 3)})
    full = " ".join(w["text"] for w in words)
    return {
        "engine": "DocTR (db_resnet50 + crnn_vgg16_bn)",
        "words": words, "word_count": len(words), "full_text": full,
        "avg_confidence": round(sum(w["confidence"] for w in words) / len(words), 3) if words else 0,
        "time_seconds": round(elapsed, 2)
    }

# ═══════════════════════════════════════════════════════════
# ENGINE 2: EasyOCR — backup, CRAFT + CRNN
# ═══════════════════════════════════════════════════════════
def run_easyocr(image_path):
    import easyocr
    t0 = time.time()
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    results = reader.readtext(image_path, detail=1, paragraph=False)
    elapsed = time.time() - t0
    words = []
    for entry in results:
        if len(entry) == 3:
            _, text, conf = entry
        elif len(entry) == 2:
            text, conf = entry
        else:
            continue
        text = text.strip()
        if text and len(text) > 1:
            words.append({"text": text, "confidence": round(float(conf), 3)})
    full = "\n".join(w["text"] for w in words)
    return {
        "engine": "EasyOCR (CRAFT + CRNN)",
        "words": words, "word_count": len(words), "full_text": full,
        "avg_confidence": round(sum(w["confidence"] for w in words) / len(words), 3) if words else 0,
        "time_seconds": round(elapsed, 2)
    }

# ═══════════════════════════════════════════════════════════
# ENGINE 3: BLIP — local image captioning
# ═══════════════════════════════════════════════════════════
_blip_processor = None
_blip_model = None

def _load_blip():
    global _blip_processor, _blip_model
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _blip_processor, _blip_model

def run_blip(image_path):
    processor, model = _load_blip()
    t0 = time.time()
    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=100)
    caption = processor.decode(out[0], skip_special_tokens=True)
    elapsed = time.time() - t0
    return {
        "engine": "BLIP (Salesforce/blip-image-captioning-base)",
        "caption": caption,
        "time_seconds": round(elapsed, 2)
    }

# ═══════════════════════════════════════════════════════════
# Camera vs Digital Detection (BLIP caption keyword matching)
# ═══════════════════════════════════════════════════════════
DIGITAL_KEYWORDS = [
    "painting", "painted", "illustration", "artwork", "drawing", "cartoon",
    "product label", "advertisement", "screenshot", "graphic", "design",
    "logo", "circuit", "chip", "microchip", "user interface", "ui",
    "poster", "banner", "sign", "menu", "diagram", "chart"
]

def classify_camera_digital(blip_caption):
    """Determine if an image is a camera photo or digital/screenshot using BLIP caption."""
    lower = blip_caption.lower()
    if any(kw in lower for kw in DIGITAL_KEYWORDS):
        return "🖥️ Digital / Screenshot"
    return "📷 Camera Photo"

def analyze_metadata(image_path):
    from PIL import Image
    import os
    img = Image.open(image_path)
    w, h = img.size
    kb = os.path.getsize(image_path) // 1024
    ratio = w / h
    gray = img.convert('L')
    px = list(gray.getdata())
    avg_brightness = sum(px) / len(px)
    
    return {
        "dimensions": f"{w}x{h}",
        "ratio": round(ratio, 2),
        "file_size_kb": kb,
        "mode": img.mode,
        "avg_brightness": round(avg_brightness),
        "is_dark": avg_brightness < 60,
    }

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='100% Local Image Analysis (OCR + Vision)')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--ocr', default='doctr',
                       choices=['doctr', 'easyocr', 'all', 'none'],
                       help='OCR engine (default: doctr)')
    parser.add_argument('--no-ocr', action='store_true', help='Skip OCR entirely')
    parser.add_argument('--no-vision', action='store_true', help='Skip BLIP captioning')
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    do_ocr = not args.no_ocr
    do_vision = not args.no_vision
    if args.ocr == 'none':
        do_ocr = False

    print("=" * 70)
    print(f"  \U0001F4F7 {os.path.basename(image_path)}")
    print(f"  \U0001F3E0 100% LOCAL — zero API calls, zero rate limits, zero cost")
    print("=" * 70)

    # ── Metadata ──
    meta = analyze_metadata(image_path)
    print(f"\n  📐 {meta['dimensions']} | {meta['ratio']}:1 | {meta['file_size_kb']}KB | {meta['mode']}")
    print(f"  ☀️ Brightness: {meta['avg_brightness']}/255", end="")

    results = {}

    # ── BLIP Vision ──
    if do_vision:
        print(f"\n  🧠 BLIP caption...", end=" ", flush=True)
        try:
            r = run_blip(image_path)
            print(f"done ({r['time_seconds']}s)")
            results['blip'] = r
            # Classify camera vs digital from BLIP caption
            img_type = classify_camera_digital(r['caption'])
            print(f"  📸 Type: {img_type}")
        except Exception as e:
            print(f"❌ {e}")
    else:
        print()  # end the brightness line

    # ── OCR ──
    if do_ocr:
        engines = ['doctr', 'easyocr'] if args.ocr == 'all' else [args.ocr]
        for eng in engines:
            print(f"  🔍 [{eng}]...", end=" ", flush=True)
            try:
                r = run_doctr(image_path) if eng == 'doctr' else run_easyocr(image_path)
                print(f"{r['word_count']} words ({r['avg_confidence']:.0%} conf) in {r['time_seconds']}s")
                results[eng] = r
            except Exception as e:
                print(f"❌ {e}")

    # ═══════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("                   ANALYSIS REPORT")
    print(f"{'=' * 70}")

    # BLIP caption
    if 'blip' in results:
        print(f"\n  ── What's in this image? (BLIP) ──")
        print(f"  \U0001F4DD {results['blip']['caption']}")

    # OCR text
    for eng in ['doctr', 'easyocr']:
        if eng not in results:
            continue
        r = results[eng]
        print(f"\n  ── Text found ({r['engine']}) ──")
        if r['word_count']:
            for w in sorted(r['words'], key=lambda x: x['confidence'], reverse=True)[:20]:
                bar = "█" * int(w['confidence'] * 20)
                print(f"    [{w['confidence']:.0%}] {w['text']} {bar}")
            if r['word_count'] > 20:
                print(f"    ... and {r['word_count'] - 20} more words")
            print(f"\n  Full text: {r['full_text'][:500]}")
        else:
            print(f"  (no text detected)")

    # Comparison
    if 'doctr' in results and 'easyocr' in results:
        d, e = results['doctr'], results['easyocr']
        print(f"\n  ── OCR Comparison ──")
        print(f"  {'':<20} {'DocTR':>10} {'EasyOCR':>10}")
        print(f"  {'Words':<20} {d['word_count']:>10} {e['word_count']:>10}")
        print(f"  {'Confidence':<20} {d['avg_confidence']:>9.0%} {e['avg_confidence']:>9.0%}")
        print(f"  {'Speed':<20} {d['time_seconds']:>9}s {e['time_seconds']:>9}s")

    print(f"\n  🏠 100% LOCAL — zero API calls, zero rate limits, zero cost")
    print(f"{'=' * 70}")
