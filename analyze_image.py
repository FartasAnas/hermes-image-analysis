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

Storage: Automatically detects available drives (avoids C: on Windows).
         Set --drive to override (e.g., --drive D:).
GPU: Auto-detects CUDA/MPS. Use --force-cpu to disable.
"""
import sys, os, time, argparse

# ── Dynamic config (no hardcoded paths) ──
from hermes_config import setup_environment, gpu_available, gpu_info_str, get_storage_drive

from PIL import Image

# We'll set up env vars during main() after parsing args

# ═══════════════════════════════════════════════════════════
# ENGINE 1: DocTR OCR — PyTorch, db_resnet50 + crnn_vgg16_bn
# ═══════════════════════════════════════════════════════════
def run_doctr(image_path, force_cpu=False):
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    t0 = time.time()
    
    has_gpu, device, _ = gpu_available()
    if force_cpu:
        has_gpu = False
    
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
        "time_seconds": round(elapsed, 2),
        "device": device if has_gpu else "cpu",
    }

# ═══════════════════════════════════════════════════════════
# ENGINE 2: EasyOCR — backup, CRAFT + CRNN
# ═══════════════════════════════════════════════════════════
def run_easyocr(image_path, force_cpu=False):
    import easyocr
    t0 = time.time()
    
    has_gpu, device, gpu_name = gpu_available()
    if force_cpu:
        has_gpu = False
        device = "cpu"
    
    reader = easyocr.Reader(['en'], gpu=has_gpu, verbose=False)
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
        "time_seconds": round(elapsed, 2),
        "device": device if has_gpu else "cpu",
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

def _load_image_safely(image_path):
    """Load image, handling LA/RGBA alpha channels properly.
    
    For LA/RGBA images, the actual visual content may be in the alpha channel.
    We composite onto a white background to preserve what the human eye sees.
    """
    from PIL import Image
    img = Image.open(image_path)
    
    if img.mode in ('LA', 'PA'):
        # Luminance+Alpha: alpha channel IS the visible content
        # Composite alpha onto white background
        background = Image.new('L', img.size, 255)
        alpha = img.getchannel('A')
        background.paste(alpha, mask=alpha)
        return background.convert('RGB')
    
    elif img.mode in ('RGBA', 'RGBa'):
        # Composite onto white background to preserve transparency
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'RGBA':
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img)
        return background
    
    elif img.mode == 'P':
        # Palette mode — convert to RGBA first
        return img.convert('RGBA').convert('RGB')
    
    else:
        return img.convert('RGB')


def run_blip(image_path, enhanced=True):
    """
    Run BLIP captioning. When enhanced=True, runs both unconditional
    and conditional generation, merging the best details from both.
    """
    processor, model = _load_blip()
    t0 = time.time()
    img = _load_image_safely(image_path)
    
    # Always get the unconditional caption (has specific details)
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=100)
    base_caption = processor.decode(out[0], skip_special_tokens=True)
    
    if enhanced:
        # Also try conditional generation for richer context
        prompts = [
            "a picture of",  # BLIP's official training prompt
            "this is a picture of",
            "the image shows",
        ]
        
        best_conditional = ""
        for prompt in prompts:
            try:
                inputs = processor(img, text=prompt, return_tensors="pt")
                out = model.generate(**inputs, max_new_tokens=120, num_beams=3,
                                    early_stopping=True)
                caption = processor.decode(out[0], skip_special_tokens=True)
                if len(caption) > len(best_conditional):
                    best_conditional = caption
            except:
                pass
        
        # Merge: use conditional for context, mention unconditional details naturally
        if best_conditional and len(best_conditional) > 20:
            # Extract meaningful unique words from unconditional
            base_words = set(base_caption.lower().split())
            cond_words = set(best_conditional.lower().split())
            unique = base_words - cond_words
            skip_words = {'a', 'an', 'the', 'is', 'of', 'in', 'on', 'at', 'to', 
                         'with', 'and', 'or', 'this', 'that', 'it', 'its', 'are',
                         'was', 'were', 'be', 'has', 'have', 'for', 'by', 'from'}
            meaningful = sorted(unique - skip_words)
            
            if meaningful:
                # Pick top 3 most impactful words
                top_words = meaningful[:3]
                if len(top_words) == 1:
                    caption = f"{best_conditional}, featuring {top_words[0]}"
                elif len(top_words) == 2:
                    caption = f"{best_conditional}, showing {top_words[0]} and {top_words[1]}"
                else:
                    caption = f"{best_conditional}, with {top_words[0]}, {top_words[1]}, and {top_words[2]}"
            else:
                caption = best_conditional
        else:
            caption = base_caption
    else:
        caption = base_caption
    
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
    # Explicitly digital/synthetic content
    "painting", "painted", "illustration", "artwork", "drawing", "cartoon",
    "product label", "advertisement", "screenshot", "graphic",
    "logo", "circuit", "chip", "microchip", "user interface", "ui",
    "poster", "banner", "sign", "menu", "diagram", "chart",
    "website",
    # Abstract/synthetic patterns
    "gradient", "checkered", "grid pattern", "striped pattern",
    "blank sheet", "dots pattern",
    "filled with various colors", "filled with different colors",
    "colorful geometric",
]
DIGITAL_BACKGROUND_KEYWORDS = [
    "background with a black border", "background with a white border",
    "solid background", "plain background",
    "background with the words", "colorful circle",
]
CAMERA_INDICATORS = [
    "man ", "woman ", "person ", "people ", "child ", "dog ", "cat ",
    "standing", "walking", "sitting", "looking at", "field of",
    "mountains", "river", "ocean", "beach", "forest", "sky",
    "building", "street", "car ", "tree", "flower", "bird",
]

def classify_camera_digital(blip_caption):
    lower = blip_caption.lower()
    words = lower.split()
    for kw in DIGITAL_KEYWORDS:
        if kw in lower:
            if ' ' in kw:
                return "🖥️ Digital / Screenshot"
            else:
                if kw in words:
                    return "🖥️ Digital / Screenshot"
    has_camera_indicator = any(ci in lower for ci in CAMERA_INDICATORS)
    if not has_camera_indicator:
        if any(kw in lower for kw in DIGITAL_BACKGROUND_KEYWORDS):
            return "🖥️ Digital / Screenshot"
        if "background" in lower and not any(
            obj in words for obj in ["mug", "bottle", "phone", "book", "chair", "table",
                                      "person", "people", "man", "woman", "child", "animal"]
        ):
            return "🖥️ Digital / Screenshot"
    return "📷 Camera Photo"

def analyze_metadata(image_path):
    from PIL import Image
    import os
    img = _load_image_safely(image_path)
    w, h = img.size
    kb = os.path.getsize(image_path) // 1024
    ratio = w / h
    gray = img.convert('L')
    px = list(gray.getdata())
    avg_brightness = sum(px) / len(px)
    return {
        "dimensions": f"{w}x{h}", "ratio": round(ratio, 2),
        "file_size_kb": kb, "mode": img.mode,
        "avg_brightness": round(avg_brightness), "is_dark": avg_brightness < 60,
    }

# ═══════════════════════════════════════════════════════════
# DETAILED DESCRIPTION GENERATOR
# ═══════════════════════════════════════════════════════════

def generate_detailed_description(blip_caption, labels=None, metadata=None):
    """
    Generates a rich multi-paragraph description by combining
    BLIP's caption with MAX classifier labels and image metadata.
    """
    if labels is None:
        try:
            from max_classifier import classify_image
            labels = classify_image(blip_caption)
        except ImportError:
            return blip_caption
    
    source = labels.get('source', ['unknown'])
    source = source[0] if source else 'unknown'
    subjects = labels.get('subject', [])
    colors = labels.get('color', [])
    text_info = labels.get('text', [])
    setting = labels.get('setting', [])
    environment = labels.get('environment', [])
    composition = labels.get('composition', [])
    mood = labels.get('mood', [])
    pattern = labels.get('pattern', [])
    material = labels.get('material', [])
    style = labels.get('style', [])
    lighting = labels.get('lighting', [])
    weather = labels.get('weather', [])
    time_of_day = labels.get('time_of_day', [])
    
    # ── Build rich, flowing description ──
    paragraphs = []
    
    # PARAGRAPH 1: Main subject + type
    p1 = []
    p1.append(blip_caption.capitalize() + ".")
    
    source_descriptions = {
        'photo': 'This appears to be a photograph or photorealistic image.',
        'digital_abstract': 'This is a digital or computer-generated image.',
        'painting': 'This is a painting.',
        'drawing': 'This is a drawing or sketch.',
        'illustration': 'This is an illustration.',
        'diagram': 'This is a diagram, chart, or infographic.',
        'screenshot': 'This appears to be a screenshot or user interface.',
        'map': 'This is a map or cartographic image.',
    }
    if source in source_descriptions:
        p1.append(source_descriptions[source])
    
    paragraphs.append(" ".join(p1))
    
    # PARAGRAPH 2: Visual details (color, lighting, composition, patterns)
    p2 = []
    
    color_map = {
        'warm_tones': 'warm tones of red, orange, and yellow',
        'cool_tones': 'cool tones of blue, purple, and teal',
        'dark_dim': 'a dark, subdued palette with deep shadows',
        'bright_light': 'a bright, well-illuminated palette',
        'vibrant_colorful': 'vibrant, saturated colors throughout',
        'monochrome_bw': 'a black and white or grayscale tonality',
        'pastel': 'soft, muted pastel hues',
        'high_contrast': 'strong contrast between light and dark areas',
    }
    color_parts = [color_map[c] for c in colors if c in color_map]
    if color_parts:
        p2.append("The color palette is dominated by " + "; ".join(color_parts) + ".")
    
    if lighting:
        p2.append(f"The lighting is {', '.join(l.replace('_',' ') for l in lighting)}.")
    
    if composition:
        p2.append(f"The composition uses a {', '.join(c.replace('_',' ') for c in composition)} perspective.")
    
    if pattern:
        p2.append(f"Visible patterns and structures include {', '.join(p.replace('_',' ') for p in pattern)}.")
    
    if weather:
        p2.append(f"Weather conditions: {', '.join(w.replace('_',' ') for w in weather)}.")
    
    if time_of_day:
        p2.append(f"The time of day appears to be {', '.join(t.replace('_',' ') for t in time_of_day)}.")
    
    if p2:
        paragraphs.append(" ".join(p2))
    
    # PARAGRAPH 3: Subject, setting, environment, mood
    p3 = []
    
    if subjects:
        subject_str = ", ".join(s.replace('_', ' ') for s in subjects)
        p3.append(f"The primary subject matter includes: {subject_str}.")
    
    if setting:
        p3.append(f"The setting is {', '.join(s.replace('_',' ') for s in setting)}.")
    
    if environment:
        p3.append(f"The environment can be characterized as {', '.join(e.replace('_',' ') for e in environment)}.")
    
    if material:
        p3.append(f"Notable materials and textures: {', '.join(m.replace('_',' ') for m in material)}.")
    
    if style:
        p3.append(f"The visual style is {', '.join(s.replace('_',' ') for s in style)}.")
    
    if mood:
        p3.append(f"The overall mood conveyed is {', '.join(m.replace('_',' ') for m in mood)}.")
    
    if p3:
        paragraphs.append(" ".join(p3))
    
    # PARAGRAPH 4: Text content and metadata
    p4 = []
    
    if text_info:
        if 'has_text' in text_info:
            if 'text_heavy' in text_info:
                p4.append("The image contains substantial visible text or typography.")
                if 'sign_present' in text_info:
                    p4.append("Signs, labels, or banners with text are present.")
            else:
                p4.append("Some visible text or lettering is present in the image.")
        else:
            p4.append("No visible text, labels, or lettering is present.")
    else:
        p4.append("No visible text or lettering is present.")
    
    if metadata:
        dims = metadata.get('dimensions', '')
        kb = metadata.get('file_size_kb', '')
        if dims:
            p4.append(f"Image dimensions: {dims} ({kb}KB).")
    
    if p4:
        paragraphs.append(" ".join(p4))
    
    return "\n\n".join(paragraphs)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='100% Local Image Analysis (OCR + Vision)')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--ocr', default='doctr', choices=['doctr', 'easyocr', 'all', 'none'])
    parser.add_argument('--no-ocr', action='store_true', help='Skip OCR entirely')
    parser.add_argument('--no-vision', action='store_true', help='Skip BLIP captioning')
    parser.add_argument('--drive', default=None, help='Storage drive (e.g., D:, E:, /mnt/data). Auto-detected if not set.')
    parser.add_argument('--force-cpu', action='store_true', help='Disable GPU even if available')
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    # ── Set up environment ──
    drive = args.drive if args.drive else get_storage_drive(auto=True)
    config = setup_environment(drive)
    has_gpu, device, gpu_name = gpu_available()

    do_ocr = not args.no_ocr
    do_vision = not args.no_vision
    if args.ocr == 'none':
        do_ocr = False

    print("=" * 70)
    print(f"  📷 {os.path.basename(image_path)}")
    print(f"  🏠 100% LOCAL — zero API calls, zero rate limits, zero cost")
    print(f"  💾 Storage: {config['drive']}")
    print(f"  🖥️  GPU: {gpu_info_str()}")
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
            # Use MAX classifier for camera/digital (34 dimensions, space-aware)
            try:
                from max_classifier import classify_camera_digital as max_cd
                img_type = "🖥️ Digital / Screenshot" if max_cd(r['caption']) == "digital" else "📷 Camera Photo"
            except ImportError:
                img_type = classify_camera_digital(r['caption'])
            print(f"  📸 Type: {img_type}")
        except Exception as e:
            print(f"❌ {e}")
    else:
        print()

    # ── OCR ──
    if do_ocr:
        engines = ['doctr', 'easyocr'] if args.ocr == 'all' else [args.ocr]
        for eng in engines:
            print(f"  🔍 [{eng}]...", end=" ", flush=True)
            try:
                if eng == 'doctr':
                    r = run_doctr(image_path, force_cpu=args.force_cpu)
                else:
                    r = run_easyocr(image_path, force_cpu=args.force_cpu)
                gpu_tag = f" [GPU]" if r.get('device') != 'cpu' else ""
                print(f"{r['word_count']} words ({r['avg_confidence']:.0%} conf) in {r['time_seconds']}s{gpu_tag}")
                results[eng] = r
            except Exception as e:
                print(f"❌ {e}")

    # ═══════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("                   ANALYSIS REPORT")
    print(f"{'=' * 70}")

    if 'blip' in results:
        print(f"\n  ── What's in this image? (BLIP) ──")
        print(f"  📝 {results['blip']['caption']}")
        
        # ── Detailed description (BLIP + MAX classifier) ──
        try:
            from max_classifier import classify_image
            from describe_engine import generate_detailed_description
            labels = classify_image(results['blip']['caption'])
            detailed = generate_detailed_description(results['blip']['caption'], labels, meta)
            print(f"\n  ── Detailed Description ──")
            # Word-wrap the detailed description
            import textwrap
            for line in textwrap.wrap(detailed, width=65):
                print(f"  {line}")
        except ImportError:
            pass  # max_classifier not available — skip detailed description

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

    if 'doctr' in results and 'easyocr' in results:
        d, e = results['doctr'], results['easyocr']
        print(f"\n  ── OCR Comparison ──")
        print(f"  {'':<20} {'DocTR':>10} {'EasyOCR':>10}")
        print(f"  {'Words':<20} {d['word_count']:>10} {e['word_count']:>10}")
        print(f"  {'Confidence':<20} {d['avg_confidence']:>9.0%} {e['avg_confidence']:>9.0%}")
        print(f"  {'Speed':<20} {d['time_seconds']:>9}s {e['time_seconds']:>9}s")

    print(f"\n  🏠 100% LOCAL — zero API calls, zero rate limits, zero cost")
    print(f"{'=' * 70}")
