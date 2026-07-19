#!/usr/bin/env python3
"""
Image Analysis — 100% LOCAL Pipeline (Zero API Calls)
=====================================================
Vision (choose one):
  LLaVA-1.5-7B (4-bit GPU) — rich multi-paragraph descriptions, ~4GB VRAM
  BLIP-base               — fast short captions, works on CPU, ~1GB model
OCR:      DocTR (primary) + EasyOCR (backup, GPU auto-detect)
Analysis: Pixel colors + MAX classifier (34 dims, 11.5M keywords)
=====================================================

Usage:
  python analyze_image.py <image_path>                    # Auto engine (config or GPU detect)
  python analyze_image.py <image_path> --engine llava     # Force LLaVA
  python analyze_image.py <image_path> --engine blip      # Force BLIP (CPU-friendly)
  python analyze_image.py <image_path> --ocr all           # Compare DocTR vs EasyOCR
  python analyze_image.py <image_path> --no-vision         # OCR only
  python analyze_image.py <image_path> --no-ocr            # Vision only
  python analyze_image.py <image_path> --drive D:          # Override storage drive
  python analyze_image.py <image_path> --force-cpu         # Disable GPU
  python analyze_image.py --show-engines                   # Show GPU and recommendation

Storage: Auto-detects available drives (avoids C: on Windows).
GPU: LLaVA needs 6GB+ VRAM. BLIP works on CPU.
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
        device = "cpu"
    
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    if has_gpu:
        model = model.cuda()
    
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
# VISION ENGINE: LLaVA-1.5-7B (4-bit GPU) or BLIP-base (CPU)
# ═══════════════════════════════════════════════════════════

def run_llava(image_path, detail="rich"):
    """Run LLaVA vision model for detailed image descriptions."""
    from llava_engine import get_llava
    engine = get_llava()
    result = engine.describe(image_path, detail=detail)
    return {
        "engine": "LLaVA-1.5-7B (4-bit GPU)",
        "caption": result['description'],
        "time_seconds": result['time_seconds'],
        "vram_gb": result.get('vram_gb', 0),
    }

def run_blip(image_path):
    """Run BLIP-base for fast short captions. Works on CPU."""
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch as _torch
    
    t0 = time.time()
    img = _load_image_safely(image_path)
    
    # Use correct HF cache path
    cache_dir = os.environ.get('HF_HOME', None)
    kwargs = {'cache_dir': cache_dir} if cache_dir else {}
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", **kwargs)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", **kwargs)
    
    # Use GPU if available for BLIP too
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = processor(img, text="a picture of", return_tensors="pt").to(device)
    
    with _torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, num_beams=3)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    elapsed = time.time() - t0
    
    return {
        "engine": "BLIP-base (Salesforce/blip-image-captioning-base)",
        "caption": caption,
        "time_seconds": round(elapsed, 2),
        "vram_gb": round(_torch.cuda.memory_allocated(0) / 1024**3, 2) if device == "cuda" else 0,
    }

def run_vision(image_path, engine="auto"):
    """Run the selected vision engine. Engine: 'llava', 'blip', or 'auto'."""
    from engine_config import get_engine_choice
    
    choice = get_engine_choice(force=engine if engine != "auto" else None)
    
    if choice == "llava":
        return run_llava(image_path)
    else:
        return run_blip(image_path)

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
    parser.add_argument('image', nargs='?', help='Path to image file')
    parser.add_argument('--ocr', default='doctr', choices=['doctr', 'easyocr', 'all', 'none'])
    parser.add_argument('--no-ocr', action='store_true', help='Skip OCR entirely')
    parser.add_argument('--no-vision', action='store_true', help='Skip vision/captioning')
    parser.add_argument('--engine', default='auto', choices=['auto', 'llava', 'blip'],
                       help='Vision engine: llava (rich), blip (fast), auto (detect)')
    parser.add_argument('--show-engines', action='store_true',
                       help='Show GPU info and engine recommendation, then exit')
    parser.add_argument('--drive', default=None, help='Storage drive (e.g., D:, E:, /mnt/data). Auto-detected if not set.')
    parser.add_argument('--force-cpu', action='store_true', help='Disable GPU even if available')
    args = parser.parse_args()
    
    # --show-engines: display recommendation and exit
    if args.show_engines:
        from engine_config import print_engine_recommendation
        print_engine_recommendation()
        sys.exit(0)
    
    if not args.image:
        parser.error("image path required (or use --show-engines)")
    
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

    # ── LLaVA / BLIP Vision ──
    if do_vision:
        engine_name = args.engine if hasattr(args, 'engine') else 'auto'
        print(f"\n  🧠 Vision ({engine_name})...", end=" ", flush=True)
        try:
            r = run_vision(image_path, engine=engine_name)
            print(f"done ({r['time_seconds']}s, {r.get('vram_gb',0):.1f}GB VRAM)")
            results['vision'] = r
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

    if 'vision' in results:
        print(f"\n  ── LLaVA Description ──")
        caption = results['vision']['caption']
        # Truncate very long descriptions for display
        if len(caption) > 800:
            caption = caption[:800] + "..."
        print(f"  📝 {caption}")
        
        # Detailed structured description
        try:
            from max_classifier import classify_image
            from describe_engine import generate_detailed_description
            labels = classify_image(caption)
            detailed = generate_detailed_description(caption, labels, meta)
            
            # Pixel analysis for color/motion details
            try:
                from pixel_analysis import analyze_pixels, pixel_analysis_to_text
                pixel = analyze_pixels(image_path)
                pixel_txt = pixel_analysis_to_text(pixel)
                print(f"\n  ── Pixel Analysis ──")
                for line in textwrap.wrap(pixel_txt, width=65):
                    print(f"  {line}")
            except ImportError:
                pass
            print(f"\n  ── Structured Summary ──")
            import textwrap
            for line in textwrap.wrap(detailed, width=65):
                print(f"  {line}")
        except ImportError:
            pass

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
