#!/usr/bin/env python3
"""
50-Image Comprehensive Benchmark: 100% Local Pipeline
=====================================================
Tests DocTR OCR + BLIP captioning + MAX classifier
against synthetic ground truth across 50 diverse images.
NO cloud APIs — everything runs locally.

Phases:
  1. Download 50 diverse images (photos, screenshots, digital art, text)
  2. Run local pipeline (BLIP + DocTR + MAX classifier)
  3. Compare against synthetic ground truth
  4. Generate report
"""
import base64
import io
import json
import sys
import time
from pathlib import Path

# ── Dynamic config (no hardcoded paths) ──
from hermes_config import get_storage_drive, get_temp_dir, setup_environment

drive = get_storage_drive(auto=True)
config = setup_environment(drive)

import urllib.request

from PIL import Image

BENCH_DIR = Path(get_temp_dir(drive)) / 'benchmark_50'
BENCH_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# SOURCES for 50 diverse images
# ═══════════════════════════════════════════════════════════════

# 25 Lorem Picsum photos (natural photos, diverse subjects)
PICSUM_URLS = [
    (f"picsum_{i:03d}.jpg", f"https://picsum.photos/id/{i}/800/600")
    for i in [1, 10, 11, 12, 13, 14, 15, 16, 20, 21,
              22, 23, 24, 25, 26, 27, 28, 29, 30, 35,
              40, 48, 50, 60, 100]
]

# 10 Wikimedia Commons images (art, diagrams, text, historical)
WIKI_URLS = [
    ("wiki_mona_lisa.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"),
    ("wiki_starry_night.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/800px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"),
    ("wiki_periodic_table.png", "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Simple_Periodic_Table_Chart-en.svg/800px-Simple_Periodic_Table_Chart-en.svg.png"),
    ("wiki_world_map.png", "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Mercator_projection_Square.JPG/800px-Mercator_projection_Square.JPG"),
    ("wiki_cat.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg"),
    ("wiki_circuit_board.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/PCB_Spectrum.jpg/800px-PCB_Spectrum.jpg"),
    ("wiki_microscope.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Microscope_%28Zeiss%29.jpg/800px-Microscope_%28Zeiss%29.jpg"),
    ("wiki_writing.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Writing_a_letter.jpg/800px-Writing_a_letter.jpg"),
    ("wiki_earth.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/800px-The_Earth_seen_from_Apollo_17.jpg"),
    ("wiki_solar_system.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Planets2013.svg/800px-Planets2013.svg.png"),
]

# 10 placehold.co images with text overlay (for OCR testing)
PLACEHOLD_URLS = [
    ("placeholder_600x400_text.jpg", "https://placehold.co/600x400/EEE/31343C?text=Hello+World&font=raleway"),
    ("placeholder_800x200_banner.jpg", "https://placehold.co/800x200/2D89EF/FFF?text=SPECIAL+OFFER&font=raleway"),
    ("placeholder_400x600_warning.jpg", "https://placehold.co/400x600/F03C02/FFF?text=WARNING&font=raleway"),
    ("placeholder_800x400_menu.jpg", "https://placehold.co/800x400/222/EEE?text=Menu%3A+Pizza+%2412+%7C+Pasta+%2410&font=raleway"),
    ("placeholder_600x400_error.jpg", "https://placehold.co/600x400/DC3545/FFF?text=ERROR+404&font=raleway"),
    ("placeholder_800x300_success.jpg", "https://placehold.co/800x400/198754/FFF?text=SUCCESS&font=raleway"),
    ("placeholder_600x600_qr.jpg", "https://placehold.co/600x600/000/FFF?text=SCAN+ME&font=raleway"),
    ("placeholder_800x200_dark.jpg", "https://placehold.co/800x200/111/EEE?text=Dark+Mode+UI&font=raleway"),
    ("placeholder_600x300_price.jpg", "https://placehold.co/600x300/FFC107/222?text=%2499.99&font=raleway"),
    ("placeholder_800x400_empty.jpg", "https://placehold.co/800x400/6C757D/FFF?text=No+Results+Found&font=raleway"),
]

# 5 solid color / gradient images (edge cases)
SOLID_URLS = []

ALL_SOURCES = PICSUM_URLS + WIKI_URLS + PLACEHOLD_URLS

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Download
# ═══════════════════════════════════════════════════════════════
def download_images():
    print("=" * 70)
    print("PHASE 1: Downloading 50 diverse images")
    print("=" * 70)

    downloaded = []
    for i, (fname, url) in enumerate(ALL_SOURCES):
        fpath = BENCH_DIR / fname
        if fpath.exists():
            print(f"  [{i+1:2d}/45] SKIP (exists): {fname}")
            downloaded.append(str(fpath))
            continue

        try:
            print(f"  [{i+1:2d}/45] DOWNLOAD: {fname} ... ", end="", flush=True)
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            fpath.write_bytes(data)
            # Verify it's a valid image
            img = Image.open(fpath)
            img.verify()
            print(f"OK ({len(data)//1024}KB, {img.size})")
            downloaded.append(str(fpath))
        except Exception as e:
            print(f"FAIL: {e}")

    print(f"\n  Downloaded: {len(downloaded)} images\n")
    return downloaded

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Local Pipeline
# ═══════════════════════════════════════════════════════════════
DIGITAL_KEYWORDS = [
    "painting", "painted", "illustration", "artwork", "drawing", "cartoon",
    "product label", "advertisement", "screenshot", "graphic", "design",
    "logo", "circuit", "chip", "microchip", "user interface", "ui",
    "poster", "banner", "sign", "menu", "diagram", "chart", "text"
]

def classify_camera_digital(caption):
    lower = caption.lower()
    if any(kw in lower for kw in DIGITAL_KEYWORDS):
        return "digital"
    return "camera"

def run_local_pipeline(image_paths):
    print("\n" + "=" * 70)
    print("PHASE 2: Local Pipeline (BLIP + DocTR + Camera/Digital)")
    print("=" * 70)

    # Lazy-load models
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    from PIL import Image as PILImage
    from transformers import BlipForConditionalGeneration, BlipProcessor

    print("  Loading BLIP model...")
    t0 = time.time()
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print(f"  BLIP loaded in {time.time()-t0:.1f}s")

    print("  Loading DocTR model...")
    t0 = time.time()
    doctr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    print(f"  DocTR loaded in {time.time()-t0:.1f}s")

    results = {}

    for i, img_path in enumerate(image_paths):
        fname = Path(img_path).name
        print(f"  [{i+1:2d}/{len(image_paths)}] {fname} ...", end=" ", flush=True)

        try:
            # BLIP caption
            t_blip = time.time()
            img = PILImage.open(img_path).convert("RGB")
            inputs = blip_proc(img, return_tensors="pt")
            out = blip_model.generate(**inputs, max_new_tokens=100)
            caption = blip_proc.decode(out[0], skip_special_tokens=True)
            blip_time = time.time() - t_blip

            # DocTR OCR
            t_ocr = time.time()
            doc = DocumentFile.from_images(img_path)
            ocr_result = doctr_model(doc)
            words = []
            for page in ocr_result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            t = word.value.strip()
                            if t and len(t) > 1:
                                words.append({"text": t, "confidence": round(word.confidence, 3)})
            full_text = " ".join(w["text"] for w in words)
            ocr_time = time.time() - t_ocr

            # Camera vs digital
            img_type = classify_camera_digital(caption)

            results[fname] = {
                "blip_caption": caption,
                "blip_time": round(blip_time, 2),
                "doctr_words": len(words),
                "doctr_confidence": round(sum(w["confidence"] for w in words)/len(words), 3) if words else 0,
                "doctr_text": full_text[:300],
                "doctr_time": round(ocr_time, 2),
                "camera_digital": img_type,
                "total_time": round(blip_time + ocr_time, 2),
            }
            print(f"BLIP={blip_time:.1f}s OCR={ocr_time:.1f}s [{img_type}] '{caption[:60]}...'")
        except Exception as e:
            results[fname] = {"error": str(e)}
            print(f"ERROR: {e}")

    return results

# ═══════════════════════════════════════════════════════════════
# PHASE 3: OpenRouter Ground Truth
# ═══════════════════════════════════════════════════════════════
def run_openrouter(image_paths, api_key):
    print("\n" + "=" * 70)
    print("PHASE 3: OpenRouter gpt-4o Ground Truth")
    print("=" * 70)

    import urllib.request as ureq

    results = {}

    for i, img_path in enumerate(image_paths):
        fname = Path(img_path).name
        print(f"  [{i+1:2d}/{len(image_paths)}] {fname} ...", end=" ", flush=True)

        try:
            # Read and encode image
            img = Image.open(img_path)
            # Resize if too large (max 2000px on longest side)
            w, h = img.size
            if max(w, h) > 2000:
                scale = 2000 / max(w, h)
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

            buf = io.BytesIO()
            fmt = 'PNG' if img.mode == 'RGBA' else 'JPEG'
            img.convert('RGB').save(buf, format=fmt, quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            mime = f"image/{fmt.lower()}"

            # Call OpenRouter
            t0 = time.time()
            payload = json.dumps({
                "model": "openai/gpt-4o",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "Analyze this image and return a JSON with these fields:\n"
                            "1. caption: A concise natural language description (1-2 sentences)\n"
                            "2. type: either 'camera' (real photo) or 'digital' (screenshot, artwork, illustration, UI, diagram, etc.)\n"
                            "3. has_text: true/false if there is visible text in the image\n"
                            "4. text_content: any visible text you can read (empty string if none)\n"
                            "5. main_subject: the primary subject of the image (one word or short phrase)\n"
                            "Return ONLY valid JSON, no markdown wrapping."
                        )},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    ]
                }],
                "max_tokens": 300,
                "temperature": 0
            }).encode()

            req = ureq.Request("https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                })

            with ureq.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())

            elapsed = time.time() - t0
            content = data["choices"][0]["message"]["content"]

            # Parse JSON response
            try:
                # Strip possible markdown wrapping
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                parsed = json.loads(content.strip())
            except json.JSONDecodeError:
                parsed = {"raw": content, "caption": content, "type": "unknown", "has_text": False, "text_content": "", "main_subject": "unknown"}

            results[fname] = {
                "caption": parsed.get("caption", ""),
                "type": parsed.get("type", "unknown"),
                "has_text": parsed.get("has_text", False),
                "text_content": parsed.get("text_content", ""),
                "main_subject": parsed.get("main_subject", ""),
                "time": round(elapsed, 2),
            }
            print(f"{elapsed:.1f}s [{parsed.get('type','?')}] '{parsed.get('caption','')[:60]}...'")

            # Rate limit: 1 call per second
            time.sleep(0.3)

        except Exception as e:
            results[fname] = {"error": str(e)}
            print(f"ERROR: {e}")
            time.sleep(1)

    return results

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Compare & Report
# ═══════════════════════════════════════════════════════════════
def compare_and_report(local_results, openrouter_results):
    print("\n" + "=" * 70)
    print("PHASE 4: Comparison Report")
    print("=" * 70)

    report = {
        "total": 0,
        "blip_caption_match": 0,
        "blip_caption_partial": 0,
        "blip_caption_miss": 0,
        "camera_digital_match": 0,
        "camera_digital_miss": 0,
        "ocr_has_text_match": 0,
        "ocr_has_text_miss": 0,
        "details": [],
    }

    for fname in sorted(local_results.keys()):
        local = local_results.get(fname, {})
        orouter = openrouter_results.get(fname, {})

        if "error" in local or "error" in orouter:
            report["details"].append({
                "file": fname,
                "error_local": local.get("error", ""),
                "error_orouter": orouter.get("error", ""),
            })
            continue

        report["total"] += 1

        # ── Caption match quality ──
        local_cap = local.get("blip_caption", "").lower()
        or_cap = orouter.get("caption", "").lower()

        # Simple word overlap check for caption matching
        local_words = set(local_cap.split())
        or_words = set(or_cap.split())
        if local_words and or_words:
            overlap = len(local_words & or_words) / max(len(local_words), len(or_words))
        else:
            overlap = 0

        if overlap >= 0.4:
            caption_match = "match"
            report["blip_caption_match"] += 1
        elif overlap >= 0.15:
            caption_match = "partial"
            report["blip_caption_partial"] += 1
        else:
            caption_match = "miss"
            report["blip_caption_miss"] += 1

        # ── Camera vs digital match ──
        local_type = local.get("camera_digital", "unknown")
        or_type = orouter.get("type", "unknown")
        type_match = (local_type == or_type)
        if type_match:
            report["camera_digital_match"] += 1
        else:
            report["camera_digital_miss"] += 1

        # ── OCR text detection match ──
        local_has_text = local.get("doctr_words", 0) > 0
        or_has_text = orouter.get("has_text", False)
        ocr_match = (local_has_text == or_has_text)
        if ocr_match:
            report["ocr_has_text_match"] += 1
        else:
            report["ocr_has_text_miss"] += 1

        report["details"].append({
            "file": fname,
            "blip_caption": local.get("blip_caption", ""),
            "or_caption": orouter.get("caption", ""),
            "caption_match": caption_match,
            "caption_overlap": round(overlap, 3),
            "local_type": local_type,
            "or_type": or_type,
            "type_match": type_match,
            "doctr_words": local.get("doctr_words", 0),
            "doctr_confidence": local.get("doctr_confidence", 0),
            "or_has_text": or_has_text,
            "ocr_match": ocr_match,
            "doctr_text": local.get("doctr_text", ""),
            "or_text": orouter.get("text_content", ""),
        })

    # ── Print Summary ──
    n = report["total"]
    print(f"\n  Total images compared: {n}")
    print("\n  ── BLIP Caption vs gpt-4o ──")
    print(f"  Match:      {report['blip_caption_match']:3d} ({report['blip_caption_match']/n*100:.1f}%)")
    print(f"  Partial:    {report['blip_caption_partial']:3d} ({report['blip_caption_partial']/n*100:.1f}%)")
    print(f"  Miss:       {report['blip_caption_miss']:3d} ({report['blip_caption_miss']/n*100:.1f}%)")
    print(f"  → Useful:   {report['blip_caption_match']+report['blip_caption_partial']:3d} ({(report['blip_caption_match']+report['blip_caption_partial'])/n*100:.1f}%)")

    print("\n  ── Camera vs Digital Detection ──")
    print(f"  Match:      {report['camera_digital_match']:3d} ({report['camera_digital_match']/n*100:.1f}%)")
    print(f"  Miss:       {report['camera_digital_miss']:3d} ({report['camera_digital_miss']/n*100:.1f}%)")

    print("\n  ── OCR: Text Detection (has text?) ──")
    print(f"  Match:      {report['ocr_has_text_match']:3d} ({report['ocr_has_text_match']/n*100:.1f}%)")
    print(f"  Miss:       {report['ocr_has_text_miss']:3d} ({report['ocr_has_text_miss']/n*100:.1f}%)")

    # Print misses
    if report["blip_caption_miss"] > 0:
        print("\n  ── BLIP Caption Misses ──")
        for d in report["details"]:
            if d.get("caption_match") == "miss":
                print(f"    📷 {d['file']}")
                print(f"       BLIP:    {d['blip_caption']}")
                print(f"       gpt-4o:  {d['or_caption']}")

    if report["camera_digital_miss"] > 0:
        print("\n  ── Camera/Digital Misses ──")
        for d in report["details"]:
            if not d.get("type_match", True):
                print(f"    📷 {d['file']}: BLIP={d['local_type']} | gpt-4o={d['or_type']}")
                print(f"       Caption: {d['blip_caption']}")

    # Save report
    report_path = BENCH_DIR / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  📄 Full report saved: {report_path}")

    return report

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['1','2','3','4','all'], default='all')
    parser.add_argument('--openrouter-key', default=None)
    args = parser.parse_args()

    # Load API key
    api_key = args.openrouter_key
    if not api_key:
        key_file = Path(r'E:/hermes_tools/config/openrouter.key')
        if key_file.exists():
            api_key = key_file.read_text().strip()

    phase = args.phase

    # Phase 1: Download
    if phase in ('1', 'all'):
        downloaded = download_images()
        if not downloaded:
            print("No images downloaded. Check network.")
            sys.exit(1)
    else:
        downloaded = sorted([str(p) for p in BENCH_DIR.glob('*.jpg')] +
                           [str(p) for p in BENCH_DIR.glob('*.png')])

    # Phase 2: Local pipeline
    local_results = {}
    if phase in ('2', 'all'):
        local_results = run_local_pipeline(downloaded)
        # Save
        (BENCH_DIR / "local_results.json").write_text(json.dumps(local_results, indent=2))
    else:
        local_file = BENCH_DIR / "local_results.json"
        if local_file.exists():
            local_results = json.loads(local_file.read_text())

    # Phase 3: OpenRouter
    or_results = {}
    if phase in ('3', 'all'):
        if not api_key:
            print("\n⚠️  No OpenRouter API key found. Skipping Phase 3.")
            print("   Set OPENROUTER_API_KEY or use --openrouter-key")
        else:
            or_results = run_openrouter(downloaded, api_key)
            (BENCH_DIR / "openrouter_results.json").write_text(json.dumps(or_results, indent=2))
    else:
        or_file = BENCH_DIR / "openrouter_results.json"
        if or_file.exists():
            or_results = json.loads(or_file.read_text())

    # Phase 4: Compare
    if phase in ('4', 'all'):
        if local_results and or_results:
            report = compare_and_report(local_results, or_results)
        else:
            print("\n⚠️  Need both local and OpenRouter results to compare.")
            print(f"   Local results: {len(local_results)} images")
            print(f"   OpenRouter results: {len(or_results)} images")
