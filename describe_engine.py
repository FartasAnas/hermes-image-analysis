"""
Synthesis Engine — Unified Multi-Engine Output Generator
=========================================================
Fuses vision (BLIP/LLaVA), OCR (DocTR), pixel analysis, and MAX classifier
outputs into a SINGLE flowing natural-language paragraph.

Key capabilities:
  - Cross-engine intelligence: detects technical screenshots from OCR text
  - Hallucination suppression: filters LLaVA visual false-positives on terminal shots
  - Unified JSON state: all engine payloads merged into one structured object
  - Debug mode: raw engine outputs available behind --debug flag

Architecture:
  analyze_image.py → runs engines → builds JSON state → describe_engine.synthesize()
"""

# ═══════════════════════════════════════════════════════════
# TECHNICAL SCREENSHOT DETECTION
# ═══════════════════════════════════════════════════════════

TECH_SIGNALS = [
    # Shell / CLI commands
    "curl", "bash", "npm", "pip", "uv", "git", "python", "docker",
    "sudo", "apt-get", "chmod", "ssh", "scp", "wget", "make", "gcc",
    # Windows paths and commands
    "C:\\", "D:\\", "E:\\", "C:/", "D:/", "E:/",
    "cmd.exe", "powershell", "msbuild", "dotnet",
    # Programming tokens (space after keyword avoids matching "define"/"important")
    "def ", "import ", "class ", "function", "const ", "let ", "var ",
    "return", "print(", "console.", "<?php", "#!/",
    "True", "False", "None", "__", "lambda",
    # Error / log patterns
    "error:", "warning:", "traceback", "exception",
    "stack trace", "line ", "column ",
    "failed", "timeout", "connection refused",
    # Package managers / build tools
    "cargo", "go build", "mvn ", "gradle", "yarn", "npx",
    "node ", "tsc ", "webpack", "eslint",
    # Config / env
    "export ", "PATH=", "HOME=", "env ",
    # Database / API
    "SELECT ", "INSERT ", "UPDATE ", "DELETE ",
    "mysql", "postgres", "mongodb", "redis",
    "localhost", "127.0.0.1", ":8080", ":3000",
    # Hermes-specific
    "hermes", "analyze_image.py", "engine_config",
]

# LLaVA hallucination patterns to suppress on technical screenshots
VISUAL_HALLUCINATION_SIGNALS = [
    "face", "person", "people", "man", "woman", "child",
    "outdoor", "nature", "landscape", "mountain", "forest",
    "beach", "ocean", "river", "field of", "sky with",
    "animal", "bird", "dog", "cat", "tree", "flower",
    "building", "street", "car ",
    "headphone", "phone", "cell phone",
    "art", "artwork", "painting", "drawing",
]


def detect_technical_screenshot(ocr_text):
    """
    Analyze OCR text for system/terminal signals.

    Returns (is_technical: bool, confidence: float, matched_signals: list)
    """
    if not ocr_text or not ocr_text.strip():
        return False, 0.0, []

    lower = ocr_text.lower()
    matched = []

    for signal in TECH_SIGNALS:
        if signal.lower() in lower:
            matched.append(signal)

    if not matched:
        return False, 0.0, []

    # Confidence: ratio of matched signals, capped at 1.0
    confidence = min(len(matched) / 3.0, 1.0)
    return True, confidence, matched


def _suppress_hallucinations(description, ocr_text):
    """
    If the image is a technical screenshot, strip visual hallucination
    patterns from the vision description and replace with terminal framing.
    """
    is_tech, confidence, signals = detect_technical_screenshot(ocr_text)

    if not is_tech or confidence < 0.3:
        return description, False

    lower_desc = description.lower()
    has_hallucination = any(
        signal in lower_desc for signal in VISUAL_HALLUCINATION_SIGNALS
    )

    if not has_hallucination:
        return description, is_tech

    # Build sanitized technical description
    known_cmds = {"curl", "bash", "git", "python", "npm", "pip", "docker",
                   "uv", "node", "cargo", "mvn", "gradle"}
    cmd_hints = [s.strip() for s in signals if s.strip() in known_cmds]

    if cmd_hints:
        context = f"This appears to be a terminal or command-line screenshot showing output from {'/'.join(cmd_hints[:3])}."
    else:
        context = "This appears to be a terminal or command-line screenshot."

    return context, is_tech


# ═══════════════════════════════════════════════════════════
# UNIFIED JSON STATE BUILDER
# ═══════════════════════════════════════════════════════════

def build_state(meta, vision_result=None, ocr_result=None,
                pixel_result=None, engine_used="blip"):
    """
    Merge all engine outputs into a single structured JSON state object.

    Args:
        meta: dict from analyze_metadata()
        vision_result: dict from run_vision() / run_llava() / run_blip()
        ocr_result: dict from run_doctr() or run_easyocr()
        pixel_result: dict from analyze_pixels()
        engine_used: 'blip' or 'llava'

    Returns:
        dict with unified state
    """
    state = {
        "meta": meta or {},
        "engine": engine_used,
        "vision": {},
        "ocr": {},
        "pixel": {},
        "classification": {},
        "flags": {},
    }

    # Vision
    if vision_result:
        caption = vision_result.get("caption", "")
        state["vision"] = {
            "caption": caption,
            "engine": vision_result.get("engine", "unknown"),
            "time_s": vision_result.get("time_seconds", 0),
            "vram_gb": vision_result.get("vram_gb", 0),
        }

        # MAX classification
        try:
            from max_classifier import classify_image
            state["classification"] = classify_image(caption)
        except ImportError:
            pass

    # OCR
    if ocr_result:
        full_text = ocr_result.get("full_text", "")
        state["ocr"] = {
            "full_text": full_text,
            "word_count": ocr_result.get("word_count", 0),
            "avg_confidence": ocr_result.get("avg_confidence", 0),
            "engine": ocr_result.get("engine", "unknown"),
            "words": ocr_result.get("words", []),
        }

        # Technical screenshot detection
        is_tech, tech_conf, tech_signals = detect_technical_screenshot(full_text)
        state["flags"]["technical_screenshot"] = is_tech
        state["flags"]["tech_confidence"] = round(tech_conf, 2)
        state["flags"]["tech_signals"] = tech_signals[:10]  # top 10

    # Pixel analysis
    if pixel_result:
        state["pixel"] = {
            "dominant_colors": pixel_result.get("dominant_colors", []),
            "vibrancy": pixel_result.get("vibrancy", ""),
            "brightness_desc": pixel_result.get("brightness_desc", ""),
            "contrast": pixel_result.get("contrast", {}),
            "motion_effect": pixel_result.get("motion_effect", ""),
        }

    return state


# ═══════════════════════════════════════════════════════════
# UNIFIED SYNTHESIS — single flowing paragraph
# ═══════════════════════════════════════════════════════════

def synthesize(state):
    """
    Produce ONE unified natural-language paragraph from all engine outputs.

    This replaces the old fragmented output (separate headings for Vision,
    OCR, Pixel Analysis, Structured Summary) with a single flowing description
    that merges visual content, text extraction, colors, and mood.

    Args:
        state: dict from build_state()

    Returns:
        str: unified description paragraph
    """
    parts = []

    vision = state.get("vision", {})
    ocr = state.get("ocr", {})
    pixel = state.get("pixel", {})
    classification = state.get("classification", {})
    flags = state.get("flags", {})
    meta = state.get("meta", {})

    caption = vision.get("caption", "").strip()
    ocr_text = ocr.get("full_text", "").strip()
    is_tech = flags.get("technical_screenshot", False)

    # ═══ STEP 1: Vision description (with hallucination suppression) ═══
    if caption:
        processed_caption, was_suppressed = _suppress_hallucinations(
            caption, ocr_text
        )
        # Clean up common LLaVA prefixes
        for prefix in ["the image shows ", "the image showcases ",
                       "the image depicts ", "this image shows "]:
            if processed_caption.lower().startswith(prefix):
                processed_caption = processed_caption[len(prefix):]
                processed_caption = processed_caption[0].upper() + processed_caption[1:]
                break
        parts.append(processed_caption.rstrip("."))
    else:
        parts.append("An image")

    # ═══ STEP 2: Source/type context ═══
    source = classification.get("source", ["unknown"])
    source = source[0] if source else "unknown"

    if is_tech:
        pass  # already handled in _suppress_hallucinations
    elif source == "screenshot":
        parts.append("This appears to be a screenshot or interface capture")
    elif source == "digital_abstract":
        parts.append("It is digitally created or computer-generated")
    elif source == "painting":
        parts.append("It is a painting")
    elif source == "illustration":
        parts.append("It is an illustration")
    elif source == "drawing":
        parts.append("It appears to be a drawing or sketch")
    elif source == "diagram":
        parts.append("It is a diagram or chart")

    # ═══ STEP 3: Text content (from OCR) ═══
    if ocr_text and not is_tech:
        word_count = ocr.get("word_count", 0)
        conf = ocr.get("avg_confidence", 0)
        if word_count > 5 and conf > 0.4:
            # Human-readable text found — incorporate a sample
            sample = ocr_text[:200].strip()
            parts.append(f"Visible text reads: \"{sample}\"")
        elif word_count > 0:
            parts.append("Some visible text or labels are present")
    elif is_tech and ocr_text:
        # Technical screenshot — show commands
        sample = ocr_text[:300].strip()
        if len(sample) > 10:
            parts.append(f"The terminal output includes: \"{sample}\"")

    # ═══ STEP 4: Color, light, composition ═══
    color_parts = []
    if pixel.get("vibrancy"):
        color_parts.append(pixel["vibrancy"])
    if pixel.get("brightness_desc"):
        color_parts.append(f"the exposure is {pixel['brightness_desc']}")

    dominant = pixel.get("dominant_colors", [])
    if dominant:
        names = [f"{c['name']} ({c['percentage']:.0f}%)" for c in dominant[:4]
                 if c.get("percentage", 0) > 2]
        if names:
            color_parts.append(f"dominant hues are {', '.join(names)}")

    contrast = pixel.get("contrast", {})
    if contrast.get("level") == "very high":
        color_parts.append("with very high contrast")
    elif contrast.get("level") == "high":
        color_parts.append("with strong contrast")

    if pixel.get("motion_effect"):
        color_parts.append(f"there is a {pixel['motion_effect']}")

    if color_parts:
        prefix = "With " if parts else "The image has "
        parts.append(prefix + "; ".join(color_parts))

    # ═══ STEP 5: Mood, style, setting ═══
    mood = classification.get("mood", [])
    style = classification.get("style", [])
    setting = classification.get("setting", [])

    context = []
    if mood:
        context.append(f"the mood is {', '.join(m.replace('_',' ') for m in mood)}")
    if style:
        context.append(f"the style is {', '.join(s.replace('_',' ') for s in style)}")
    if setting:
        context.append(f"set in a {', '.join(s.replace('_',' ') for s in setting)} context")

    if context:
        parts.append("; ".join(context))

    # ═══ STEP 6: Metadata ═══
    dims = meta.get("dimensions", "")
    kb = meta.get("file_size_kb", 0)
    if dims and kb:
        parts.append(f"{dims}, {kb}KB")

    # ═══ Assembly ═══
    paragraph = ". ".join(p for p in parts if p).replace("..", ".")
    if not paragraph.endswith("."):
        paragraph += "."

    return paragraph


# ═══════════════════════════════════════════════════════════
# DEBUG DUMP — raw engine outputs (behind --debug)
# ═══════════════════════════════════════════════════════════

def debug_dump(state):
    """Print all raw engine outputs in the old fragmented format."""
    import textwrap
    lines = []
    lines.append("=" * 70)
    lines.append("  DEBUG — RAW ENGINE OUTPUTS")
    lines.append("=" * 70)

    vision = state.get("vision", {})
    if vision.get("caption"):
        lines.append(f"\n  ── Vision ({vision.get('engine', '')}) ──")
        cap = vision["caption"]
        if len(cap) > 800:
            cap = cap[:800] + "..."
        lines.append(f"  {cap}")

    ocr = state.get("ocr", {})
    if ocr.get("words"):
        lines.append(f"\n  ── OCR ({ocr.get('engine', '')}) ──")
        lines.append(f"  Words: {ocr['word_count']} | Confidence: {ocr['avg_confidence']:.0%}")
        for w in sorted(ocr["words"], key=lambda x: x["confidence"], reverse=True)[:15]:
            bar = "\u2588" * int(w["confidence"] * 20)
            lines.append(f"    [{w['confidence']:.0%}] {w['text']} {bar}")
        full = ocr.get("full_text", "")
        if full:
            lines.append(f"\n  Full text: {full[:500]}")

    classification = state.get("classification", {})
    if classification:
        lines.append(f"\n  ── Classification ──")
        for dim, labs in sorted(classification.items()):
            if labs:
                lines.append(f"    {dim}: {', '.join(labs)}")

    pixel = state.get("pixel", {})
    if pixel:
        lines.append(f"\n  ── Pixel Analysis ──")
        if pixel.get("dominant_colors"):
            dc = [f"{c['name']} #{c['hex']} {c['percentage']:.0f}%"
                  for c in pixel["dominant_colors"][:5]]
            lines.append(f"    Colors: {', '.join(dc)}")
        if pixel.get("vibrancy"):
            lines.append(f"    Vibrancy: {pixel['vibrancy']}")
        if pixel.get("brightness_desc"):
            lines.append(f"    Exposure: {pixel['brightness_desc']}")
        if pixel.get("motion_effect"):
            lines.append(f"    Motion: {pixel['motion_effect']}")
        c = pixel.get("contrast", {})
        if c:
            lines.append(f"    Contrast: {c.get('level', '?')} (\u03c3={c.get('std_dev', '?')})")

    flags = state.get("flags", {})
    if flags.get("technical_screenshot"):
        lines.append(f"\n  ── Cross-Engine Flags ──")
        lines.append(f"    Technical Screenshot: YES (confidence: {flags.get('tech_confidence', 0):.0%})")
        lines.append(f"    Signals: {', '.join(flags.get('tech_signals', []))}")

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Legacy compatibility wrapper
# ═══════════════════════════════════════════════════════════

def generate_detailed_description(blip_caption, labels=None, metadata=None):
    """
    Legacy wrapper — retained for backward compatibility.
    New code should use build_state() + synthesize() instead.
    """
    state = build_state(
        meta=metadata or {},
        vision_result={"caption": blip_caption, "engine": "BLIP"},
    )
    if labels:
        state["classification"] = labels
    return synthesize(state)
