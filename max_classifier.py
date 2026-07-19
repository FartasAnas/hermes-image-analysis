#!/usr/bin/env python3
"""
MAX Classifier — 34 dimensions, 37K+ keywords (optimized)
=========================================================
The most comprehensive BLIP caption classifier ever built.
34 dimensions covering every aspect of image content detectable from text.

Optimizations (Phase 2):
  - Inverted index: word → {dimension: [labels]} for O(1) lookup per word
  - Lazy loading: mega/super/10M keywords not expanded at import time
  - get_keyword_count() no longer expands all modules eagerly
"""
import os as _os
import sys

# Import the massive keyword database
_scripts_dir = _os.path.dirname(_os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

try:
    from keyword_generator import ALL_DIMENSIONS
except ImportError:
    try:
        from hermes_config import get_scripts_dir, get_storage_drive
        _dyn_dir = get_scripts_dir(get_storage_drive())
        if _dyn_dir not in sys.path:
            sys.path.insert(0, _dyn_dir)
        from keyword_generator import ALL_DIMENSIONS
    except ImportError:
        raise ImportError(
            "Cannot find keyword_generator.py. "
            "Place it in the same directory as max_classifier.py "
            "or set up hermes_config.py for dynamic path detection."
        )

# ═══════════════════════════════════════════════════════════════
# INVERTED INDEX — pre-built at import time for O(1) word lookup
# Structure: {word: {dim_name: [label1, label2]}}
# ═══════════════════════════════════════════════════════════════

def _build_inverted_index(dimensions):
    """
    Build an inverted index: word → {dimension: [labels]}.
    
    Single-word keywords go into the index for O(1) set-membership lookup.
    Multi-word phrases (containing spaces) are kept separate for substring matching.
    
    This replaces the old O(D × L × K) triple-nested loop with:
      - O(W) word-set lookups for single words
      - O(P) substring checks for multi-word phrases
    where W = unique words in caption (~5-15) and P = multi-word phrases.
    """
    single_word_index = {}   # {word: [(dim, label)]}
    multi_word_phrases = []   # [(dim, label, phrase)]

    for dim_name, dim_data in dimensions.items():
        for label, keywords in dim_data.items():
            for kw in keywords:
                if not kw or len(kw) < 2:
                    continue
                if ' ' in kw:
                    # Multi-word phrase — keep for substring matching
                    multi_word_phrases.append((dim_name, label, kw))
                else:
                    # Single word — add to inverted index
                    if kw not in single_word_index:
                        single_word_index[kw] = []
                    single_word_index[kw].append((dim_name, label))

    return single_word_index, multi_word_phrases


# Build the index once at import time
_SINGLE_WORD_INDEX, _MULTI_WORD_PHRASES = _build_inverted_index(ALL_DIMENSIONS)


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION ENGINE (optimized with inverted index)
# ═══════════════════════════════════════════════════════════════

def classify_image(blip_caption):
    """
    Classifies a BLIP caption across all 34 dimensions.
    
    Uses inverted index for O(1) single-word lookups instead of
    the old O(D × L × K) triple-nested loop.
    
    Returns dict: {dimension_name: [matched_labels]}
    """
    lower = blip_caption.lower()
    words = set(lower.split())

    results = {}

    # Initialize all dimensions
    for dim_name in ALL_DIMENSIONS:
        results[dim_name] = []

    # ── SINGLE-WORD MATCHING (O(W) via inverted index) ──
    for word in words:
        if word in _SINGLE_WORD_INDEX:
            for dim_name, label in _SINGLE_WORD_INDEX[word]:
                if label not in results[dim_name]:
                    results[dim_name].append(label)

    # ── MULTI-WORD PHRASE MATCHING (O(P) substring checks) ──
    for dim_name, label, phrase in _MULTI_WORD_PHRASES:
        if phrase in lower:
            if label not in results[dim_name]:
                results[dim_name].append(label)

    # ── DERIVED RULES ──
    _apply_derived_rules(results, lower, words)

    # ── OVERRIDE: Fix camera/digital false positives ──
    source_final = results.get("source", [])
    physical_computer_signals = ["desktop computer", "laptop computer", "computer on",
                                  "computer with", "a computer"]
    if "digital_abstract" in source_final:
        if any(s in lower for s in physical_computer_signals):
            results["source"] = ["photo"]

    space_digital_signals = [
        "of the earth", "of the universe", "in the universe",
        "network of", "connected lines", "nodes and",
        "illustration of", "digital art", "digital illustration",
        "concept art", "render of", "rendering of",
    ]
    if "photo" in source_final:
        if any(s in lower for s in space_digital_signals):
            results["source"] = ["digital_abstract"]

    # Deduplicate
    for dim in results:
        results[dim] = sorted(set(results[dim]))

    return results


def _apply_derived_rules(results, lower, words):
    """Apply cross-dimension inference rules."""

    source = results.get("source", [])
    if not source:
        has_camera = any(ci in lower for ci in [
            "man ", "woman ", "person ", "people ", "child ",
            "standing", "walking", "sitting", "looking at",
            "field of", "mountains", "river", "ocean", "beach",
            "forest", "sky with", "street", "car ", "tree",
            "flower", "bird", "dog ", "cat ", "building",
            "desktop computer", "laptop computer",
        ])

        digital_signals = [
            "background with", "gradient", "pattern", "solid",
            "blank", "checkered", "striped", "polka", "grid",
            "diagram", "chart", "screenshot", "website",
            "menu with", "game over", "flash sale",
            "system error", "price for",
            "two circles", "one red and one blue",
            "circles with", "overlapping",
            "screenshote", "screen with", "screen showing",
            "app interface", "user interface", "ui ",
            "maze", "circular maze", "labyrinth", "puzzle",
            "with a white background", "with a black background",
            "of the earth", "of the universe", "in the universe",
            "network of", "connected lines", "nodes and",
            "lines and dots", "globe with", "planet with",
        ]
        has_digital = any(ds in lower for ds in digital_signals)

        if has_digital and not has_camera:
            results["source"] = ["digital_abstract"]
        elif has_camera:
            results["source"] = ["photo"]
        else:
            art_signals = ["painting", "painted", "drawing", "sketch"]
            if any(a in lower for a in art_signals):
                if "painting" in lower or "painted" in lower:
                    results["source"] = ["painting"]
                else:
                    results["source"] = ["drawing"]
            else:
                results["source"] = ["photo"]

    # Derived relationships
    source_final = results.get("source", [])
    subject = results.get("subject", [])
    environment = results.get("environment", [])

    if "building_structure" in subject and "architectural" not in environment:
        environment.append("architectural")
        results["environment"] = environment

    if ("plant" in subject or "landscape_scenery" in subject) and \
       "natural_wilderness" not in environment and "urban" not in environment:
        environment.append("natural_wilderness")
        results["environment"] = environment

    if "food_drink" in subject and not results.get("setting"):
        results["setting"] = ["indoor"]

    if "person" in subject and not results.get("setting"):
        if any(w in lower for w in ["field", "mountain", "beach", "street", "park"]):
            results["setting"] = ["outdoor"]
        elif any(w in lower for w in ["room", "office", "kitchen", "studio"]):
            results["setting"] = ["indoor"]

    if "vehicle_transport" in subject and not results.get("setting"):
        results["setting"] = ["outdoor"]

    if "dark_dim" in results.get("color", []) and not results.get("time_of_day"):
        if any(w in lower for w in ["night", "moon", "stars", "dark sky"]):
            results["time_of_day"] = ["night"]

    if "warm_tones" in results.get("color", []) and \
       any(w in lower for w in ["sunset", "sunrise", "golden"]):
        results["time_of_day"] = results.get("time_of_day", []) + ["golden_hour"]

    if "has_text" in results.get("text", []) and \
       "digital_abstract" in source_final and \
       "text_heavy" not in results.get("text", []):
        results["text"] = results.get("text", []) + ["text_heavy"]


def classify_camera_digital(blip_caption):
    """Camera vs Digital using the full classifier."""
    labels = classify_image(blip_caption)
    source = labels.get("source", [])

    digital_sources = {
        "painting", "illustration", "drawing", "diagram",
        "screenshot", "map", "digital_abstract",
        "xray_scan", "microscope", "night_vision",
    }

    if digital_sources & set(source):
        return "digital"
    return "camera"


# ═══════════════════════════════════════════════════════════════
# UTILITY (lazy — doesn't import mega modules on load)
# ═══════════════════════════════════════════════════════════════

def get_keyword_count():
    """
    Return total keyword count across all sources.
    
    Now LAZILY imports mega modules to avoid expanding 11.5M keywords
    at import time just for a display number.
    """
    # Base count from ALL_DIMENSIONS (always loaded)
    base = sum(sum(len(v) for v in dim.values()) for dim in ALL_DIMENSIONS.values())

    # These are estimates unless the modules are explicitly imported
    # We avoid importing them eagerly to prevent memory explosion
    total = base

    # Try importing additional sources (cached, won't re-expand if already done)
    try:
        from mega_keywords import generate_all as _mega_gen
        mega_count = sum(len(v) for v in _mega_gen(n_jobs=1).values())
        total += mega_count
    except (ImportError, MemoryError):
        total += 96744  # known estimate

    try:
        from super_scale_keywords import generate_massive as _super_gen
        super_count = len(_super_gen())
        total += super_count
    except (ImportError, MemoryError):
        total += 911504  # known estimate

    try:
        from keywords_10m import generate_10m as _10m_gen
        tenm_count = len(_10m_gen())
        total += tenm_count
    except (ImportError, MemoryError):
        total += 10500000  # known estimate

    return total


def print_classification(caption, detailed=False):
    """Pretty-print classification for a caption."""
    labels = classify_image(caption)

    print(f"\U0001f4f7 '{caption}'")
    print(f"   Camera/Digital: {classify_camera_digital(caption)}")

    for dim, labs in labels.items():
        if labs:
            print(f"   {dim}: {', '.join(labs)}")

    if detailed:
        missing_dims = [d for d in ALL_DIMENSIONS if not labels.get(d)]
        if missing_dims:
            print(f"   (no labels for: {', '.join(missing_dims)})")

    print()


if __name__ == '__main__':
    idx_size = len(_SINGLE_WORD_INDEX)
    phrase_count = len(_MULTI_WORD_PHRASES)
    print(f"MAX Classifier loaded: {idx_size:,} indexed words, {phrase_count:,} phrases, "
          f"{len(ALL_DIMENSIONS)} dimensions\n")

    # Test
    tests = [
        "a yellow background with the words warning high voltage keep",
        "a red barn with a brown roof",
        "a man standing in a field looking at the mountains",
        "a close up of a colorful bird sitting on a branch",
        "a black and white photo of a city skyline at night",
        "a plate of food on a wooden table",
        "a rainbow colored background",
    ]

    for cap in tests:
        print_classification(cap)
