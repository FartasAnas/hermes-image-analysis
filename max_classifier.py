#!/usr/bin/env python3
"""
MAX Classifier — 34 dimensions, 37K+ keywords
=============================================
The most comprehensive BLIP caption classifier ever built.
34 dimensions covering every aspect of image content detectable from text.
Uses word-boundary matching, multi-word phrase matching, and derived rules.
"""
import sys, os

# Import the massive keyword database
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or '.')
try:
    from keyword_generator import ALL_DIMENSIONS
except ImportError:
    # Fallback: import from scripts directory
    sys.path.insert(0, r'E:/hermes_tools/scripts')
    from keyword_generator import ALL_DIMENSIONS

# ═══════════════════════════════════════════════════════════════
# MATCHING ENGINE
# ═══════════════════════════════════════════════════════════════

def classify_image(blip_caption):
    """
    Classifies a BLIP caption across all 34 dimensions.
    Returns dict: {dimension_name: [matched_labels]}
    """
    lower = blip_caption.lower()
    words = set(lower.split())
    
    results = {}
    
    for dim_name, dim_data in ALL_DIMENSIONS.items():
        results[dim_name] = []
        
        for label, keywords in dim_data.items():
            matched = False
            
            for kw in keywords:
                if not kw or len(kw) < 2:
                    continue
                    
                if ' ' in kw:
                    # Multi-word phrase — substring match in full caption
                    if kw in lower:
                        matched = True
                        break
                else:
                    # Single word — must be a whole word
                    if kw in words:
                        matched = True
                        break
            
            if matched:
                results[dim_name].append(label)
    
    # ── DERIVED RULES ──
    _apply_derived_rules(results, lower, words)
    
    # ── OVERRIDE: Fix camera/digital false positives ──
    source_final = results.get("source", [])
    # If classified as digital but caption describes physical objects → flip to photo
    physical_computer_signals = ["desktop computer", "laptop computer", "computer on", 
                                  "computer with", "a computer"]
    if "digital_abstract" in source_final:
        if any(s in lower for s in physical_computer_signals):
            results["source"] = ["photo"]
    
    # Deduplicate
    for dim in results:
        results[dim] = sorted(set(results[dim]))
    
    return results


def _apply_derived_rules(results, lower, words):
    """Apply cross-dimension inference rules."""
    
    # SOURCE inference
    source = results.get("source", [])
    if not source:
        has_camera = any(ci in lower for ci in [
            "man ", "woman ", "person ", "people ", "child ",
            "standing", "walking", "sitting", "looking at",
            "field of", "mountains", "river", "ocean", "beach",
            "forest", "sky with", "street", "car ", "tree",
            "flower", "bird", "dog ", "cat ", "building",
            "desktop computer", "laptop computer",  # photos of computers
        ])
        
        digital_signals = [
            "background with", "gradient", "pattern", "solid",
            "blank", "checkered", "striped", "polka", "grid",
            "diagram", "chart", "screenshot", "website",
            "menu with", "game over", "flash sale",
            "system error", "price for",
            "two circles", "one red and one blue",
            "circles with", "overlapping",
            # Only digital when in screen context
            "screenshote", "screen with", "screen showing",
            "app interface", "user interface", "ui ",
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
    
    # Building → architectural
    if "building_structure" in subject and "architectural" not in environment:
        environment.append("architectural")
        results["environment"] = environment
    
    # Plant/tree → natural
    if ("plant" in subject or "landscape_scenery" in subject) and \
       "natural_wilderness" not in environment and "urban" not in environment:
        environment.append("natural_wilderness")
        results["environment"] = environment
    
    # Food → indoor
    if "food_drink" in subject and not results.get("setting"):
        results["setting"] = ["indoor"]
    
    # Person + outdoor → could be outdoor
    if "person" in subject and not results.get("setting"):
        if any(w in lower for w in ["field", "mountain", "beach", "street", "park"]):
            results["setting"] = ["outdoor"]
        elif any(w in lower for w in ["room", "office", "kitchen", "studio"]):
            results["setting"] = ["indoor"]
    
    # Vehicle → likely outdoor
    if "vehicle_transport" in subject and not results.get("setting"):
        results["setting"] = ["outdoor"]
    
    # Night + dark → likely night setting
    if "dark_dim" in results.get("color", []) and not results.get("time_of_day"):
        if any(w in lower for w in ["night", "moon", "stars", "dark sky"]):
            results["time_of_day"] = ["night"]
    
    # Sunset colors → golden_hour
    if "warm_tones" in results.get("color", []) and \
       any(w in lower for w in ["sunset", "sunrise", "golden"]):
        results["time_of_day"] = results.get("time_of_day", []) + ["golden_hour"]
    
    # Has text + digital → text_heavy
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
# UTILITY
# ═══════════════════════════════════════════════════════════════

def get_keyword_count():
    """Return total keyword count."""
    return sum(
        sum(len(v) for v in dim.values())
        for dim in ALL_DIMENSIONS.values()
    )


def print_classification(caption, detailed=False):
    """Pretty-print classification for a caption."""
    labels = classify_image(caption)
    
    print(f"📷 '{caption}'")
    print(f"   Camera/Digital: {classify_camera_digital(caption)}")
    
    for dim, labs in labels.items():
        if labs:
            print(f"   {dim}: {', '.join(labs)}")
    
    if detailed:
        # Show which labels are not found
        missing_dims = [d for d in ALL_DIMENSIONS if not labels.get(d)]
        if missing_dims:
            print(f"   (no labels for: {', '.join(missing_dims)})")
    
    print()


if __name__ == '__main__':
    print(f"MAX Classifier loaded: {get_keyword_count():,} keywords, {len(ALL_DIMENSIONS)} dimensions\n")
    
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
