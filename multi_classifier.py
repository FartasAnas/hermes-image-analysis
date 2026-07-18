#!/usr/bin/env python3
"""
Multi-Category Image Classifier — v3 (BLIP-Optimized)
======================================================
Rebuilt from scratch with keywords matching BLIP's ACTUAL vocabulary.
Integrates the proven camera/digital classifier from analyze_image.py.

Categories derived from BLIP captions:
  1. source: photo, digital_abstract, painting, illustration, drawing, diagram, screenshot, map
  2. setting: indoor, outdoor, night, sunset, daytime
  3. environment: urban, rural, natural, architectural, coastal, mountainous
  4. subject: person, animal, bird, insect, fish, plant, landscape, building, vehicle, food, text_document
  5. composition: closeup, panorama, aerial_view
  6. color: monochrome, vibrant, dark, bright, warm_tones, cool_tones
  7. text: has_text, no_text
"""

# ═══════════════════════════════════════════════════════════════
# CATEGORIES — keywords tuned to BLIP-base actual output
# ═══════════════════════════════════════════════════════════════

CATEGORIES = {
    # ── SOURCE TYPE (what kind of image is this?) ──
    "source": {
        "painting": {
            "keywords": ["painting", "painted", "oil painting", "watercolor",
                        "canvas", "portrait painting", "fresco"],
        },
        "illustration": {
            "keywords": ["illustration", "illustrated", "cartoon", "comic",
                        "anime", "manga", "vector art", "clip art", "drawn in",
                        "child's drawing"],
        },
        "drawing": {
            "keywords": ["drawing", "sketch", "pencil drawing", "line drawing",
                        "doodle", "charcoal", "pen and ink"],
        },
        "diagram": {
            "keywords": ["diagram", "chart", "graph", "flowchart", "blueprint",
                        "schematic", "bar chart", "pie chart", "venn diagram",
                        "flow chart", "stages of", "showing the different"],
        },
        "screenshot": {
            "keywords": ["screenshot", "screen shot", "user interface", "website",
                        "web page", "computer screen", "mobile app", "app interface",
                        "calculator", "keyboard on", "text message", "chat",
                        "the website for", "screenshote"],
        },
        "map": {
            "keywords": ["map of", "world map", "road map", "atlas", "cartography"],
        },
        "digital_abstract": {
            # Catches: gradients, patterns, solid colors, geometric shapes
            # These are synthetic/generated, not photographs
            "keywords": [
                "gradient", "gradient background",
                "checkered", "checkered pattern",
                "striped pattern", "striped",
                "polka dot", "dots pattern",
                "grid pattern", "grid of", "graph paper",
                "blank sheet", "blank white",
                "solid color", "plain background",
                "color swatch", "color block",
                "geometric pattern", "geometric design",
                "abstract design", "abstract pattern",
                "rainbow colored", "rainbow gradient",
                "concentric circle", "concentric",
                "mosaic of", "low poly",
                "bar chart", "pie chart", "flow chart", "venn diagram",
                "background with a",  # Strong signal for digital
                # UI / screen elements
                "menu with", "a green menu", "a menu",
                "game over", "score", "start to",
                "flash sale", "sale up to",
                "price for the", "total",
                "two circles with", "two circle",
                "system error",
                "% off", "% 0",
                "warning high voltage",
                "weather app", "forecast",
            ],
        },
    },
    
    # ── SETTING (where/when?) ──
    "setting": {
        "indoor": {
            "keywords": ["indoor", "inside", "room", "interior", "hall", "office",
                        "living room", "bedroom", "kitchen", "bathroom", "museum",
                        "library", "classroom", "church", "cathedral"],
        },
        "outdoor": {
            "keywords": ["outdoor", "outside", "field", "landscape", "sky",
                        "garden", "park", "street", "road", "path", "trail",
                        "mountain", "beach", "ocean", "forest", "river", "lake"],
        },
        "night": {
            "keywords": ["night", "nighttime", "night sky", "moon", "stars",
                        "dark sky", "evening", "night scene", "at night"],
        },
        "sunset": {
            "keywords": ["sunset", "sunrise", "dusk", "dawn", "golden hour",
                        "twilight", "sun setting", "sun rising"],
        },
        "daytime": {
            "keywords": ["daytime", "sunny", "bright day", "blue sky",
                        "sunlight", "clear sky", "daylight"],
        },
    },
    
    # ── ENVIRONMENT ──
    "environment": {
        "urban": {
            "keywords": ["city", "urban", "downtown", "skyline", "skyscraper",
                        "street scene", "traffic", "sidewalk", "town",
                        "building in the", "buildings in the"],
        },
        "rural": {
            "keywords": ["rural", "countryside", "farm", "village", "barn",
                        "pasture", "meadow", "agricultural", "crop"],
        },
        "natural": {
            "keywords": ["nature", "natural", "wilderness", "forest", "jungle",
                        "mountain", "ocean", "lake", "river", "waterfall",
                        "canyon", "valley", "cliff", "cave", "hills"],
        },
        "architectural": {
            "keywords": ["architecture", "building", "tower", "bridge",
                        "castle", "palace", "monument", "skyscraper",
                        "cathedral", "church", "temple", "house",
                        "barn with", "red barn"],
        },
        "coastal": {
            "keywords": ["beach", "coast", "shore", "seaside", "ocean",
                        "sea", "wave", "sand", "harbor", "port", "pier",
                        "dock", "cliff by the"],
        },
        "mountainous": {
            "keywords": ["mountain", "mountains", "alpine", "peak", "summit",
                        "ridge", "snow capped", "glacier", "volcano", "hill"],
        },
    },
    
    # ── SUBJECT (what's in the image?) ──
    "subject": {
        "person": {
            "keywords": ["man ", "woman ", "person ", "boy ", "girl ",
                        "child ", "lady", "gentleman", "baby", "toddler",
                        "elderly", "people", "group of", "crowd", "family",
                        "couple", "friends", "team", "audience", "parade",
                        "a man", "a woman", "a person", "a boy", "a girl"],
        },
        "animal": {
            "keywords": ["dog", "cat", "horse", "cow", "sheep", "goat", "pig",
                        "deer", "bear", "wolf", "fox", "rabbit", "squirrel",
                        "elephant", "lion", "tiger", "giraffe", "zebra",
                        "monkey", "animal", "mammal", "wildlife", "creature",
                        "pet", "puppy", "kitten", "calf", "lamb"],
        },
        "bird": {
            "keywords": ["bird", "eagle", "hawk", "owl", "sparrow", "pigeon",
                        "duck", "swan", "flamingo", "parrot", "penguin",
                        "seagull", "winged", "feather", "nest"],
        },
        "insect": {
            "keywords": ["insect", "butterfly", "bee", "ant", "spider",
                        "beetle", "dragonfly", "caterpillar", "moth",
                        "ladybug", "bug"],
        },
        "fish": {
            "keywords": ["fish", "shark", "whale", "dolphin", "tuna",
                        "salmon", "goldfish", "aquatic", "marine life"],
        },
        "plant": {
            "keywords": ["plant", "flower", "tree", "leaf", "blossom",
                        "bloom", "petal", "rose", "lily", "daisy",
                        "sunflower", "tulip", "orchid", "garden",
                        "forest", "woods", "jungle", "bush", "foliage",
                        "vegetation", "cactus", "palm tree", "pine",
                        "grass", "fern", "moss", "vine", "green plant",
                        "yellow flower", "red flower", "pink flower",
                        "white flower", "purple flower"],
        },
        "landscape": {
            "keywords": ["landscape", "scenery", "view of", "panorama",
                        "vista", "horizon", "wide view", "countryside"],
        },
        "building": {
            "keywords": ["building", "house", "skyscraper", "tower",
                        "castle", "church", "cathedral", "temple",
                        "mosque", "palace", "bridge", "stadium",
                        "airport", "station", "factory", "barn",
                        "lighthouse", "windmill", "cabin", "cottage"],
        },
        "vehicle": {
            "keywords": ["car", "truck", "bus", "train", "plane",
                        "airplane", "boat", "ship", "bicycle",
                        "motorcycle", "helicopter", "vehicle",
                        "automobile", "scooter", "subway", "jet",
                        "sailboat", "yacht", "van ", "taxi"],
        },
        "food": {
            "keywords": ["food", "meal", "dish", "fruit", "vegetable",
                        "bread", "cake", "pizza", "pasta", "rice",
                        "meat", "dessert", "breakfast", "lunch",
                        "dinner", "plate of", "cuisine", "sandwich",
                        "burger", "soup", "salad", "cheese", "coffee",
                        "drink", "wine", "beer", "tea", "menu with",
                        "a plate of", "bowl of"],
        },
        "text_document": {
            "keywords": ["text", "document", "writing", "letter",
                        "page of", "book", "newspaper", "magazine",
                        "sign that", "label on", "words on",
                        "the word", "the words", "reads '",
                        "a sign", "sign with", "poster with",
                        "banner with", "menu with",
                        "with the word", "with the words",
                        "that reads", "with text", "written on",
                        "receipt", "invoice", "certificate"],
        },
    },
    
    # ── COMPOSITION ──
    "composition": {
        "closeup": {
            "keywords": ["close up", "close-up", "closeup", "macro",
                        "zoomed in", "detail of", "extreme close",
                        "magnified"],
        },
        "panorama": {
            "keywords": ["panorama", "panoramic", "wide angle",
                        "wide view", "sweeping view", "wide shot"],
        },
        "aerial_view": {
            "keywords": ["aerial", "from above", "overhead",
                        "bird's eye", "top view", "satellite view",
                        "drone", "bird's eye view"],
        },
    },
    
    # ── COLOR PALETTE ──
    "color": {
        "monochrome": {
            "keywords": ["black and white", "monochrome", "grayscale",
                        "bw photo", "sepia", "black & white"],
        },
        "vibrant": {
            "keywords": ["vibrant", "colorful", "bright colors",
                        "vivid", "multicolored", "rainbow",
                        "brightly colored", "many colors",
                        "different colors", "various colors",
                        "colorful design", "colorful pattern",
                        "rainbow colored", "colorful abstract"],
        },
        "dark": {
            "keywords": ["dark", "dim", "shadowy", "low light",
                        "night", "darkened", "dark background",
                        "black background", "dark blue"],
        },
        "bright": {
            "keywords": ["bright", "well lit", "sunny",
                        "brightly lit", "white background",
                        "light background", "bright white"],
        },
        "warm_tones": {
            "keywords": ["warm", "golden", "red and orange",
                        "yellow background", "orange background",
                        "red background", "sunset colors",
                        "warm light", "brown and yellow",
                        "red and yellow", "orange and red",
                        "orange gradient", "red gradient",
                        "yellow gradient", "brown gradient",
                        "yellow sign", "orange sign",
                        "yellow flower", "red flower",
                        "orange flower", "red barn",
                        "red and white", "orange and yellow",
                        "golden brown", "tan and brown"],
        },
        "cool_tones": {
            "keywords": ["cool", "blue and", "teal", "cyan",
                        "blue background", "green background",
                        "purple and", "blue sky", "cold",
                        "icy", "cool colors", "blue sky with",
                        "blue gradient", "green gradient",
                        "purple gradient", "blue screen",
                        "dark blue", "light blue", "turquoise",
                        "navy blue", "blue water", "green field",
                        "green and blue", "blue and green",
                        "purple and blue", "teal and"],
        },
    },
    
    # ── TEXT CONTENT ──
    "text": {
        "has_text": {
            "keywords": [
                "words", "text", "lettering", "writing",
                "written", "label", "sign", "caption",
                "headline", "title", "with the word",
                "with the words", "that reads",
                "menu with", "sign that", "reads '",
                "a sign", "poster with", "banner with",
                "the text", "word ", "font",
                "the words '", "that says",
                # UI/screen patterns that imply text
                "system error", "error screen",
                "sale up to", "flash sale",
                "price for", "store at",
                "game over", "score",
                "keyboard with", "keyboard button",
                "pie chart", "bar chart", "flow chart",
                "diagram showing", "chart with",
                "percentage of", "percent",
                "receipt", "invoice",
                "menu board", "a green menu",
                "calculator", "weather app",
                "chat interface", "message",
                # Number-heavy captions (BLIP read the text as numbers)
                "1 /", "2 /", "3 /", "4 /", "5 /",
            ],
        },
    },
}

# Camera indicators — if present, likely a real photo (natural scene)
CAMERA_INDICATORS = {
    "man ", "woman ", "person ", "people ", "child ", "dog ", "cat ",
    "standing", "walking", "sitting", "looking at", "field of",
    "mountains", "river in", "ocean", "beach", "forest",
    "building", "street", "car ", "tree", "flower", "bird",
    "sky with", "lake", "water", "cloud", "mountain",
    "a man", "a woman", "a group", "a person",
}

# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION ENGINE
# ═══════════════════════════════════════════════════════════════

def classify_image(blip_caption):
    """
    Takes a BLIP caption, returns multi-category labels.
    Uses word-boundary matching for single words.
    
    Returns: dict with category -> list of labels
    """
    lower = blip_caption.lower()
    words = set(lower.split())
    
    results = {}
    
    for category, labels in CATEGORIES.items():
        results[category] = []
        for label, config in labels.items():
            keywords = config.get("keywords", [])
            anti_keywords = config.get("anti_keywords", [])
            
            # Check anti-keywords first
            if any(ak in lower for ak in anti_keywords):
                continue
            
            # Check keywords
            matched = False
            for kw in keywords:
                if ' ' in kw:
                    # Multi-word phrase — substring match
                    if kw in lower:
                        matched = True
                        break
                else:
                    # Single word — word boundary check
                    if kw in words:
                        matched = True
                        break
            
            if matched:
                results[category].append(label)
    
    # ── Derived rules ──
    
    # If we have digital_abstract in source, add "digital" info
    source = results.get("source", [])
    
    # If no source detected, check if it looks like a photo
    if not source:
        has_camera = any(ci in lower for ci in CAMERA_INDICATORS)
        if has_camera:
            source = ["photo"]
        else:
            # Check for digital indicators
            digital_signals = ["background with", "gradient", "pattern",
                              "solid", "blank", "color swatch", "checkered",
                              "striped", "polka", "grid", "diagram",
                              "chart", "screenshot", "website"]
            if any(ds in lower for ds in digital_signals):
                source = ["digital_abstract"]
            else:
                source = ["photo"]  # Default
    
    results["source"] = source
    
    # If we have building in subject, add architectural to environment
    if "building" in results.get("subject", []) and "architectural" not in results.get("environment", []):
        results.setdefault("environment", []).append("architectural")
    
    # If food in subject, it's likely indoor
    if "food" in results.get("subject", []) and not results.get("setting"):
        results["setting"] = ["indoor"]
    
    # Deduplicate
    for cat in results:
        results[cat] = sorted(set(results[cat]))
    
    return results


def classify_camera_digital(blip_caption):
    """
    Determine camera vs digital using the rich classifier.
    Digital if source contains: painting, illustration, drawing, 
    diagram, screenshot, map, digital_abstract
    Photo otherwise.
    """
    labels = classify_image(blip_caption)
    source = labels.get("source", [])
    
    digital_sources = {"painting", "illustration", "drawing", "diagram", 
                       "screenshot", "map", "digital_abstract"}
    
    if digital_sources & set(source):
        return "digital"
    return "camera"


# ═══════════════════════════════════════════════════════════════
# GROUND TRUTH LABEL EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_ground_truth_labels(description):
    """Extract structured labels from a description using the same keyword system."""
    if not description:
        return {}
    return classify_image(description)
