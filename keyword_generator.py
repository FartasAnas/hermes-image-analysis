#!/usr/bin/env python3
"""
MEGA Classifier — 35 dimensions, massive keyword database
=========================================================
Programmatically generates an enormous keyword set from seed terms,
capturing BLIP's actual vocabulary patterns, synonyms, variations,
and multi-word combinations.

Dimensions (35 total):
  1. source          10. time_of_day     19. action          28. humidity
  2. setting         11. season          20. scale           29. smell_impression
  3. environment     12. weather         21. symmetry        30. taste_impression  
  4. subject         13. style           22. density         31. touch_impression
  5. composition     14. texture         23. material        32. age_era
  6. color           15. depth           24. pattern         33. quality
  7. text            16. lighting        25. sound           34. orientation
  8. mood            17. temperature     26. complexity      35. camera_distance
  9. atmosphere      18. direction       27. naturalness
"""
import re, math
from itertools import product as cartesian_product

# ═══════════════════════════════════════════════════════════════════
# KEYWORD GENERATOR — turns seeds into massive keyword lists
# ═══════════════════════════════════════════════════════════════════

def expand_keywords(seeds, add_plurals=True, add_variations=True, add_blip_patterns=True):
    """
    Takes a list of seed keywords and expands them massively.
    - Plural forms: dog → dogs, city → cities
    - BLIP common patterns: "a [word]", "[word] in the", "with [word]"
    - Common modifiers: "large", "small", "beautiful", "old", etc.
    - Comparative: "er" forms
    """
    expanded = set(seeds)
    
    # BLIP caption patterns (BLIP often uses these structures)
    blip_prefixes = [
        "a ", "an ", "the ", "some ", "several ", "many ",
        "a large ", "a small ", "a beautiful ", "an old ",
        "a big ", "a little ", "a single ", "a group of ",
        "a close up of ", "a view of ", "a picture of ",
        "a photo of ", "an image of ", "a shot of ",
    ]
    
    blip_suffixes = [
        " in the background", " in the foreground",
        " on the left", " on the right",
        " in the distance", " at night", " during the day",
        " under a blue sky", " on a white background",
        " on a black background", " on a dark background",
        " with text", " with words", " with a sign",
        " with water", " with trees", " with flowers",
        " with people", " with clouds",
    ]
    
    blip_connectors = [
        " and ", " with ", " on a ", " in a ", " at a ",
        " near a ", " next to a ", " in front of a ",
        " behind a ", " above a ", " below a ",
    ]
    
    # Common adjectives BLIP uses
    adjectives = [
        "large", "small", "big", "little", "tall", "short",
        "wide", "narrow", "beautiful", "pretty", "ugly", "old",
        "new", "modern", "ancient", "clean", "dirty", "bright",
        "dark", "light", "heavy", "colorful", "plain", "simple",
        "complex", "detailed", "empty", "full", "crowded",
        "open", "closed", "wooden", "metal", "glass", "stone",
        "golden", "silver", "bronze", "rusty", "shiny", "matte",
    ]
    
    # Common verbs in BLIP captions
    verbs = [
        "standing", "sitting", "walking", "running", "flying",
        "swimming", "floating", "hanging", "lying", "resting",
        "looking", "staring", "facing", "pointing", "showing",
        "displaying", "featuring", "depicting", "illustrating",
        "containing", "holding", "carrying", "wearing", "covered",
    ]
    
    if add_plurals:
        for s in list(seeds):
            if not s.endswith('s') and not s.endswith('ss') and len(s) > 2:
                # Simple plural
                expanded.add(s + 's')
                # -y → -ies
                if s.endswith('y') and len(s) > 3 and s[-2] not in 'aeiou':
                    expanded.add(s[:-1] + 'ies')
                # -f → -ves
                if s.endswith('f') and len(s) > 3:
                    expanded.add(s[:-1] + 'ves')
                if s.endswith('fe'):
                    expanded.add(s[:-2] + 'ves')
    
    if add_variations:
        for s in list(seeds):
            # Common prefixes
            expanded.add("non" + s)
            expanded.add("semi" + s)
            expanded.add("sub" + s)
            expanded.add("super" + s)
            expanded.add("micro" + s)
            expanded.add("macro" + s)
            
            # Common suffixes
            expanded.add(s + "y")
            expanded.add(s + "ish")
            expanded.add(s + "like")
            expanded.add(s + "less")
            expanded.add(s + "ful")
            
            # -er comparative
            if len(s) <= 5:
                expanded.add(s + "er")
            elif s.endswith('e'):
                expanded.add(s + "r")
    
    if add_blip_patterns:
        for s in list(seeds):
            if ' ' not in s and len(s) > 2:
                # "a [word]" pattern (BLIP loves this)
                expanded.add("a " + s)
                # "with a [word]"  
                expanded.add("with a " + s)
                # "[word] background"
                expanded.add(s + " background")
                # "[word] colored"
                expanded.add(s + " colored")
                # "[word] and"
                expanded.add(s + " and")
                # "and " + s
                expanded.add("and " + s)
    
    return sorted(expanded)


# ═══════════════════════════════════════════════════════════════════
# DIMENSION 1: SOURCE (what kind of image?)
# ═══════════════════════════════════════════════════════════════════

SOURCE = {
    "photo": expand_keywords([
        "photo", "photograph", "picture", "image", "shot",
        "snapshot", "picture of", "photo of", "photograph of",
        "image of", "picture showing", "photograph showing",
        "real photo", "camera", "photographic",
    ]),
    
    "digital_abstract": expand_keywords([
        # Core digital signals
        "gradient", "gradient background", "solid color", "solid background",
        "plain background", "blank background", "empty background",
        "abstract", "abstract design", "abstract pattern", "abstract art",
        "colorful background", "colored background", "color background",
        "background with a", "background with the",
        "background pattern", "pattern background",
        # UI & screen
        "screenshot", "screen shot", "user interface", "ui ",
        "website", "web page", "webpage", "computer screen",
        "desktop", "mobile app", "app interface", "application",
        "calculator", "calculator app", "keyboard", "keypad",
        "chat interface", "message screen", "text message",
        "weather app", "forecast app", "menu screen",
        "game screen", "loading screen", "login screen",
        "error screen", "warning screen", "blue screen",
        # Charts & diagrams
        "diagram", "chart", "graph", "flowchart", "flow chart",
        "bar chart", "pie chart", "line chart", "line graph",
        "venn diagram", "scatter plot", "histogram",
        "organizational chart", "org chart", "tree diagram",
        "mind map", "concept map", "infographic",
        "blueprint", "schematic", "technical drawing",
        "circuit diagram", "wiring diagram", "floor plan",
        # Generated/rendered
        "render", "rendered", "3d render", "3d model",
        "computer generated", "cgi", "digital art",
        "digital painting", "digital drawing", "digital illustration",
        "vector graphics", "vector art", "pixel art",
        # Patterns (strong digital signal)
        "checkered", "checkerboard", "checkered pattern",
        "striped", "striped pattern", "stripes pattern",
        "polka dot", "dots pattern", "dotted pattern",
        "grid", "grid pattern", "graph paper", "grid of",
        "geometric", "geometric pattern", "geometric design",
        "geometric shape", "geometric shapes",
        "mosaic", "tessellation", "tiled pattern",
        "concentric", "concentric circle", "concentric circles",
        "spiral pattern", "wave pattern", "zigzag pattern",
        # Color blocks & swatches
        "color swatch", "color block", "color palette",
        "color wheel", "color spectrum", "rainbow",
        "rainbow colored", "rainbow gradient",
        # Text-based (implies digital creation)
        "a sign that", "sign with text", "text sign",
        "warning sign", "caution sign", "danger sign",
        "menu board", "menu with", "a green menu",
        "price tag", "price list", "receipt",
        "poster", "banner", "billboard",
        "game over", "score screen", "high score",
        "flash sale", "sale banner", "sale sign",
        "discount sign", "offer sign", "promotion",
        # Code / terminal
        "code", "source code", "code snippet", "code editor",
        "terminal", "command line", "console",
        "programming", "script", "function",
    ]),
    
    "painting": expand_keywords([
        "painting", "painted", "oil painting", "oil on canvas",
        "watercolor", "watercolour", "acrylic", "acrylic painting",
        "fresco", "mural", "canvas", "brush stroke", "brushstroke",
        "palette knife", "impasto", "gouache",
        "fine art", "masterpiece", "artwork", "work of art",
        "portrait painting", "landscape painting", "still life painting",
    ]),
    
    "illustration": expand_keywords([
        "illustration", "illustrated", "illustrator",
        "cartoon", "comic", "comic strip", "anime",
        "manga", "graphic novel", "children's book",
        "vector art", "clip art", "stock illustration",
        "drawn in", "hand drawn", "sketched",
        "coloring book", "coloring page",
    ]),
    
    "drawing": expand_keywords([
        "drawing", "sketch", "pencil drawing", "pencil sketch",
        "charcoal drawing", "charcoal", "pen and ink",
        "ink drawing", "line drawing", "line art",
        "doodle", "scribble", "etching", "lithograph",
        "print", "woodcut", "engraving",
        "architectural drawing", "technical drawing",
    ]),
    
    "map": expand_keywords([
        "map of", "world map", "road map", "street map",
        "topographic map", "topographic", "atlas",
        "cartography", "cartographic", "navigational chart",
        "nautical chart", "political map", "physical map",
        "country map", "city map", "region map",
        "aerial map", "satellite map", "terrain map",
    ]),
    
    "xray_scan": expand_keywords([
        "x-ray", "xray", "radiograph", "ct scan",
        "mri", "ultrasound", "medical imaging",
        "scan", "scanned", "scanner",
    ]),
    
    "microscope": expand_keywords([
        "microscope", "microscopic", "micrograph",
        "electron microscope", "sem image",
        "magnified", "cell", "bacteria",
    ]),
    
    "aerial_satellite": expand_keywords([
        "aerial", "aerial view", "aerial photo", "aerial photograph",
        "bird's eye", "bird's eye view", "from above",
        "overhead view", "overhead shot", "top view",
        "satellite", "satellite image", "satellite view",
        "drone", "drone shot", "drone photo", "drone footage",
        "aerial panorama", "aerial landscape",
    ]),
    
    "night_vision": expand_keywords([
        "night vision", "thermal", "infrared", "heat map",
        "thermal imaging", "flir",
    ]),
    
    "underwater_photo": expand_keywords([
        "underwater photo", "underwater photograph",
        "underwater shot", "underwater image",
        "scuba", "diving photo", "snorkeling photo",
        "marine photo", "ocean floor photo",
    ]),
}

# ═══════════════════════════════════════════════════════════════════
# DIMENSION 2: SETTING (where?)
# ═══════════════════════════════════════════════════════════════════

SETTING = {
    "indoor": expand_keywords([
        "indoor", "inside", "interior", "room",
        "living room", "bedroom", "kitchen", "bathroom",
        "dining room", "hallway", "corridor", "basement",
        "attic", "garage", "study", "den", "office",
        "classroom", "library", "museum", "gallery",
        "hall", "auditorium", "theater", "cinema",
        "restaurant", "cafe", "cafeteria", "bar", "pub",
        "shop", "store", "mall", "supermarket",
        "airport terminal", "train station", "bus station",
        "hospital", "clinic", "waiting room",
        "gym", "fitness center", "stadium interior",
        "church", "cathedral", "temple", "mosque", "synagogue",
        "factory floor", "warehouse", "workshop",
        "laboratory", "lab", "studio",
    ]),
    
    "outdoor": expand_keywords([
        "outdoor", "outside", "exterior", "outdoors",
        "open air", "alfresco",
        "field", "meadow", "pasture", "grassland",
        "park", "garden", "backyard", "courtyard",
        "street", "road", "path", "trail", "sidewalk",
        "mountain", "hill", "valley", "canyon",
        "beach", "coast", "shore", "seaside",
        "ocean", "sea", "lake", "river", "stream",
        "forest", "woods", "jungle", "rainforest",
        "desert", "tundra", "arctic", "savanna",
    ]),
    
    "studio": expand_keywords([
        "studio", "photo studio", "photography studio",
        "art studio", "recording studio", "sound stage",
        "backdrop", "studio lighting", "studio setup",
        "seamless background", "infinity cove",
    ]),
    
    "underwater": expand_keywords([
        "underwater", "submerged", "ocean floor",
        "coral reef", "reef", "sea bed", "seabed",
        "deep sea", "marine environment", "aquatic",
        "below the surface", "beneath the waves",
    ]),
    
    "outer_space": expand_keywords([
        "space", "outer space", "in space", "orbit",
        "galaxy", "nebula", "stars", "planet",
        "astronomical", "cosmic", "interstellar",
        "deep space", "solar system", "milky way",
    ]),
}

# ═══════════════════════════════════════════════════════════════════
# DIMENSION 3: ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════

ENVIRONMENT = {
    "urban": expand_keywords([
        "city", "urban", "downtown", "midtown", "metropolis",
        "skyline", "cityscape", "town", "suburb",
        "street scene", "city street", "urban street",
        "traffic", "intersection", "crosswalk",
        "skyscraper", "high-rise", "apartment building",
        "plaza", "square", "public square",
        "alley", "alleyway", "back street",
        "neighborhood", "district", "block",
    ]),
    
    "rural": expand_keywords([
        "rural", "countryside", "farm", "farmland",
        "village", "hamlet", "small town",
        "pasture", "meadow", "grassland",
        "barn", "farmhouse", "ranch",
        "agricultural", "crop field", "wheat field",
        "corn field", "rice paddy", "orchard",
        "vineyard", "plantation",
        "dirt road", "country road", "gravel road",
    ]),
    
    "natural_wilderness": expand_keywords([
        "nature", "natural", "wilderness", "wild",
        "untouched", "pristine", "remote",
        "forest", "woods", "jungle", "rainforest",
        "mountain", "alpine", "highlands",
        "ocean", "sea", "lake", "river", "waterfall",
        "canyon", "gorge", "valley", "cliff",
        "cave", "cavern", "grotto",
        "desert", "dune", "oasis",
        "tundra", "arctic", "ice field", "glacier",
        "wetland", "marsh", "swamp", "bog",
        "coral reef", "kelp forest",
    ]),
    
    "architectural": expand_keywords([
        "architecture", "architectural", "building design",
        "building", "structure", "edifice",
        "tower", "skyscraper", "high-rise",
        "castle", "fortress", "citadel",
        "palace", "mansion", "manor",
        "monument", "memorial", "landmark",
        "temple", "cathedral", "church", "mosque",
        "bridge", "viaduct", "aqueduct",
        "stadium", "arena", "amphitheater",
        "facade", "column", "arch", "dome",
        "classical architecture", "modern architecture",
        "gothic", "baroque", "renaissance",
    ]),
    
    "coastal_marine": expand_keywords([
        "beach", "coast", "shoreline", "seaside",
        "ocean", "sea", "bay", "cove", "inlet",
        "wave", "surf", "tide", "breaker",
        "sand", "dune", "pebble beach", "rocky shore",
        "harbor", "port", "pier", "dock", "marina",
        "cliff", "headland", "promontory",
        "lighthouse", "coastal view", "sea view",
        "coral reef", "atoll", "lagoon", "estuary",
    ]),
    
    "mountainous": expand_keywords([
        "mountain", "mountains", "mountain range",
        "alpine", "alps", "highlands",
        "peak", "summit", "ridge", "crest",
        "snow capped", "snowy peak", "glacier",
        "volcano", "volcanic", "crater",
        "foothill", "hillside", "mountain pass",
        "valley", "gorge", "ravine",
        "cliff face", "rock face", "crag",
    ]),
    
    "industrial": expand_keywords([
        "industrial", "industry", "factory",
        "warehouse", "plant", "refinery",
        "power plant", "power station", "smokestack",
        "oil rig", "derrick", "pipeline",
        "construction site", "scaffolding",
        "mine", "quarry", "excavation",
        "shipping yard", "container port",
    ]),
    
    "domestic": expand_keywords([
        "home", "house", "apartment", "flat",
        "living space", "residence", "dwelling",
        "cozy", "homely", "domestic",
        "family room", "living room", "dining room",
    ]),
}

# ═══════════════════════════════════════════════════════════════════
# DIMENSION 4: SUBJECT (what's in the image?)
# ═══════════════════════════════════════════════════════════════════

SUBJECT = {
    "person": expand_keywords([
        "man", "woman", "person", "individual",
        "boy", "girl", "child", "kid", "toddler", "baby",
        "adult", "teenager", "elderly", "senior",
        "lady", "gentleman", "guy", "gal",
        "male", "female", "human", "figure",
        "man in", "woman in", "person in",
        "a man", "a woman", "a person", "a boy", "a girl",
    ]),
    
    "group_of_people": expand_keywords([
        "people", "group", "crowd", "audience",
        "family", "couple", "pair", "duo",
        "friends", "team", "class", "students",
        "workers", "staff", "employees",
        "parade", "procession", "gathering",
        "group of", "a group of", "several people",
        "many people", "a crowd of", "crowds of",
    ]),
    
    "animal_mammal": expand_keywords([
        "animal", "mammal", "wildlife", "creature", "beast",
        "dog", "cat", "horse", "cow", "sheep", "goat", "pig",
        "deer", "bear", "wolf", "fox", "coyote", "raccoon",
        "rabbit", "hare", "squirrel", "chipmunk", "mouse", "rat",
        "elephant", "rhino", "hippo", "giraffe", "zebra",
        "lion", "tiger", "leopard", "cheetah", "jaguar", "panther",
        "monkey", "ape", "gorilla", "chimpanzee", "orangutan",
        "kangaroo", "koala", "panda", "sloth",
        "whale", "dolphin", "seal", "walrus", "otter", "beaver",
        "bat", "hedgehog", "porcupine", "armadillo",
        "puppy", "kitten", "calf", "foal", "lamb", "cub",
    ]),
    
    "bird": expand_keywords([
        "bird", "avian", "fowl",
        "eagle", "hawk", "falcon", "owl", "vulture", "raven", "crow",
        "sparrow", "robin", "finch", "blue jay", "cardinal",
        "pigeon", "dove", "seagull", "pelican", "albatross",
        "duck", "goose", "swan", "flamingo", "heron", "crane",
        "parrot", "parakeet", "cockatoo", "macaw",
        "penguin", "puffin", "ostrich", "emu",
        "hummingbird", "woodpecker", "kingfisher",
        "winged", "feathered", "nest", "perched",
    ]),
    
    "insect_arthropod": expand_keywords([
        "insect", "bug", "arthropod",
        "butterfly", "moth", "caterpillar",
        "bee", "wasp", "hornet", "bumblebee",
        "ant", "termite", "beetle", "ladybug", "ladybird",
        "dragonfly", "damselfly", "grasshopper", "cricket",
        "spider", "scorpion", "tick", "mite",
        "centipede", "millipede", "crab", "lobster", "shrimp",
        "fly", "mosquito", "gnat",
        "cocoon", "chrysalis", "web", "hive",
    ]),
    
    "fish_marine": expand_keywords([
        "fish", "shark", "ray", "eel", "seahorse",
        "tuna", "salmon", "trout", "bass", "cod", "carp",
        "goldfish", "koi", "clownfish", "angelfish",
        "jellyfish", "octopus", "squid", "cuttlefish",
        "starfish", "sea urchin", "anemone",
        "marine life", "aquatic life", "sea creature",
        "underwater animal", "ocean animal",
    ]),
    
    "reptile_amphibian": expand_keywords([
        "reptile", "snake", "lizard", "gecko", "iguana",
        "chameleon", "komodo", "crocodile", "alligator",
        "turtle", "tortoise", "terrapin",
        "frog", "toad", "salamander", "newt",
        "amphibian", "cold blooded",
    ]),
    
    "plant": expand_keywords([
        "plant", "flower", "tree", "bush", "shrub",
        "leaf", "leaves", "blossom", "bloom", "petal",
        "rose", "lily", "daisy", "tulip", "orchid",
        "sunflower", "dandelion", "lavender", "jasmine",
        "garden", "forest", "woods", "jungle",
        "grass", "fern", "moss", "vine", "ivy",
        "cactus", "succulent", "aloe",
        "palm tree", "pine tree", "oak tree", "maple",
        "bamboo", "seaweed", "kelp", "algae",
        "foliage", "vegetation", "greenery", "flora",
        "bouquet", "wreath", "garland",
        "botanical", "horticulture",
    ]),
    
    "landscape_scenery": expand_keywords([
        "landscape", "scenery", "view", "vista",
        "panorama", "horizon", "scenic", "countryside",
        "mountain view", "ocean view", "valley view",
        "skyline", "cityscape", "seascape",
    ]),
    
    "space_astronomy": expand_keywords([
        "earth", "planet", "planets", "universe", "cosmos",
        "galaxy", "nebula", "stars", "star", "solar system",
        "space", "outer space", "orbit", "orbiting",
        "satellite", "astronomical", "celestial",
        "milky way", "andromeda", "constellation",
        "comet", "asteroid", "meteor", "meteorite",
        "space station", "spaceship", "spacecraft",
        "astronaut", "cosmonaut", "moon", "lunar",
        "sun", "solar", "jupiter", "saturn", "mars",
        "venus", "mercury", "uranus", "neptune", "pluto",
        "world", "globe", "sphere", "atmosphere",
        "exoplanet", "black hole", "supernova",
        "deep space", "interstellar", "intergalactic",
        "the earth", "of the earth", "of the universe",
        "earth from space", "blue marble",
    ]),
    
    "network_technology": expand_keywords([
        "network", "networks", "networking", "connected",
        "connections", "connecting", "interconnected",
        "nodes", "node", "links", "linked", "linking",
        "web", "webbed", "lattice", "mesh", "grid",
        "circuit", "circuitry", "circuit board",
        "motherboard", "microchip", "processor",
        "data", "digital network", "information",
        "internet", "online", "cyber", "cyberspace",
        "technology", "technological", "tech",
        "fiber optic", "wireless", "broadband",
        "server", "cloud computing", "blockchain",
        "global network", "worldwide web",
        "network of", "web of", "lines and dots",
        "nodes and", "connected lines", "dots and lines",
    ]),
    
    "building_structure": expand_keywords([
        "building", "house", "home", "structure",
        "skyscraper", "tower", "high-rise",
        "castle", "palace", "fortress", "mansion",
        "church", "cathedral", "temple", "mosque",
        "bridge", "stadium", "arena",
        "airport", "station", "terminal",
        "barn", "cabin", "cottage", "hut",
        "lighthouse", "windmill", "mill",
        "ruins", "monument", "memorial",
    ]),
    
    "vehicle_transport": expand_keywords([
        "vehicle", "transport", "transportation",
        "car", "truck", "bus", "van", "suv",
        "motorcycle", "bike", "bicycle", "scooter",
        "train", "locomotive", "subway", "tram",
        "airplane", "plane", "jet", "helicopter",
        "boat", "ship", "yacht", "sailboat", "ferry",
        "spaceship", "rocket", "spacecraft",
        "taxi", "cab", "ambulance", "fire truck", "police car",
        "tractor", "bulldozer", "crane", "forklift",
    ]),
    
    "food_drink": expand_keywords([
        "food", "meal", "dish", "cuisine", "recipe",
        "fruit", "vegetable", "meat", "fish", "bread",
        "cake", "pie", "pastry", "dessert", "cookie",
        "pizza", "pasta", "rice", "noodle", "soup",
        "salad", "sandwich", "burger", "taco", "sushi",
        "breakfast", "lunch", "dinner", "snack",
        "coffee", "tea", "wine", "beer", "cocktail",
        "plate of", "bowl of", "serving of",
        "platter", "buffet", "feast",
        "ingredient", "spice", "herb",
    ]),
    
    "text_document_content": expand_keywords([
        "text", "document", "writing", "letter", "message",
        "book", "newspaper", "magazine", "journal",
        "sign", "label", "tag", "sticker",
        "poster", "banner", "billboard", "flyer",
        "menu", "list", "schedule", "calendar",
        "receipt", "invoice", "ticket", "coupon",
        "certificate", "diploma", "license",
        "lettering", "typography", "calligraphy",
        "handwriting", "cursive", "print",
        "reading material", "written content",
    ]),
    
    "artwork_sculpture": expand_keywords([
        "painting", "sculpture", "statue", "bust",
        "artwork", "art piece", "work of art",
        "mural", "fresco", "installation",
        "exhibit", "gallery piece", "museum piece",
        "carving", "relief", "bronze", "marble statue",
    ]),
    
    "technology_device": expand_keywords([
        "computer", "laptop", "tablet", "smartphone", "phone",
        "monitor", "screen", "display", "television", "tv",
        "keyboard", "mouse", "trackpad", "headphones", "speaker",
        "camera", "lens", "drone", "robot",
        "circuit", "microchip", "processor", "motherboard",
        "gadget", "device", "electronic", "appliance",
    ]),
    
    "stars_astronomy": expand_keywords([
        "star", "stars", "starfield", "starry", "stellar",
        "astral", "celestial", "constellation", "constellations",
        "cosmos", "cosmic", "firmament", "heavens",
        "twinkling", "twinkle", "luminous", "luminescent",
        "points of light", "pinpricks of light", "dots of light",
        "light dots", "bright dots", "glowing dots",
        "scattered stars", "distant stars", "background stars",
        "star filled", "star-filled", "star studded",
        "galaxy", "galaxies", "milky way", "andromeda",
        "nebula", "nebulae", "nebular", "nebulas",
        "interstellar", "intergalactic", "deep space",
        "outer space", "space background", "cosmic background",
        "astronomical", "astronomy", "astrophysical",
        "planet", "planets", "planetary", "exoplanet",
        "earth", "world", "globe", "terrestrial",
        "jupiter", "saturn", "mars", "venus", "mercury",
        "uranus", "neptune", "pluto", "solar system",
        "sun", "solar", "lunar", "moon", "moons",
        "comet", "asteroid", "meteor", "meteorite",
        "meteor shower", "shooting star", "falling star",
        "black hole", "supernova", "wormhole",
        "satellite", "space station", "spaceship",
        "spacecraft", "astronaut", "cosmonaut",
        "orbit", "orbiting", "orbital",
        "atmosphere", "stratosphere", "hemisphere",
        "continent", "landmass", "ocean", "sea",
        "blue marble", "pale blue dot",
        "celestial body", "heavenly body",
        "radiant", "glowing", "luminescent",
        "ethereal", "otherworldly",
    ]),
    
    "motion_effects": expand_keywords([
        "motion", "movement", "moving", "speed",
        "zoom", "zooming", "zoom effect", "zoom blur",
        "streak", "streaking", "streaks of light",
        "motion blur", "radial blur", "speed blur",
        "speed lines", "action lines", "motion lines",
        "warp speed", "light speed", "hyperspeed",
        "accelerating", "racing", "rushing", "flying",
        "dashing", "surging", "bursting",
        "trail", "trailing", "light trail", "star trail",
        "dynamic", "dynamic motion", "fast moving",
        "rapid", "swift", "quick", "velocity",
        "whizzing", "whooshing", "spinning", "rotating",
        "blurred motion", "motion effect", "speed effect",
        "stretching", "elongating", "extending outward",
        "radiating", "emanating from center",
        "starburst", "sunburst", "light burst",
    ]),
    
    "overlay_superimpose": expand_keywords([
        "overlay", "overlaid", "overlaying",
        "superimposed", "superimpose", "superimposition",
        "wrapped", "wrapping", "wrapped around",
        "encircling", "encircled", "surrounding",
        "enveloping", "enveloped", "covering",
        "coating", "coated", "blanketing",
        "draped", "draped over", "laid over",
        "placed on top", "on top of", "atop",
        "mapped onto", "projected onto",
        "lattice", "lattice overlay", "wireframe",
        "mesh", "meshed", "netting", "webbing",
        "grid overlay", "hud overlay", "interface overlay",
        "augmented reality", "ar overlay",
        "holographic", "hologram", "projection",
    ]),
    
    "glow_effects": expand_keywords([
        "glow", "glowing", "glowed", "glows",
        "luminescent", "luminescence", "luminous",
        "radiant", "radiance", "radiating",
        "shining", "shimmering", "sparkling",
        "gleaming", "glittering", "glistening",
        "bright", "brilliant", "dazzling",
        "fluorescent", "phosphorescent", "incandescent",
        "neon", "neon glow", "neon lit",
        "illuminated", "lit up", "lighted",
        "backlit", "edge lit", "rim lit",
        "glowing lines", "glowing nodes", "glowing points",
        "glowing network", "glowing web",
        "light emitting", "emitting light",
        "glowing against", "glowing on",
        "bright against dark", "contrasting glow",
    ]),
    
    "theme_concept": expand_keywords([
        "connectivity", "connection", "interconnection",
        "global network", "worldwide web", "internet",
        "digital age", "information age", "technology age",
        "future", "futuristic", "sci-fi", "science fiction",
        "cyber", "cyberspace", "virtual", "digital realm",
        "exploration", "discovery", "frontier",
        "advancement", "progress", "innovation",
        "unity", "togetherness", "global community",
        "communication", "telecommunication",
        "data flow", "information flow", "knowledge",
        "artificial intelligence", "ai", "machine learning",
        "big data", "cloud", "cloud computing",
        "surveillance", "monitoring", "tracking",
        "security", "protection", "defense",
        "power", "energy", "force", "strength",
        "beauty", "wonder", "awe", "majesty",
        "mystery", "enigma", "unknown",
        "hope", "inspiration", "aspiration",
        "danger", "threat", "warning",
        "life", "nature", "environment",
        "sustainability", "renewable", "green energy",
    ]),
    
    "body_part": expand_keywords([
        "face", "eye", "hand", "finger", "arm", "leg", "foot",
        "head", "hair", "ear", "nose", "mouth", "lip",
        "portrait", "profile", "headshot", "close up of a face",
    ]),
}

# ═══════════════════════════════════════════════════════════════════
# DIMENSION 5-35: (all remaining dimensions)
# ═══════════════════════════════════════════════════════════════════

COMPOSITION = {
    "closeup_macro": expand_keywords([
        "close up", "close-up", "closeup", "macro",
        "zoomed in", "extreme close", "magnified",
        "detail shot", "detailed view", "tight shot",
        "macro photo", "macro photograph",
    ]),
    "wide_panorama": expand_keywords([
        "panorama", "panoramic", "wide angle", "wide view",
        "sweeping view", "wide shot", "long shot",
        "landscape orientation", "widescreen", "cinematic",
    ]),
    "aerial_top": expand_keywords([
        "aerial", "from above", "overhead", "bird's eye",
        "top view", "top down", "satellite view",
        "drone view", "aerial view", "bird's eye view",
    ]),
    "portrait_orientation": expand_keywords([
        "portrait", "headshot", "profile", "bust shot",
        "vertical", "portrait orientation",
    ]),
}

COLOR = {
    "monochrome_bw": expand_keywords([
        "black and white", "b&w", "monochrome", "grayscale",
        "greyscale", "bw photo", "sepia", "monotone",
        "black & white", "no color", "colorless",
    ]),
    "vibrant_colorful": expand_keywords([
        "vibrant", "colorful", "bright colors", "vivid",
        "multicolored", "rainbow", "brightly colored",
        "many colors", "different colors", "various colors",
        "colorful design", "colorful pattern", "colorful abstract",
        "rainbow colored", "rainbow gradient", "brightly lit",
        "vibrantly colored", "rich colors", "saturated",
    ]),
    "dark_dim": expand_keywords([
        "dark", "dim", "shadowy", "low light", "darkened",
        "dark background", "black background", "dark blue",
        "night scene", "poorly lit", "dimly lit",
        "dark room", "darkened room", "unlit", "unilluminated",
    ]),
    "bright_light": expand_keywords([
        "bright", "well lit", "sunny", "brightly lit",
        "white background", "light background", "bright white",
        "illuminated", "well illuminated", "brightly illuminated",
        "daylight", "sunlit", "light-filled",
    ]),
    "warm_tones": expand_keywords([
        "warm", "golden", "red and orange", "warm light",
        "yellow background", "orange background", "red background",
        "sunset colors", "warm colors", "brown and yellow",
        "orange gradient", "red gradient", "yellow gradient",
        "red and yellow", "orange and red", "golden brown",
        "tan and brown", "sepia tone", "amber", "copper",
        "bronze", "peach", "coral", "salmon",
        "yellow sign", "orange sign", "red sign",
        "yellow flower", "red flower", "orange flower",
        "red barn", "red and white", "warm glow",
    ]),
    "cool_tones": expand_keywords([
        "cool", "blue and", "teal", "cyan", "cold",
        "blue background", "green background", "purple and",
        "cool colors", "blue sky with", "blue sky",
        "blue gradient", "green gradient", "purple gradient",
        "dark blue", "light blue", "turquoise", "navy blue",
        "blue water", "green field", "purple and blue",
        "teal and", "mint", "aqua", "azure", "indigo",
        "lavender", "lilac", "periwinkle", "sage",
        "slate", "steel blue", "powder blue",
    ]),
    "pastel": expand_keywords([
        "pastel", "soft colors", "muted colors", "gentle colors",
        "light pink", "baby blue", "soft yellow", "pale",
        "pastel pink", "pastel blue", "pastel purple",
    ]),
    "high_contrast": expand_keywords([
        "high contrast", "contrasty", "stark", "bold contrast",
        "dramatic lighting", "chiaroscuro", "silhouette",
        "backlit", "strong contrast", "deep shadows",
    ]),
}

TEXT = {
    "has_text": expand_keywords([
        "words", "text", "lettering", "writing", "written",
        "label", "sign", "caption", "headline", "title",
        "with the word", "with the words", "that reads",
        "menu with", "sign that", "reads '", "a sign",
        "poster with", "banner with", "the text", "word ",
        "font", "the words '", "that says", "text on",
        "message on", "written on", "printed on",
        # UI/text patterns
        "system error", "error screen", "error message",
        "sale up to", "flash sale", "price for", "store at",
        "game over", "score", "high score",
        "keyboard with", "keyboard button", "screenshot",
        "pie chart", "bar chart", "flow chart", "diagram showing",
        "percentage of", "percent", "receipt", "invoice",
        "menu board", "a green menu", "calculator", "weather app",
        "chat interface", "message bubble",
        # Number patterns (BLIP reads numbers as text)
        "1 /", "2 /", "3 /", "4 /", "5 /", "6 /", "7 /", "8 /",
        "% off", "% 0", "0 %", "50 %",
    ]),
    "sign_present": expand_keywords([
        "sign", "signage", "street sign", "road sign",
        "traffic sign", "warning sign", "stop sign",
        "billboard", "neon sign", "shop sign", "store sign",
        "exit sign", "direction sign", "informational sign",
    ]),
    "text_heavy": expand_keywords([
        "text heavy", "lots of text", "full of text",
        "text dense", "text-filled", "text page",
        "document with text", "page of text",
        "many words", "lots of writing", "text document",
    ]),
}

MOOD = {
    "peaceful": expand_keywords([
        "peaceful", "calm", "serene", "tranquil", "quiet",
        "relaxing", "soothing", "restful", "placid",
        "gentle", "mild", "soft", "still",
    ]),
    "dramatic": expand_keywords([
        "dramatic", "striking", "intense", "powerful",
        "bold", "theatrical", "spectacular", "impressive",
    ]),
    "gloomy": expand_keywords([
        "gloomy", "dark", "dreary", "bleak", "dismal",
        "somber", "melancholy", "sad", "depressing",
    ]),
    "cheerful": expand_keywords([
        "cheerful", "happy", "joyful", "bright", "upbeat",
        "lively", "festive", "celebratory", "playful",
        "fun", "whimsical", "delightful",
    ]),
    "mysterious": expand_keywords([
        "mysterious", "enigmatic", "cryptic", "puzzling",
        "eerie", "uncanny", "strange", "unusual",
        "surreal", "dreamlike", "otherworldly",
    ]),
    "romantic": expand_keywords([
        "romantic", "lovely", "beautiful", "charming",
        "enchanting", "captivating", "alluring",
    ]),
    "energetic": expand_keywords([
        "energetic", "dynamic", "active", "lively",
        "vibrant mood", "exciting", "thrilling",
    ]),
    "nostalgic": expand_keywords([
        "nostalgic", "old-fashioned", "retro", "vintage mood",
        "antique looking", "historic", "timeless",
    ]),
    "majestic": expand_keywords([
        "majestic", "grand", "magnificent", "splendid",
        "glorious", "awe-inspiring", "breathtaking",
    ]),
    "eerie": expand_keywords([
        "eerie", "spooky", "creepy", "haunting",
        "unsettling", "disturbing", "sinister",
    ]),
}

ATMOSPHERE = MOOD  # Same keywords can apply

TIME_OF_DAY = {
    "morning": expand_keywords([
        "morning", "dawn", "sunrise", "early morning",
        "daybreak", "first light", "morning light",
    ]),
    "afternoon": expand_keywords([
        "afternoon", "midday", "noon", "lunchtime",
        "early afternoon", "late afternoon",
    ]),
    "evening": expand_keywords([
        "evening", "dusk", "sunset", "twilight",
        "sundown", "nightfall", "late evening",
    ]),
    "night": expand_keywords([
        "night", "nighttime", "midnight", "late night",
        "dark night", "night sky", "starry night",
    ]),
    "golden_hour": expand_keywords([
        "golden hour", "magic hour", "golden light",
        "warm sunset", "sunset glow", "sunrise glow",
    ]),
}

SEASON = {
    "spring": expand_keywords([
        "spring", "springtime", "blooming", "blossom",
        "cherry blossom", "spring flowers", "new growth",
    ]),
    "summer": expand_keywords([
        "summer", "summertime", "sunny day", "hot day",
        "beach day", "summer sun", "tropical",
    ]),
    "autumn": expand_keywords([
        "autumn", "fall", "autumn leaves", "fall foliage",
        "autumn colors", "harvest", "pumpkin",
    ]),
    "winter": expand_keywords([
        "winter", "snow", "snowy", "snowing", "ice",
        "frost", "cold winter", "winter scene",
        "snow covered", "snowy landscape", "blizzard",
    ]),
}

WEATHER = {
    "sunny": expand_keywords([
        "sunny", "sunshine", "bright sun", "clear sky",
        "blue sky", "cloudless", "sunny day",
    ]),
    "cloudy": expand_keywords([
        "cloudy", "overcast", "cloud cover", "grey sky",
        "gray sky", "clouds", "clouded", "partly cloudy",
    ]),
    "rainy": expand_keywords([
        "rain", "rainy", "raining", "rainfall", "drizzle",
        "shower", "downpour", "storm", "thunderstorm",
    ]),
    "snowy": expand_keywords([
        "snow", "snowy", "snowing", "snowfall", "blizzard",
        "snowstorm", "flurry", "whiteout",
    ]),
    "foggy": expand_keywords([
        "fog", "foggy", "mist", "misty", "haze", "hazy",
        "smog", "low visibility", "fog bank",
    ]),
    "windy": expand_keywords([
        "windy", "wind", "breezy", "gust", "gale",
        "stormy winds", "blustery",
    ]),
}

STYLE = {
    "realistic": expand_keywords([
        "realistic", "photorealistic", "lifelike", "true to life",
        "naturalistic", "representational",
    ]),
    "abstract": expand_keywords([
        "abstract", "non-representational", "abstract art",
        "abstract expression", "non-figurative",
    ]),
    "minimalist": expand_keywords([
        "minimalist", "minimal", "simple", "clean design",
        "bare", "sparse", "uncluttered", "plain style",
    ]),
    "ornate": expand_keywords([
        "ornate", "elaborate", "decorative", "detailed",
        "intricate", "complex design", "embellished",
    ]),
    "vintage_retro": expand_keywords([
        "vintage", "retro", "old style", "classic style",
        "antique", "old school", "throwback",
    ]),
    "modern_contemporary": expand_keywords([
        "modern", "contemporary", "current", "new style",
        "modern design", "sleek", "cutting edge",
    ]),
    "cartoon": expand_keywords([
        "cartoon", "cartoony", "animated", "comic style",
        "anime style", "cartoonish",
    ]),
    "pixel_art": expand_keywords([
        "pixel art", "pixelated", "8-bit", "16-bit",
        "retro game", "pixel style",
    ]),
    "hand_drawn": expand_keywords([
        "hand drawn", "sketched", "drawn by hand",
        "pencil drawn", "hand painted", "handmade",
    ]),
}

TEXTURE = {
    "smooth": expand_keywords([
        "smooth", "sleek", "polished", "glossy", "silky",
        "velvety", "satiny", "glass-like",
    ]),
    "rough": expand_keywords([
        "rough", "coarse", "gritty", "bumpy", "uneven",
        "jagged", "rugged", "rocky texture",
    ]),
    "grainy": expand_keywords([
        "grainy", "noisy", "granular", "grain",
        "film grain", "pixelated", "speckled",
    ]),
    "metallic": expand_keywords([
        "metal", "metallic", "steel", "iron", "copper",
        "bronze", "silver", "gold", "chrome", "aluminum",
        "shiny metal", "brushed metal", "rusted metal",
    ]),
    "wooden": expand_keywords([
        "wood", "wooden", "timber", "lumber", "log",
        "hardwood", "pine", "oak", "mahogany", "walnut",
        "wood grain", "wooden surface", "plank",
    ]),
    "glassy": expand_keywords([
        "glass", "glassy", "crystal", "transparent",
        "translucent", "frosted glass", "stained glass",
    ]),
    "stone": expand_keywords([
        "stone", "rock", "marble", "granite", "slate",
        "concrete", "brick", "cobblestone", "gravel",
    ]),
    "fabric_textile": expand_keywords([
        "fabric", "cloth", "textile", "woven", "knitted",
        "cotton", "silk", "wool", "linen", "denim",
        "velvet", "lace", "leather", "suede", "fur",
    ]),
    "watery": expand_keywords([
        "water", "liquid", "fluid", "wet", "moist",
        "rippling", "reflective water", "watery surface",
    ]),
}

DEPTH = {
    "shallow": expand_keywords([
        "shallow depth", "blurred background", "bokeh",
        "out of focus background", "soft background",
        "portrait mode", "depth of field",
    ]),
    "deep": expand_keywords([
        "deep depth", "everything in focus", "sharp throughout",
        "deep focus", "infinite focus", "landscape focus",
    ]),
    "flat": expand_keywords([
        "flat", "two dimensional", "2d", "no depth",
        "flat design", "flat image", "flat surface",
    ]),
    "layered": expand_keywords([
        "layered", "multi-layered", "foreground and background",
        "depth layers", "multiple layers", "overlay",
    ]),
}

LIGHTING = {
    "natural": expand_keywords([
        "natural light", "sunlight", "daylight", "natural lighting",
        "sun lit", "sunlit", "day lit",
    ]),
    "artificial": expand_keywords([
        "artificial light", "indoor lighting", "lamp light",
        "fluorescent", "studio light", "flash photography",
    ]),
    "backlit": expand_keywords([
        "backlit", "backlight", "silhouette", "against the light",
        "rim light", "contre-jour",
    ]),
    "soft": expand_keywords([
        "soft light", "soft lighting", "diffused light",
        "gentle light", "ambient light", "even light",
    ]),
    "harsh": expand_keywords([
        "harsh light", "harsh lighting", "direct sunlight",
        "strong light", "hard light", "glaring",
    ]),
    "golden": expand_keywords([
        "golden light", "golden hour", "warm glow",
        "sunset light", "sunrise light", "magic hour",
    ]),
    "neon": expand_keywords([
        "neon", "neon light", "neon sign", "neon glow",
        "fluorescent", "colored light", "vibrant light",
    ]),
    "candlelight": expand_keywords([
        "candle", "candlelight", "firelight", "flame light",
        "dim light", "flickering light", "warm flame",
    ]),
}

TEMPERATURE = {
    "hot": expand_keywords(["hot", "scorching", "blazing", "sweltering", "boiling"]),
    "warm": expand_keywords(["warm", "mild", "pleasant", "temperate", "balmy"]),
    "cool": expand_keywords(["cool", "chilly", "crisp", "fresh"]),
    "cold": expand_keywords(["cold", "freezing", "icy", "frozen", "frigid", "bitter cold"]),
}

DIRECTION = {
    "looking_left": expand_keywords(["looking left", "facing left", "left side", "on the left"]),
    "looking_right": expand_keywords(["looking right", "facing right", "right side", "on the right"]),
    "looking_up": expand_keywords(["looking up", "facing up", "upward", "looking upward"]),
    "looking_down": expand_keywords(["looking down", "facing down", "downward", "looking downward"]),
    "centered": expand_keywords(["centered", "center", "middle", "in the center", "centrally"]),
}

ACTION = {
    "static": expand_keywords(["static", "still", "motionless", "stationary", "fixed", "unmoving"]),
    "dynamic": expand_keywords(["dynamic", "moving", "in motion", "active", "action", "in action"]),
    "flying": expand_keywords(["flying", "in flight", "soaring", "airborne", "hovering"]),
    "swimming": expand_keywords(["swimming", "diving", "floating on water", "submerged"]),
    "running": expand_keywords(["running", "sprinting", "dashing", "racing"]),
    "jumping": expand_keywords(["jumping", "leaping", "bounding", "vaulting"]),
}

SCALE = {
    "macro_micro": expand_keywords(["macro", "micro", "tiny", "small scale"]),
    "close_up": expand_keywords(["close up", "close-up", "zoomed", "tight"]),
    "medium": expand_keywords(["medium shot", "mid shot", "waist up"]),
    "wide": expand_keywords(["wide", "long shot", "full body", "full view"]),
    "extreme_wide": expand_keywords(["extreme wide", "establishing shot", "vast"]),
}

SYMMETRY = {
    "symmetrical": expand_keywords(["symmetrical", "symmetric", "balanced", "mirrored", "reflection"]),
    "asymmetrical": expand_keywords(["asymmetrical", "asymmetric", "unbalanced", "offset"]),
}

DENSITY = {
    "sparse": expand_keywords(["sparse", "empty", "bare", "minimal", "scattered"]),
    "dense": expand_keywords(["dense", "crowded", "busy", "cluttered", "packed", "full of"]),
}

MATERIAL = {
    "wood": expand_keywords(["wood", "wooden", "timber", "log", "plank"]),
    "metal": expand_keywords(["metal", "metallic", "steel", "iron", "aluminum"]),
    "glass": expand_keywords(["glass", "glassy", "crystal", "transparent"]),
    "stone": expand_keywords(["stone", "rock", "marble", "granite", "concrete", "brick"]),
    "fabric": expand_keywords(["fabric", "cloth", "textile", "woven", "cotton", "silk"]),
    "plastic": expand_keywords(["plastic", "acrylic", "synthetic", "polymer"]),
    "paper": expand_keywords(["paper", "cardboard", "parchment", "papyrus"]),
    "leather": expand_keywords(["leather", "suede", "hide"]),
    "ceramic": expand_keywords(["ceramic", "pottery", "porcelain", "china"]),
}

PATTERN = {
    "striped": expand_keywords(["striped", "stripe", "stripes", "bands"]),
    "checkered": expand_keywords(["checkered", "checkerboard", "checked", "plaid"]),
    "dotted": expand_keywords(["dotted", "dots", "polka dot", "speckled"]),
    "floral": expand_keywords(["floral", "flower pattern", "botanical pattern"]),
    "geometric": expand_keywords(["geometric", "geometric pattern", "shapes pattern"]),
}

HUMIDITY = {
    "dry": expand_keywords(["dry", "arid", "parched", "desert"]),
    "humid": expand_keywords(["humid", "moist", "damp", "muggy", "wet"]),
}

QUALITY = {
    "sharp": expand_keywords(["sharp", "crisp", "clear", "high resolution", "detailed"]),
    "blurry": expand_keywords(["blurry", "blurred", "out of focus", "fuzzy", "unclear"]),
    "noisy": expand_keywords(["noisy", "grainy", "pixelated", "low quality"]),
}

AGE_ERA = {
    "modern": expand_keywords(["modern", "contemporary", "current", "recent"]),
    "vintage": expand_keywords(["vintage", "retro", "old-fashioned", "classic"]),
    "ancient": expand_keywords(["ancient", "antique", "historic", "old"]),
    "futuristic": expand_keywords(["futuristic", "sci-fi", "futuristic looking", "space age"]),
}

ORIENTATION = {
    "landscape": expand_keywords(["landscape", "horizontal", "wide", "wider than tall"]),
    "portrait": expand_keywords(["portrait", "vertical", "tall", "taller than wide"]),
    "square": expand_keywords(["square", "equal sides"]),
}

CAMERA_DISTANCE = {
    "extreme_close": expand_keywords(["extreme close", "macro", "magnified"]),
    "close": expand_keywords(["close", "close up", "near"]),
    "medium": expand_keywords(["medium", "mid distance", "waist shot"]),
    "far": expand_keywords(["far", "distant", "long shot", "in the distance"]),
}

NATURALNESS = {
    "natural": expand_keywords(["natural", "organic", "nature", "real"]),
    "artificial": expand_keywords(["artificial", "synthetic", "man-made", "manufactured"]),
}

COMPLEXITY = {
    "simple": expand_keywords(["simple", "basic", "plain", "minimal"]),
    "complex": expand_keywords(["complex", "complicated", "intricate", "detailed"]),
}

SOUND_IMPRESSION = {
    "loud": expand_keywords(["loud", "noisy", "boisterous", "thundering"]),
    "quiet": expand_keywords(["quiet", "silent", "hushed", "peaceful"]),
}

SMELL_IMPRESSION = {
    "fresh": expand_keywords(["fresh", "clean", "crisp"]),
    "earthy": expand_keywords(["earthy", "musty", "damp"]),
    "sweet": expand_keywords(["sweet", "fragrant", "flowery"]),
}

TASTE_IMPRESSION = {
    "sweet": expand_keywords(["sweet", "sugary", "honeyed"]),
    "savory": expand_keywords(["savory", "salty", "umami"]),
    "bitter": expand_keywords(["bitter", "tart", "sour"]),
}

TOUCH_IMPRESSION = {
    "soft": expand_keywords(["soft", "fluffy", "furry", "plush"]),
    "hard": expand_keywords(["hard", "solid", "rigid", "stiff"]),
    "rough": expand_keywords(["rough", "coarse", "scratchy"]),
    "smooth": expand_keywords(["smooth", "slick", "silky"]),
}

# ═══════════════════════════════════════════════════════════════════
# ALL DIMENSIONS
# ═══════════════════════════════════════════════════════════════════

ALL_DIMENSIONS = {
    "source": SOURCE,
    "setting": SETTING,
    "environment": ENVIRONMENT,
    "subject": SUBJECT,
    "composition": COMPOSITION,
    "color": COLOR,
    "text": TEXT,
    "mood": MOOD,
    "time_of_day": TIME_OF_DAY,
    "season": SEASON,
    "weather": WEATHER,
    "style": STYLE,
    "texture": TEXTURE,
    "depth": DEPTH,
    "lighting": LIGHTING,
    "temperature": TEMPERATURE,
    "direction": DIRECTION,
    "action": ACTION,
    "scale": SCALE,
    "symmetry": SYMMETRY,
    "density": DENSITY,
    "material": MATERIAL,
    "pattern": PATTERN,
    "humidity": HUMIDITY,
    "quality": QUALITY,
    "age_era": AGE_ERA,
    "orientation": ORIENTATION,
    "camera_distance": CAMERA_DISTANCE,
    "naturalness": NATURALNESS,
    "complexity": COMPLEXITY,
    "sound": SOUND_IMPRESSION,
    "smell": SMELL_IMPRESSION,
    "taste": TASTE_IMPRESSION,
    "touch": TOUCH_IMPRESSION,
}

# ═══════════════════════════════════════════════════════════════════
# TOTAL KEYWORD COUNT
# ═══════════════════════════════════════════════════════════════════

def count_keywords():
    total = 0
    for dim_name, dim in ALL_DIMENSIONS.items():
        dim_total = sum(len(v) for v in dim.values())
        total += dim_total
        print(f"  {dim_name}: {dim_total:,} keywords across {len(dim)} labels")
    print(f"\n  TOTAL: {total:,} keywords across {len(ALL_DIMENSIONS)} dimensions")

if __name__ == '__main__':
    count_keywords()
