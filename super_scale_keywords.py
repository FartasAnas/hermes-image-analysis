#!/usr/bin/env python3
"""
SUPER-SCALE Keyword Generator — Targets 1.5M+ keywords
======================================================
Adds massive word lists and higher-order combinations.
"""

# ═══════════════════════════════════════════════════════════════
# MASSIVE WORD LISTS
# ═══════════════════════════════════════════════════════════════

# Top 500 most common English nouns (relevant to image description)
COMMON_NOUNS = """
time year people way day man woman child world life hand part place case 
week company system program question government number night point home 
water room mother area money story fact month lot right study book eye 
job word business issue side kind head house service friend father power 
hour game line end member law car city community name president team 
minute idea body information back parent face others level office door 
health person art war history party result morning reason research girl 
guy moment air teacher force education foot boy age policy process music 
market sense nation plan college interest death experience effect use 
class control care field development role change movement light series 
animal heart figure fire road letter science rule food sister table 
nature order practice soul doctor film window culture million product 
blood ground condition paper center article brother energy period course 
summer effect building century beach machine section station language 
wall forest grass plant tree flower sun rain snow ice wind storm sky 
moon star cloud rock stone sand soil dust smoke steam fog shadow wave 
lake river ocean sea stream pond pool water fall island mountain hill 
valley desert plain field garden park wood jungle cave cliff bridge 
tower castle church temple mosque palace fort wall gate door window 
roof floor ceiling stair hall room kitchen bath bedroom living dining 
chair table desk bed shelf cabinet drawer lamp clock mirror picture 
painting sculpture statue vase pot bowl plate cup glass bottle jar 
box bag basket container tool weapon gun sword knife shield helmet 
armor clothing shoe boot hat cap coat jacket shirt pants dress skirt 
sock glove belt tie scarf ring necklace bracelet watch chain jewel 
gem diamond pearl gold silver iron copper bronze steel aluminum 
plastic rubber glass ceramic paper cloth leather fur feather wool 
cotton silk linen rope string wire cable chain pipe tube sheet 
board panel frame pole stick rod bar block ball wheel gear motor 
engine machine device tool instrument equipment apparatus vehicle 
car truck bus train plane ship boat bicycle motorcycle scooter 
computer phone tablet camera television radio speaker microphone 
headphone screen monitor keyboard mouse printer scanner copier 
food drink meal breakfast lunch dinner snack dessert cake bread 
rice pasta noodle soup salad sandwich burger pizza taco sushi 
fruit apple banana orange grape lemon lime melon berry cherry 
peach pear plum mango pineapple coconut vegetable carrot potato 
tomato onion garlic pepper corn bean pea lettuce spinach broccoli 
meat beef pork chicken fish shrimp crab lobster egg cheese milk 
butter cream yogurt sauce spice herb salt sugar honey oil vinegar 
animal dog cat horse cow pig sheep goat deer bear wolf fox rabbit 
squirrel mouse rat bird eagle hawk owl duck goose swan chicken 
turkey fish shark whale dolphin seal turtle frog snake lizard 
crocodile alligator insect spider ant bee butterfly fly mosquito 
plant flower rose lily daisy tulip sunflower orchid tree oak 
pine palm bamboo cactus grass fern moss mushroom fungus bacteria 
person man woman boy girl baby child adult teenager elder lady 
gentleman worker farmer doctor nurse teacher student soldier 
police firefighter pilot driver cook chef artist musician 
actor singer dancer writer poet painter sculptor photographer 
athlete runner swimmer boxer wrestler player coach referee 
king queen prince princess emperor empress president leader 
boss manager director executive officer secretary clerk agent
""".split()

# Remove empty strings and duplicates
COMMON_NOUNS = list(set(w.strip().lower() for w in COMMON_NOUNS if w.strip() and len(w.strip()) > 1))

# Top adjectives for image description
COMMON_ADJECTIVES = """
good new first last long great little own other old right big high 
small large different next early young important few public bad same 
able possible late hard real best better economic strong free true 
full special easy clear recent certain personal open red green blue 
yellow orange purple pink brown black white gray grey dark light 
bright soft hard smooth rough wet dry hot cold warm cool fresh 
clean dirty fast slow heavy light sharp dull thick thin wide narrow 
tall short deep shallow loud quiet rich poor happy sad angry calm 
excited nervous proud ashamed jealous curious confused surprised 
beautiful ugly gorgeous hideous pretty handsome cute adorable 
elegant graceful clumsy awkward gentle fierce mild spicy sweet 
salty sour bitter tasty delicious disgusting fragrant stinky 
ancient modern vintage retro antique classic contemporary 
traditional futuristic sleek rustic ornate simple complex 
busy empty crowded sparse dense open closed bright dim 
colorful plain vibrant muted shiny matte glossy dull 
transparent opaque translucent reflective absorbent 
smooth rough bumpy flat curved straight crooked round 
square rectangular triangular oval circular spherical 
wooden metallic plastic glass ceramic stone concrete 
brick marble granite slate paper fabric leather rubber 
furry feathery hairy scaly slimy sticky powdery grainy 
silky velvety cottony woolly sandy muddy dusty smoky 
steamy foggy misty cloudy sunny rainy snowy windy 
stormy calm turbulent peaceful chaotic orderly messy 
neat tidy cluttered organized disorganized symmetrical 
asymmetrical balanced unbalanced stable unstable solid 
liquid gaseous frozen melted boiling burning glowing 
sparkling shimmering dazzling blinding dim flickering 
""".split()

COMMON_ADJECTIVES = list(set(w.strip().lower() for w in COMMON_ADJECTIVES if w.strip() and len(w.strip()) > 1))

# Expanded colors (beyond basic ones)
EXTENDED_COLORS = """
red blue green yellow orange purple pink brown black white gray grey 
cyan teal magenta maroon navy olive lime aqua coral gold silver bronze 
copper beige tan cream ivory lavender mint peach salmon turquoise violet 
indigo crimson scarlet amber jade ruby sapphire emerald charcoal slate 
khaki mahogany ebony alabaster onyx pearl opal topaz garnet amethyst 
citrine peridot lapis malachite obsidian quartz sandstone limestone 
granite basalt rust patina verdigris terracotta ochre umber sienna 
cerulean ultramarine cobalt azure sky midnight powder royal electric 
neon fluorescent pastel pale deep dark light bright muted vivid warm 
cool hot cold fiery earthy oceanic tropical arctic desert forest 
jungle savanna tundra alpine urban industrial metallic iridescent 
pearlescent opalescent translucent transparent opaque frosted 
""".split()

EXTENDED_COLORS = list(set(w.strip().lower() for w in EXTENDED_COLORS if w.strip() and len(w.strip()) > 1))

# Common BLIP phrase templates
BLIP_TEMPLATES = [
    "a {a1} {n1}",
    "a {a1} {a2} {n1}",
    "a {c1} {n1}",
    "a {c1} and {c2} {n1}",
    "a {a1} {c1} {n1}",
    "{n1} with a {n2}",
    "{n1} and {n2}",
    "{n1} in the {n3}",
    "a {n1} on a {n2}",
    "a {n1} with {n2}",
    "the {n1} is {c1}",
    "a {c1} background with {n1}",
    "a {a1} {n1} with {n2}",
    "{n1} sitting on a {n2}",
    "{n1} standing in a {n3}",
]

# ═══════════════════════════════════════════════════════════════
# MASSIVE COMBINATORIAL GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_template(template, word_bank, max_samples=500000):
    """
    Fill a template with words from the bank.
    Returns sampled set if too large.
    """
    import random
    results = set()
    nouns = word_bank['n'][:200]  # Top 200 nouns
    adjs = word_bank['a'][:60]   # Top 60 adjectives
    colors = word_bank['c'][:60]  # Top 60 colors
    
    # Generate all combinations for small templates, sample for large ones
    if "{c1}" in template and "{c2}" in template and "{n1}" in template:
        # 3 variables → ~60*60*200 = 720K combinations → sample
        for _ in range(min(max_samples, 500000)):
            c1 = random.choice(colors)
            c2 = random.choice(colors)
            n1 = random.choice(nouns)
            if c1 != c2:
                text = template.replace("{c1}", c1).replace("{c2}", c2).replace("{n1}", n1)
                results.add(text)
    elif "{a1}" in template and "{n1}" in template and "{a2}" not in template:
        # 2 variables → 60*200 = 12K combinations → all
        for a1 in adjs:
            for n1 in nouns:
                text = template.replace("{a1}", a1).replace("{n1}", n1)
                results.add(text)
    elif "{a1}" in template and "{a2}" in template and "{n1}" in template:
        # 3 variables → sample
        for _ in range(min(max_samples, 300000)):
            a1 = random.choice(adjs)
            a2 = random.choice(adjs)
            n1 = random.choice(nouns)
            if a1 != a2:
                text = template.replace("{a1}", a1).replace("{a2}", a2).replace("{n1}", n1)
                results.add(text)
    elif "{c1}" in template and "{n1}" in template:
        # Color + noun: 60*200 = 12K
        for c1 in colors:
            for n1 in nouns:
                text = template.replace("{c1}", c1).replace("{n1}", n1)
                results.add(text)
    elif "{n1}" in template and "{n2}" in template:
        # Two nouns: 200*200 = 40K → sample
        for _ in range(min(max_samples, 100000)):
            n1 = random.choice(nouns)
            n2 = random.choice(nouns)
            if n1 != n2:
                text = template.replace("{n1}", n1).replace("{n2}", n2)
                results.add(text)
    elif "{n1}" in template and "{n3}" in template:
        for _ in range(min(max_samples, 50000)):
            n1 = random.choice(nouns)
            n3 = random.choice(nouns)
            if n1 != n3:
                text = template.replace("{n1}", n1).replace("{n3}", n3)
                results.add(text)
    elif "{n1}" in template:
        for n1 in nouns:
            results.add(template.replace("{n1}", n1))
    
    return results

def generate_massive():
    """Generate 1.5M+ keywords from templates and combinatorial explosion."""
    word_bank = {
        'n': COMMON_NOUNS,
        'a': COMMON_ADJECTIVES,
        'c': EXTENDED_COLORS,
    }
    
    all_keywords = set()
    
    print(f"Word bank: {len(COMMON_NOUNS)} nouns, {len(COMMON_ADJECTIVES)} adjectives, {len(EXTENDED_COLORS)} colors")
    print(f"Generating from {len(BLIP_TEMPLATES)} templates...\n")
    
    for template in BLIP_TEMPLATES:
        kw_set = generate_template(template, word_bank)
        all_keywords.update(kw_set)
        print(f"  {template}: {len(kw_set):,}")
    
    # Also add all individual nouns, adjectives, colors with BLIP prefixes
    print(f"\n  Prefix patterns...")
    prefixes = ["a ", "the ", "a large ", "a small ", "a beautiful ", "a close up of a "]
    for prefix in prefixes:
        for noun in COMMON_NOUNS[:300]:
            all_keywords.add(f"{prefix}{noun}")
        for adj in COMMON_ADJECTIVES[:100]:
            all_keywords.add(f"{prefix}{adj}")
    print(f"  Total after prefixes: {len(all_keywords):,}")
    
    # Add all color variations
    print(f"  Color patterns...")
    for c1 in EXTENDED_COLORS[:80]:
        all_keywords.add(f"{c1} background")
        all_keywords.add(f"a {c1} background")
        all_keywords.add(f"{c1} colored")
        all_keywords.add(f"{c1} and white")
        all_keywords.add(f"{c1} and black")
        for c2 in EXTENDED_COLORS[:40]:
            if c1 != c2:
                all_keywords.add(f"{c1} and {c2}")
    print(f"  Total after colors: {len(all_keywords):,}")
    
    return all_keywords

if __name__ == '__main__':
    kw = generate_massive()
    print(f"\n{'='*50}")
    print(f"FINAL TOTAL: {len(kw):,} keywords")
    print(f"{'='*50}")
