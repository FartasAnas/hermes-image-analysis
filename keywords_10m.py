#!/usr/bin/env python3
"""
10M KEYWORD GENERATOR — Massive Combinatorial Expansion
=======================================================
Uses the top 10,000 English words + combinatorial patterns
to generate 10 million+ image description keywords.

Strategy:
  1. Download top 10K English words from Google frequency list
  2. Generate 2-word bigrams (selective — image-relevant)
  3. Generate 3-word and 4-word patterns
  4. Use all existing keyword sources as base
  5. Efficient memory usage — write to disk in batches, not all in RAM
"""
import urllib.request, os, sys, json, random, math
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# WORD LIST ACQUISITION
# ═══════════════════════════════════════════════════════════════

def download_word_list():
    """Download top 10K English words."""
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        words = resp.read().decode('utf-8').splitlines()
    return [w.lower().strip() for w in words if 3 <= len(w) <= 15 and w.isalpha() and w.isascii()]

# Image-relevant word subsets (words likely to appear in image descriptions)
IMAGE_NOUNS = """
photo picture image photograph snapshot screenshot painting drawing sketch 
illustration diagram chart graph map background foreground scene view landscape 
portrait panorama closeup macro sky cloud sun moon star mountain hill valley 
river lake ocean sea beach coast shore wave sand rock stone tree flower plant 
grass leaf forest wood garden park field meadow path road street highway bridge 
building house home room door window wall roof floor ceiling table chair desk 
bed sofa couch shelf cabinet drawer lamp light candle bulb clock watch 
phone computer laptop tablet screen monitor keyboard mouse printer camera lens 
book magazine newspaper letter note card paper pen pencil paint brush canvas 
color pattern texture shape form line circle square triangle dot grid stripe 
person man woman child boy girl baby face eye hand foot body hair head arm 
leg finger toe nose mouth ear smile frown tear laugh cry animal dog cat bird 
fish horse cow pig sheep goat deer bear wolf fox rabbit mouse rat squirrel 
insect butterfly bee ant spider fly mosquito beetle food fruit vegetable meat 
bread cake rice pasta pizza salad soup drink water coffee tea wine beer milk 
juice car truck bus train plane boat ship bicycle motorcycle vehicle transport 
bag box bottle cup glass plate bowl jar basket container tool machine device 
clothing shirt pants dress jacket coat shoe boot hat cap glove scarf jewelry 
ring watch necklace bracelet earring gold silver metal wood plastic glass 
stone concrete brick fabric leather paper cardboard sign label logo text 
word letter number symbol flag banner poster billboard sculpture statue 
art artwork craft decoration ornament gift present toy game sport ball 
bat racket club diamond spade heart umbrella bag luggage suitcase 
""".split()

IMAGE_ADJECTIVES = """
big small large little tiny huge enormous massive gigantic miniature 
tall short long wide narrow thick thin heavy light dark bright dim 
colorful plain simple complex detailed vibrant muted vivid pale deep 
shallow soft hard smooth rough wet dry hot cold warm cool fresh stale 
clean dirty new old young ancient modern vintage antique retro classic 
beautiful pretty ugly gorgeous hideous cute adorable handsome stunning 
elegant graceful clumsy awkward fast slow quick rapid gentle fierce 
happy sad angry calm excited nervous proud brave scared shy quiet loud 
noisy silent peaceful chaotic busy empty full crowded sparse dense 
open closed broken fixed intact shiny dull matte glossy transparent 
opaque reflective clear blurry sharp fuzzy bright dark light heavy 
wooden metallic plastic glass ceramic stone concrete fabric paper 
leather rubber golden silver bronze copper iron steel aluminum rusty 
polished weathered painted carved molded sculpted woven knitted sewn 
printed written drawn sketched painted colorful monochrome grayscale 
sepia pastel neon fluorescent glowing sparkling shimmering dazzling 
""".split()

# Remove duplicates and ensure uniqueness
IMAGE_NOUNS = list(set(IMAGE_NOUNS))
IMAGE_ADJECTIVES = list(set(IMAGE_ADJECTIVES))
IMAGE_NOUNS.sort()
IMAGE_ADJECTIVES.sort()

# ═══════════════════════════════════════════════════════════════
# PATTERN GENERATORS
# ═══════════════════════════════════════════════════════════════

def generate_bigrams(word_list, max_pairs=3_000_000):
    """Generate 2-word combinations from a word list (selective)."""
    n = len(word_list)
    # Full bigrams would be n² — we sample randomly for large n
    if n * n <= max_pairs:
        # Generate all
        for w1 in word_list:
            for w2 in word_list:
                if w1 != w2:
                    yield f"{w1} {w2}"
    else:
        # Sample
        samples = min(max_pairs, n * n)
        seen = set()
        attempts = 0
        while len(seen) < samples and attempts < samples * 2:
            w1 = random.choice(word_list)
            w2 = random.choice(word_list)
            if w1 != w2:
                pair = f"{w1} {w2}"
                if pair not in seen:
                    seen.add(pair)
                    yield pair
            attempts += 1

def generate_patterns(nouns, adjectives, max_total=10_000_000):
    """
    Generate keyword patterns from word lists.
    Returns total count and saves batches to a set.
    """
    all_kw = set()
    
    patterns = [
        # Pattern: (description, generator function, target_count)
        ("adj+noun 'a {adj} {noun}'", lambda: (
            f"a {adj} {noun}"
            for adj in random.sample(adjectives, min(500, len(adjectives)))
            for noun in random.sample(nouns, min(500, len(nouns)))
        ), 250000),
        
        ("adj+adj+noun 'a {adj} {adj2} {noun}'", lambda: (
            f"a {adj} {adj2} {noun}"
            for adj in random.sample(adjectives, min(200, len(adjectives)))
            for adj2 in random.sample(adjectives, min(150, len(adjectives)))
            for noun in random.sample(nouns, min(80, len(nouns)))
            if adj != adj2
        ), 2000000),
        
        ("adj+color+noun 'a {adj} {color} {noun}'", lambda: (
            f"a {adj} {color} {noun}"
            for adj in random.sample(adjectives, min(100, len(adjectives)))
            for color in random.sample(EXTENDED_COLORS, min(50, len(EXTENDED_COLORS)))
            for noun in random.sample(nouns, min(200, len(nouns)))
        ), 500000),
        
        ("adj+adj+noun extended", lambda: (
            f"a {adj} {adj2} {noun}"
            for adj in random.sample(adjectives, min(400, len(adjectives)))
            for adj2 in random.sample(adjectives, min(400, len(adjectives)))
            for noun in random.sample(nouns, min(80, len(nouns)))
            if adj != adj2
        ), 8000000),
        
        ("article+adj+noun 'the {adj} {noun}'", lambda: (
            f"the {adj} {noun}"
            for adj in random.sample(adjectives, min(500, len(adjectives)))
            for noun in random.sample(nouns, min(500, len(nouns)))
        ), 250000),
        
        ("noun+prep+noun '{n1} with {n2}'", lambda: (
            f"{n1} with {n2}"
            for n1 in random.sample(nouns, min(500, len(nouns)))
            for n2 in random.sample(nouns, min(500, len(nouns)))
            if n1 != n2
        ), 250000),
        
        ("noun+prep+noun '{n1} on a {n2}'", lambda: (
            f"{n1} on a {n2}"
            for n1 in random.sample(nouns, min(500, len(nouns)))
            for n2 in random.sample(nouns, min(500, len(nouns)))
            if n1 != n2
        ), 250000),
        
        ("noun+prep+noun '{n1} in the {n2}'", lambda: (
            f"{n1} in the {n2}"
            for n1 in random.sample(nouns, min(500, len(nouns)))
            for n2 in random.sample(nouns, min(500, len(nouns)))
            if n1 != n2
        ), 250000),
        
        ("bigrams from nouns", lambda: generate_bigrams(
            random.sample(nouns, min(500, len(nouns))), 
            max_pairs=250000
        ), 250000),
        
        ("color+noun 'a {color} {noun}'", lambda: (
            f"a {color} {noun}"
            for color in ["red","blue","green","yellow","orange","purple","pink",
                         "brown","black","white","gray","grey"]
            for noun in random.sample(nouns, min(1000, len(nouns)))
        ), 12000),
        
        ("BLIP prefix patterns", lambda: (
            f"a close up of a {noun}"
            for noun in random.sample(nouns, min(1000, len(nouns)))
        ), 1000),
        
        ("BLIP view patterns", lambda: (
            f"a view of {noun}"
            for noun in random.sample(nouns, min(1000, len(nouns)))
        ), 1000),
        
        ("BLIP background patterns", lambda: (
            f"a {color} background with {noun}"
            for color in ["red","blue","green","yellow","white","black","dark","light","gray","grey"]
            for noun in random.sample(nouns, min(500, len(nouns)))
        ), 5000),
    ]
    
    total = 0
    for name, generator_fn, target in patterns:
        count = 0
        try:
            for kw in generator_fn():
                if len(kw) >= 5 and len(kw) <= 120:  # Reasonable keyword length
                    all_kw.add(kw)
                    count += 1
                    if count >= target * 1.5:  # Allow some overshoot
                        break
                if len(all_kw) >= max_total:
                    break
            print(f"  {name}: {count:,} keywords (target: {target:,})")
            total += count
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
        
        if len(all_kw) >= max_total:
            break
    
    return all_kw, total

# Need this from super_scale_keywords
EXTENDED_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "brown", "black", "white", "gray", "grey", "cyan", "teal",
    "magenta", "maroon", "navy", "olive", "lime", "aqua", "coral",
    "gold", "silver", "beige", "tan", "cream", "ivory", "lavender",
    "mint", "peach", "salmon", "turquoise", "violet", "indigo",
    "crimson", "scarlet", "amber", "jade", "ruby", "sapphire",
    "emerald", "charcoal", "slate", "khaki",
]

def generate_10m():
    """Generate 10M keywords and return the set."""
    print("Downloading word list...")
    try:
        common_words = download_word_list()
        print(f"  Downloaded {len(common_words):,} common English words")
    except Exception as e:
        print(f"  Download failed: {e}")
        common_words = IMAGE_NOUNS + IMAGE_ADJECTIVES
    
    # Combine curated image words with common words
    all_nouns = list(set(IMAGE_NOUNS + common_words[:3000]))
    all_adjs = list(set(IMAGE_ADJECTIVES + common_words[:1000]))
    
    print(f"  Nouns: {len(all_nouns):,}, Adjectives: {len(all_adjs):,}")
    print(f"  Theoretical max adj+adj+noun: {len(all_adjs):,}² × {len(all_nouns):,} = {len(all_adjs)**2 * len(all_nouns):,}")
    print(f"\nGenerating patterns (target: 10M)...")
    
    kw_set, total = generate_patterns(all_nouns, all_adjs, max_total=10_500_000)
    
    print(f"\n{'='*50}")
    print(f"Generated: {len(kw_set):,} unique keywords")
    return kw_set

if __name__ == '__main__':
    random.seed(42)
    t0 = __import__('time').time()
    kw = generate_10m()
    print(f"Time: {__import__('time').time()-t0:.1f}s")
    
    # Save a sample for reference
    sample = list(kw)[:10]
    print(f"\nSample keywords:")
    for s in sample:
        print(f"  '{s}'")
