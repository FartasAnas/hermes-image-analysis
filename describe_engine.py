def generate_detailed_description(blip_caption, labels=None, metadata=None):
    """
    Generates a rich, natural-language description from BLIP caption
    and MAX classifier labels. Produces flowing paragraphs, not lists.
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
    
    # Flatten nested labels
    def flatten(labels_list):
        if not labels_list:
            return []
        result = []
        for item in labels_list:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(str(item).replace('_', ' '))
        return result
    
    subjects_flat = flatten(subjects)
    
    paragraphs = []
    
    # ═══════════════════════════════════════════════════════════
    # PARAGRAPH 1: What the image shows (main content)
    # ═══════════════════════════════════════════════════════════
    p1_parts = []
    
    # Clean up BLIP caption for natural reading
    caption = blip_caption.strip()
    if caption.startswith("this is a picture of"):
        caption = caption.replace("this is a picture of", "The image shows")
    elif caption.startswith("a detailed view of"):
        caption = caption.replace("a detailed view of", "This is a detailed view of")
    elif caption.startswith("the image shows"):
        pass  # Already good
    elif caption.startswith("a "):
        caption = "The image depicts " + caption
    else:
        caption = "The image shows " + caption
    
    # Capitalize first letter
    caption = caption[0].upper() + caption[1:]
    if not caption.endswith('.'):
        caption += '.'
    
    p1_parts.append(caption)
    
    # Source type
    if source in ('digital_abstract', 'diagram'):
        p1_parts.append("This is a digitally created or computer-generated composition.")
    elif source == 'screenshot':
        p1_parts.append("This appears to be a screenshot or digital interface capture.")
    elif source == 'painting':
        p1_parts.append("This is a painting, executed in traditional or digital media.")
    elif source == 'illustration':
        p1_parts.append("This is an illustration or graphic artwork.")
    elif source == 'drawing':
        p1_parts.append("This appears to be a drawing, sketch, or hand-rendered artwork.")
    elif source == 'map':
        p1_parts.append("This is a map or cartographic representation.")
    
    paragraphs.append(" ".join(p1_parts))
    
    # ═══════════════════════════════════════════════════════════
    # PARAGRAPH 2: Visual details — colors, lighting, composition
    # ═══════════════════════════════════════════════════════════
    p2_parts = []
    
    # Colors with specific descriptions
    if colors:
        warm_colors = [c for c in colors if c in ('warm_tones',)]
        cool_colors = [c for c in colors if c in ('cool_tones',)]
        dark_colors = [c for c in colors if c in ('dark_dim',)]
        bright_colors = [c for c in colors if c in ('bright_light',)]
        vibrant_colors = [c for c in colors if c in ('vibrant_colorful',)]
        mono_colors = [c for c in colors if c in ('monochrome_bw',)]
        contrast_colors = [c for c in colors if c in ('high_contrast',)]
        
        color_phrases = []
        if vibrant_colors:
            color_phrases.append("vibrant, saturated colors dominate the scene")
        if warm_colors and cool_colors:
            color_phrases.append("a mix of warm red-orange tones and cool blue-purple hues")
        elif warm_colors:
            color_phrases.append("warm tones of red, orange, and gold are prominent")
        elif cool_colors:
            color_phrases.append("cool tones of blue, purple, and teal set the palette")
        if dark_colors:
            color_phrases.append("set against a dark, deep background")
        if bright_colors:
            color_phrases.append("with bright, well-illuminated areas")
        if mono_colors:
            color_phrases.append("presented in black and white or monochrome")
        if contrast_colors:
            color_phrases.append("featuring strong contrast between light and shadow")
        
        if color_phrases:
            p2_parts.append("The color palette features " + ", ".join(color_phrases) + ".")
    
    # Lighting
    if lighting:
        light_str = ", ".join(l.replace('_', ' ') for l in lighting)
        p2_parts.append(f"The lighting is {light_str}.")
    
    # Composition
    if composition:
        comp_str = ", ".join(c.replace('_', ' ') for c in composition)
        p2_parts.append(f"The composition employs a {comp_str} perspective.")
    
    # Patterns
    if pattern:
        pat_str = ", ".join(p.replace('_', ' ') for p in pattern)
        p2_parts.append(f"Notable visual patterns include {pat_str}.")
    
    if p2_parts:
        paragraphs.append(" ".join(p2_parts))
    
    # ═══════════════════════════════════════════════════════════
    # PARAGRAPH 3: Subject matter and environment
    # ═══════════════════════════════════════════════════════════
    p3_parts = []
    
    if subjects_flat:
        # Group subjects thematically
        space_terms = {'space astronomy', 'stars astronomy', 'space', 'astronomy', 'planet', 'star'}
        network_terms = {'network technology', 'network', 'connected', 'web', 'lines'}
        nature_terms = {'plant', 'animal', 'bird', 'fish', 'insect', 'landscape', 'flower'}
        human_terms = {'person', 'people', 'man', 'woman', 'child'}
        building_terms = {'building', 'architecture', 'structure', 'house'}
        tech_terms = {'technology', 'device', 'computer', 'screen', 'phone'}
        
        subject_sets = []
        for s in subjects_flat:
            for term_set, label in [(space_terms, 'celestial or astronomical'),
                                     (network_terms, 'network or digital connectivity'),
                                     (nature_terms, 'natural or botanical'),
                                     (human_terms, 'human or portrait'),
                                     (building_terms, 'architectural or structural'),
                                     (tech_terms, 'technological or electronic')]:
                if s in term_set:
                    if label not in [x[0] for x in subject_sets]:
                        subject_sets.append((label, []))
                    for name, lst in subject_sets:
                        if name == label:
                            lst.append(s)
        
        if subject_sets:
            for label, items in subject_sets:
                p3_parts.append(f"The image contains {label} elements" + 
                              (f" including {', '.join(items)}" if items else "") + ".")
    
    if setting:
        p3_parts.append(f"The setting is {', '.join(s.replace('_',' ') for s in setting)}.")
    
    if environment:
        env_str = ", ".join(e.replace('_', ' ') for e in environment)
        p3_parts.append(f"The environment is characterized as {env_str}.")
    
    if material:
        mat_str = ", ".join(m.replace('_', ' ') for m in material)
        p3_parts.append(f"Notable materials and textures include {mat_str}.")
    
    if weather:
        p3_parts.append(f"Weather conditions appear to be {', '.join(w.replace('_',' ') for w in weather)}.")
    
    if time_of_day:
        p3_parts.append(f"The time of day is {', '.join(t.replace('_',' ') for t in time_of_day)}.")
    
    if p3_parts:
        paragraphs.append(" ".join(p3_parts))
    
    # ═══════════════════════════════════════════════════════════
    # PARAGRAPH 4: Mood, style, themes
    # ═══════════════════════════════════════════════════════════
    p4_parts = []
    
    if mood:
        mood_str = ", ".join(m.replace('_', ' ') for m in mood)
        p4_parts.append(f"The overall mood is {mood_str}.")
    
    if style:
        style_str = ", ".join(s.replace('_', ' ') for s in style)
        p4_parts.append(f"The visual style can be described as {style_str}.")
    
    if p4_parts:
        paragraphs.append(" ".join(p4_parts))
    
    # ═══════════════════════════════════════════════════════════
    # PARAGRAPH 5: Text and metadata
    # ═══════════════════════════════════════════════════════════
    p5_parts = []
    
    if text_info:
        if 'has_text' in text_info:
            if 'text_heavy' in text_info:
                p5_parts.append("The image contains a substantial amount of visible text or typographic content.")
                if 'sign_present' in text_info:
                    p5_parts.append("Signage, labels, or banners with text are visible.")
            else:
                p5_parts.append("Some text or lettering is visible within the image.")
        else:
            p5_parts.append("No visible text, labels, or writing appears in the image.")
    else:
        p5_parts.append("No visible text or lettering is present.")
    
    if metadata:
        dims = metadata.get('dimensions', '')
        kb = metadata.get('file_size_kb', '')
        brightness = metadata.get('avg_brightness', '')
        if dims:
            p5_parts.append(f"Technical details: {dims} pixels, {kb}KB file size, average brightness {brightness}/255.")
    
    if p5_parts:
        paragraphs.append(" ".join(p5_parts))
    
    return "\n\n".join(paragraphs)
