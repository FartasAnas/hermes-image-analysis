#!/usr/bin/env python3
"""
LLaVA Vision Engine — Drop-in replacement for BLIP
===================================================
Uses LLaVA-1.5-7B in 4-bit quantization (~5GB VRAM on RTX 3060).
Provides rich, detailed, promptable image descriptions.

Usage:
  from llava_engine import LLaVAEngine
  engine = LLaVAEngine()
  description = engine.describe(image_path, detail="rich")
"""
import os
import time

import torch


class LLaVAEngine:
    """LLaVA-1.5-7B vision-language model for detailed image descriptions."""

    def __init__(self, cache_dir=None, load_in_4bit=True):
        """
        Initialize LLaVA engine.
        
        Args:
            cache_dir: HF cache directory (default: from HF_HOME env)
            load_in_4bit: Use 4-bit quantization (~5GB VRAM) vs 8-bit (~8GB)
        """
        self.cache_dir = cache_dir or os.environ.get('HF_HOME', 'E:/hermes_tools/.hf')
        # Ensure cache_dir doesn't end with /hub (HF adds that internally)
        if self.cache_dir.endswith('/hub'):
            self.cache_dir = self.cache_dir[:-4]
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.processor = None
        self._loaded = False

    def _resolve_model_path(self):
        """Resolve the actual model path from HF cache (handles Windows symlink issues)."""
        model_id = "llava-hf/llava-1.5-7b-hf"
        # Try to find the snapshot in cache
        cache_model_dir = os.path.join(self.cache_dir, "hub",
                                       f"models--{model_id.replace('/', '--')}")
        refs_file = os.path.join(cache_model_dir, "refs", "main")
        if os.path.exists(refs_file):
            with open(refs_file) as f:
                snapshot_hash = f.read().strip()
            snapshot_dir = os.path.join(cache_model_dir, "snapshots", snapshot_hash)
            if os.path.exists(snapshot_dir):
                return snapshot_dir
        # Fall back to model ID (will try to download)
        return model_id

    def load(self):
        """Load the LLaVA model."""
        if self._loaded:
            return

        model_path = self._resolve_model_path()
        print("Loading LLaVA-1.5-7B...", end=" ", flush=True)
        t0 = time.time()

        from transformers import AutoProcessor, LlavaForConditionalGeneration

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True,
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True,
            )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )

        self._loaded = True
        vram = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"done ({time.time()-t0:.1f}s, {vram:.1f}GB VRAM)")

    def unload(self):
        """Free GPU memory."""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        self._loaded = False
        torch.cuda.empty_cache()

    def _load_image(self, image_path):
        """Load image, handling alpha channels. Delegates to shared utility."""
        from image_utils import load_image_safely
        return load_image_safely(image_path)

    def describe(self, image_path, detail="rich", max_tokens=300):
        """
        Generate a detailed description of an image.
        
        Args:
            image_path: Path to the image file
            detail: 'rich' (detailed), 'concise' (short), or custom prompt
            max_tokens: Maximum tokens in response
            
        Returns:
            dict with 'description', 'time_seconds', 'vram_gb'
        """
        if not self._loaded:
            self.load()

        prompts = {
            "rich": (
                "USER: <image>\nDescribe this image in rich detail. "
                "Include: the main subject, colors and lighting, background elements, "
                "composition, any visible text, motion effects, and the overall mood or theme.\n"
                "ASSISTANT:"
            ),
            "concise": (
                "USER: <image>\nDescribe this image concisely in 2-3 sentences.\n"
                "ASSISTANT:"
            ),
            "colors": (
                "USER: <image>\nWhat are the dominant colors in this image? "
                "Describe the color palette, lighting, and any gradients or color transitions.\n"
                "ASSISTANT:"
            ),
            "spatial": (
                "USER: <image>\nDescribe the spatial layout of this image. "
                "What is in the foreground, middle ground, and background? "
                "Are there any overlays, patterns, or superimposed elements?\n"
                "ASSISTANT:"
            ),
        }

        prompt = prompts.get(detail, detail)
        img = self._load_image(image_path)
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self._device)

        t0 = time.time()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        desc = self.processor.decode(output[0], skip_special_tokens=True)
        # Extract assistant response
        if "ASSISTANT:" in desc:
            desc = desc.split("ASSISTANT:")[-1].strip()

        elapsed = time.time() - t0
        vram = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0

        # Free temporary tensors from this inference to prevent VRAM fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "engine": "LLaVA-1.5-7B (4-bit)",
            "description": desc,
            "time_seconds": round(elapsed, 2),
            "vram_gb": round(vram, 2),
        }

    def ask(self, image_path, question, max_tokens=200):
        """Ask a specific question about an image."""
        if not self._loaded:
            self.load()

        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        img = self._load_image(image_path)
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self._device)

        t0 = time.time()
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens)

        answer = self.processor.decode(output[0], skip_special_tokens=True)
        if "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[-1].strip()

        return {
            "question": question,
            "answer": answer,
            "time_seconds": round(time.time() - t0, 2),
        }


# ═══════════════════════════════════════════════════════════
# Singleton for reuse across pipeline calls
# ═══════════════════════════════════════════════════════════

_llava_instance = None

def get_llava():
    """Get or create the global LLaVA engine instance."""
    global _llava_instance
    if _llava_instance is None:
        _llava_instance = LLaVAEngine()
    return _llava_instance

def release_llava():
    """Release the global LLaVA engine to free VRAM."""
    global _llava_instance
    if _llava_instance:
        _llava_instance.unload()
        _llava_instance = None


# ═══════════════════════════════════════════════════════════
# CLI test
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python llava_engine.py <image_path> [detail_level]")
        sys.exit(1)

    engine = LLaVAEngine()
    result = engine.describe(sys.argv[1], detail=sys.argv[2] if len(sys.argv) > 2 else "rich")
    print(f"\n{'='*60}")
    print(result['description'])
    print(f"{'='*60}")
    print(f"Time: {result['time_seconds']}s, VRAM: {result['vram_gb']}GB")
    engine.unload()
