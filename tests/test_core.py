# Hermes Image Analysis Test Suite
# ==================================
# pytest tests for all core modules

import os
import sys

import numpy as np
import pytest
from PIL import Image

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_image():
    """Create a simple test image (100x100 RGB, red-to-blue gradient)."""
    img = Image.new('RGB', (100, 100))
    pixels = img.load()
    for y in range(100):
        for x in range(100):
            r = int(255 * (1 - x / 100))
            b = int(255 * (x / 100))
            pixels[x, y] = (r, 0, b)
    return img


@pytest.fixture
def sample_image_path(sample_image, tmp_path):
    """Save the sample image to a temp file and return its path."""
    path = tmp_path / "test_image.png"
    sample_image.save(path)
    return str(path)


@pytest.fixture
def dark_image_path(tmp_path):
    """Create a nearly black image."""
    img = Image.new('RGB', (50, 50), color=(5, 5, 5))
    path = tmp_path / "dark.png"
    img.save(path)
    return str(path)


@pytest.fixture
def terminal_screenshot_text():
    """Sample OCR text from a terminal screenshot."""
    return "curl -X GET https://api.github.com/repos/FartasAnas/hermes-image-analysis\npython analyze_image.py --engine llava\nnpm install express\nERROR: connection refused"


@pytest.fixture
def sample_caption():
    """Sample BLIP caption."""
    return "a man standing in a field looking at the mountains with a blue sky"


# ═══════════════════════════════════════════════════════════════
# TESTS: hermes_config.py
# ═══════════════════════════════════════════════════════════════

class TestHermesConfig:
    """Tests for dynamic drive detection and environment setup."""

    def test_gpu_available_returns_tuple(self):
        from hermes_config import gpu_available
        has_gpu, device, name = gpu_available()
        assert isinstance(has_gpu, bool)
        assert isinstance(device, str)
        assert isinstance(name, str)

    def test_gpu_info_str_returns_string(self):
        from hermes_config import gpu_info_str
        result = gpu_info_str()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_storage_drive_auto(self):
        from hermes_config import get_storage_drive
        drive = get_storage_drive(auto=True)
        assert isinstance(drive, str)
        assert len(drive) > 0

    def test_get_base_path(self):
        from hermes_config import get_base_path
        path = get_base_path('E:')
        assert 'hermes_tools' in path
        assert path.endswith('/') or path.endswith('\\')

    def test_setup_environment_returns_dict(self):
        from hermes_config import setup_environment
        config = setup_environment('E:')
        assert isinstance(config, dict)
        assert 'drive' in config
        assert 'base' in config
        assert 'cache' in config
        assert 'hf_home' in config


# ═══════════════════════════════════════════════════════════════
# TESTS: engine_config.py
# ═══════════════════════════════════════════════════════════════

class TestEngineConfig:
    """Tests for engine selection, scope, and TTY guard."""

    def test_detect_best_engine(self):
        from engine_config import detect_best_engine
        engine, reason = detect_best_engine()
        assert engine in ('blip', 'llava')
        assert isinstance(reason, str)

    def test_resolve_engine_force_blip(self):
        from engine_config import resolve_engine
        result = resolve_engine(force='blip')
        assert result == 'blip'

    def test_resolve_engine_force_llava(self):
        from engine_config import resolve_engine
        result = resolve_engine(force='llava')
        assert result == 'llava'

    def test_resolve_engine_invalid_force(self):
        from engine_config import resolve_engine
        # Should fall through to auto-detect
        result = resolve_engine(force='invalid')
        assert result in ('blip', 'llava')

    def test_is_interactive_with_env_var(self, monkeypatch):
        from engine_config import _is_interactive
        monkeypatch.setenv('HERMES_NON_INTERACTIVE', '1')
        assert _is_interactive() is False

    def test_is_interactive_with_env_var_true(self, monkeypatch):
        from engine_config import _is_interactive
        monkeypatch.setenv('HERMES_NON_INTERACTIVE', 'true')
        assert _is_interactive() is False

    def test_pid_is_alive_current_process(self):
        import os as _os

        from engine_config import _pid_is_alive
        assert _pid_is_alive(_os.getpid()) is True

    def test_pid_is_alive_dead_pid(self):
        from engine_config import _pid_is_alive
        # PID 0 and negative are always dead
        assert _pid_is_alive(0) is False
        assert _pid_is_alive(-1) is False

    def test_read_permanent_config_not_exists(self, tmp_path, monkeypatch):
        from engine_config import read_permanent_config
        # Redirect config path to temp
        monkeypatch.setattr('engine_config._get_permanent_config_path', lambda: str(tmp_path / "nonexistent.json"))
        assert read_permanent_config() == 'auto'

    def test_write_and_read_permanent_config(self, tmp_path, monkeypatch):
        from engine_config import (
            read_permanent_config,
            reset_permanent_config,
            write_permanent_config,
        )
        config_path = str(tmp_path / "engine_preference.json")
        monkeypatch.setattr('engine_config._get_permanent_config_path', lambda: config_path)

        write_permanent_config('blip')
        assert read_permanent_config() == 'blip'

        write_permanent_config('llava')
        assert read_permanent_config() == 'llava'

        reset_permanent_config()
        assert read_permanent_config() == 'auto'

    def test_session_state_write_read_clear(self, tmp_path, monkeypatch):
        from engine_config import (
            clear_session_state,
            read_session_state,
            write_session_state,
        )
        session_path = str(tmp_path / ".hermes_session")
        monkeypatch.setattr('engine_config._get_session_state_path', lambda: session_path)

        write_session_state('blip')
        # Should be readable (current PID is alive)
        result = read_session_state()
        assert result == 'blip'

        clear_session_state()
        assert read_session_state() is None


# ═══════════════════════════════════════════════════════════════
# TESTS: describe_engine.py
# ═══════════════════════════════════════════════════════════════

class TestDescribeEngine:
    """Tests for synthesis, tech detection, hallucination suppression."""

    def test_detect_technical_screenshot_terminal(self, terminal_screenshot_text):
        from describe_engine import detect_technical_screenshot
        is_tech, confidence, signals = detect_technical_screenshot(terminal_screenshot_text)
        assert is_tech is True
        assert confidence >= 0.3
        assert len(signals) >= 3
        assert 'curl' in signals

    def test_detect_technical_screenshot_normal_text(self):
        from describe_engine import detect_technical_screenshot
        is_tech, confidence, signals = detect_technical_screenshot(
            "The cat sat on the mat looking at the blue sky"
        )
        assert is_tech is False
        assert confidence == 0.0
        assert signals == []

    def test_detect_technical_screenshot_empty(self):
        from describe_engine import detect_technical_screenshot
        is_tech, confidence, signals = detect_technical_screenshot("")
        assert is_tech is False
        assert confidence == 0.0

    def test_suppress_hallucinations_on_tech(self):
        from describe_engine import _suppress_hallucinations
        ocr_text = "curl api.github.com | python script.py"
        vision_desc = "A person using a cell phone to browse the internet on a mountain"
        result, is_tech = _suppress_hallucinations(vision_desc, ocr_text)
        assert is_tech is True
        assert 'terminal' in result.lower() or 'command' in result.lower()
        assert 'person' not in result.lower()
        assert 'cell phone' not in result.lower()

    def test_suppress_hallucinations_no_tech(self):
        from describe_engine import _suppress_hallucinations
        ocr_text = "The quick brown fox jumps over the lazy dog"
        vision_desc = "A fox standing in a field"
        result, is_tech = _suppress_hallucinations(vision_desc, ocr_text)
        # Should pass through unchanged since not a tech screenshot
        assert result == vision_desc
        assert is_tech is False

    def test_suppress_hallucinations_low_confidence(self):
        from describe_engine import _suppress_hallucinations
        # No tech signals at all → should pass through unchanged
        ocr_text = "The party was loud with music and dancing"
        vision_desc = "A person at a party"
        result, is_tech = _suppress_hallucinations(vision_desc, ocr_text)
        # No tech signals matched → no suppression
        assert result == vision_desc
        assert is_tech is False

    def test_build_state_minimal(self):
        from describe_engine import build_state
        state = build_state(
            meta={"dimensions": "100x100", "file_size_kb": 5},
            engine_used="blip",
        )
        assert state['engine'] == 'blip'
        assert state['meta']['dimensions'] == '100x100'
        assert state['vision'] == {}
        assert state['ocr'] == {}
        assert state['pixel'] == {}

    def test_build_state_with_ocr_tech_detection(self):
        from describe_engine import build_state
        ocr_result = {
            "full_text": "curl https://api.example.com\ngit push origin main",
            "word_count": 8,
            "avg_confidence": 0.85,
            "engine": "DocTR",
            "words": [{"text": "curl", "confidence": 0.9}, {"text": "git", "confidence": 0.8}],
        }
        state = build_state(
            meta={"dimensions": "800x600"},
            ocr_result=ocr_result,
            engine_used="blip",
        )
        assert state['flags']['technical_screenshot'] is True
        assert state['flags']['tech_confidence'] > 0.3

    def test_synthesize_basic(self):
        from describe_engine import build_state, synthesize
        state = build_state(
            meta={"dimensions": "100x100", "file_size_kb": 10},
            vision_result={"caption": "a cat sitting on a couch", "engine": "BLIP", "time_seconds": 0.5},
            engine_used="blip",
        )
        result = synthesize(state)
        assert isinstance(result, str)
        assert len(result) > 10
        assert result.endswith('.')

    def test_synthesize_tech_screenshot(self, terminal_screenshot_text):
        from describe_engine import build_state, synthesize
        state = build_state(
            meta={"dimensions": "800x600", "file_size_kb": 50},
            vision_result={"caption": "A person using a cell phone on the beach", "engine": "LLaVA", "time_seconds": 14},
            ocr_result={
                "full_text": terminal_screenshot_text,
                "word_count": 20,
                "avg_confidence": 0.9,
                "engine": "DocTR",
                "words": [],
            },
            engine_used="llava",
        )
        result = synthesize(state)
        # Should suppress hallucinations and show terminal framing
        assert 'terminal' in result.lower() or 'command' in result.lower()

    def test_legacy_wrapper(self):
        from describe_engine import generate_detailed_description
        result = generate_detailed_description(
            "a red car parked on a street",
            labels={"source": ["photo"], "subject": ["vehicle"]},
            metadata={"dimensions": "640x480", "file_size_kb": 120},
        )
        assert isinstance(result, str)
        assert len(result) > 10

    def test_debug_dump(self):
        from describe_engine import build_state, debug_dump
        state = build_state(
            meta={"dimensions": "100x100", "file_size_kb": 5},
            vision_result={"caption": "test", "engine": "BLIP", "time_seconds": 0.5},
            engine_used="blip",
        )
        result = debug_dump(state)
        assert 'DEBUG' in result
        assert 'RAW ENGINE OUTPUTS' in result


# ═══════════════════════════════════════════════════════════════
# TESTS: pixel_analysis.py
# ═══════════════════════════════════════════════════════════════

class TestPixelAnalysis:
    """Tests for NumPy-vectorized pixel analysis."""

    def test_rgb_to_hsv_vectorized_output_shape(self):
        from pixel_analysis import _rgb_to_hsv_vectorized
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.float32)
        hsv = _rgb_to_hsv_vectorized(rgb)
        assert hsv.shape == (3, 3)
        # Red: H ~ 0, S ~ 1, V ~ 1
        assert 0 <= hsv[0, 0] <= 5  # H near 0
        assert hsv[0, 1] > 0.9  # S high
        assert hsv[0, 2] > 0.9  # V high

    def test_color_name_from_hsv(self):
        from pixel_analysis import _color_name_from_hsv
        assert _color_name_from_hsv(0, 1.0, 0.8) == 'red'
        assert _color_name_from_hsv(0, 0.0, 1.0) == 'near-white'
        assert _color_name_from_hsv(0, 0.0, 0.1) == 'near-black'
        assert _color_name_from_hsv(0, 0.1, 0.5) == 'gray'
        assert _color_name_from_hsv(240, 1.0, 0.8) == 'blue'
        assert _color_name_from_hsv(120, 1.0, 0.8) == 'green'

    def test_analyze_pixels_returns_dict(self, sample_image_path):
        from pixel_analysis import analyze_pixels
        result = analyze_pixels(sample_image_path)
        assert isinstance(result, dict)
        assert 'dominant_colors' in result
        assert 'vibrancy' in result
        assert 'brightness_desc' in result
        assert 'contrast' in result
        assert isinstance(result['dominant_colors'], list)

    def test_analyze_pixels_dark_image(self, dark_image_path):
        from pixel_analysis import analyze_pixels
        result = analyze_pixels(dark_image_path)
        assert result['brightness_desc'] == 'predominantly dark with a deep background'
        assert result['vibrancy'] == 'near-monochrome or desaturated'

    def test_pixel_analysis_to_text(self, sample_image_path):
        from pixel_analysis import analyze_pixels, pixel_analysis_to_text
        analysis = analyze_pixels(sample_image_path)
        text = pixel_analysis_to_text(analysis)
        assert isinstance(text, str)
        assert len(text) > 10


# ═══════════════════════════════════════════════════════════════
# TESTS: max_classifier.py
# ═══════════════════════════════════════════════════════════════

class TestMaxClassifier:
    """Tests for the MAX classifier and inverted index."""

    def test_classify_image_returns_all_dimensions(self):
        from max_classifier import ALL_DIMENSIONS, classify_image
        result = classify_image("a man standing in a field looking at the mountains")
        assert isinstance(result, dict)
        for dim_name in ALL_DIMENSIONS:
            assert dim_name in result
            assert isinstance(result[dim_name], list)

    def test_classify_image_photo_source(self):
        from max_classifier import classify_image
        result = classify_image("a man standing in a field looking at the mountains")
        assert 'photo' in result.get('source', [])

    def test_classify_image_digital_abstract(self):
        from max_classifier import classify_image
        result = classify_image("a diagram with connected lines and nodes showing a network")
        assert 'digital_abstract' in result.get('source', []) or 'diagram' in result.get('source', [])

    def test_classify_camera_digital(self):
        from max_classifier import classify_camera_digital
        assert classify_camera_digital("a photo of a cat") == 'camera'
        assert classify_camera_digital("a diagram of a network") == 'digital'

    def test_classify_image_dedupes_labels(self):
        from max_classifier import classify_image
        # Words that match the same dimension multiple times shouldn't duplicate
        result = classify_image("a red car red truck red bus")
        for dim, labels in result.items():
            assert len(labels) == len(set(labels)), f"Duplicates in {dim}: {labels}"

    def test_get_keyword_count(self):
        from max_classifier import get_keyword_count
        count = get_keyword_count()
        assert count > 100  # There are thousands of keywords


# ═══════════════════════════════════════════════════════════════
# TESTS: analyze_image.py (utility functions)
# ═══════════════════════════════════════════════════════════════

class TestAnalyzeImage:
    """Tests for image loading and metadata utilities."""

    def test_load_image_safely_rgb(self, sample_image_path):
        from analyze_image import _load_image_safely
        img = _load_image_safely(sample_image_path)
        assert img.mode == 'RGB'

    def test_load_image_safely_rgba(self, tmp_path):
        from analyze_image import _load_image_safely
        img = Image.new('RGBA', (50, 50), (255, 0, 0, 128))
        path = tmp_path / "rgba.png"
        img.save(path)
        loaded = _load_image_safely(str(path))
        assert loaded.mode == 'RGB'

    def test_load_image_safely_palette(self, tmp_path):
        from analyze_image import _load_image_safely
        img = Image.new('P', (50, 50))
        img.putpalette([0, 0, 0, 255, 0, 0] + [0, 0, 0] * 254)
        path = tmp_path / "palette.png"
        img.save(path)
        loaded = _load_image_safely(str(path))
        assert loaded.mode == 'RGB'

    def test_analyze_metadata(self, sample_image_path):
        from analyze_image import analyze_metadata
        meta = analyze_metadata(sample_image_path)
        assert isinstance(meta, dict)
        assert 'dimensions' in meta
        assert 'file_size_kb' in meta
        assert 'avg_brightness' in meta
        assert 'is_dark' in meta

    def test_analyze_metadata_dark(self, dark_image_path):
        from analyze_image import analyze_metadata
        meta = analyze_metadata(dark_image_path)
        assert meta['is_dark'] is True

    def test_analyze_metadata_corrupt_file(self, tmp_path):
        from analyze_image import analyze_metadata
        corrupt = tmp_path / "corrupt.png"
        corrupt.write_text("not an image")
        meta = analyze_metadata(str(corrupt))
        assert 'error' in meta
