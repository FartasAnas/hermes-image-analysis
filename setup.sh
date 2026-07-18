# Hermes Image Analysis — Setup Script
# Run once to configure environment variables (prevents C: drive pollution)
# Source this file: source setup.sh

# Detect non-C: drive
if [ -d "/d" ]; then DRIVE="D"; elif [ -d "/e" ]; then DRIVE="E"; else DRIVE="D"; fi
echo "Using drive: $DRIVE"

# Create directories
mkdir -p "${DRIVE}:/hermes_tools/"{cache/doctr,cache/easyocr,cache/torch,.hf,temp,scripts,config}

# Set environment variables
export HF_HOME="${DRIVE}:/hermes_tools/.hf"
export HUGGINGFACE_HUB_CACHE="${DRIVE}:/hermes_tools/.hf/hub"
export TRANSFORMERS_CACHE="${DRIVE}:/hermes_tools/.hf/hub"
export DOCTR_CACHE_DIR="${DRIVE}:/hermes_tools/cache/doctr"
export EASYOCR_MODULE_PATH="${DRIVE}:/hermes_tools/cache/easyocr"
export XDG_CACHE_HOME="${DRIVE}:/hermes_tools/cache"
export TORCH_HOME="${DRIVE}:/hermes_tools/cache/torch"

echo "Environment configured. Add these lines to ~/.bashrc for persistence."
echo ""
echo "To add automatically:"
echo "  source $(pwd)/setup.sh >> ~/.bashrc"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
uv pip install easyocr python-doctr transformers torch pillow 2>&1 | tail -3

# Fix torch metadata bug if present
python -c "
import importlib.metadata as im
if im.version('torch') is None:
    import torch
    site = __import__('os').path.dirname(torch.__file__) + '/../site-packages'
    with open(site + '/sitecustomize.py', 'w') as f:
        f.write('''
import importlib.metadata as _im
_v = _im.version
def _pv(p):
    x = _v(p)
    if x is None and p==\"torch\":
        import torch; return torch.__version__.split(\"+\")[0]
    return x
_im.version = _pv
''')
    print('Fixed torch version detection bug')
else:
    print('Torch version OK')
" 2>/dev/null

echo ""
echo "✅ Setup complete. Run: python analyze_image.py <image_path>"
