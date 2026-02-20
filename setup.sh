#!/bin/bash
# Setup script for qwen3-tts-cuda-graphs
# Creates a venv with uv, installs deps, and downloads models
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== Qwen3-TTS CUDA Graphs Setup ==="

# Check uv is available
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Create venv + install deps (skip if venv already exists)
if [ -f "$DIR/.venv/bin/python" ]; then
    echo "Venv already exists, skipping install. Delete .venv to force reinstall."
else
    echo "Creating venv and installing dependencies..."
    uv venv .venv --python 3.10
    uv pip install -e . --python .venv/bin/python

    # Install flash-attn if possible (optional, speeds up attention on datacenter GPUs)
    echo "Attempting to install flash-attn (optional)..."
    uv pip install flash-attn --python .venv/bin/python 2>/dev/null && echo "  flash-attn installed" || echo "  flash-attn not available (ok, will use manual attention)"
fi

# Verify CUDA
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')" || {
    echo ""
    echo "WARNING: CUDA not available. You may need to install a CUDA-enabled PyTorch wheel."
    echo "  uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --python .venv/bin/python"
}

# Pre-download models to HuggingFace cache
echo ""
echo "Pre-downloading models to HuggingFace cache..."
.venv/bin/python -c "
from huggingface_hub import snapshot_download

for model in ['Qwen3-TTS-12Hz-0.6B-Base', 'Qwen3-TTS-12Hz-1.7B-Base']:
    repo_id = f'Qwen/{model}'
    print(f'  {repo_id}: downloading...')
    snapshot_download(repo_id)
    print(f'  {repo_id}: done')
"

# Generate ref audio if missing
if [ ! -f "$DIR/ref_audio.wav" ]; then
    echo ""
    echo "Generating placeholder reference audio..."
    .venv/bin/python -c "
import numpy as np, soundfile as sf
sr = 16000
t = np.linspace(0, 1.0, sr, dtype=np.float32)
audio = 0.3 * np.sin(2 * np.pi * 440 * t)
sf.write('$DIR/ref_audio.wav', audio, sr)
print('  Generated placeholder ref_audio.wav (replace with real speech for best quality)')
"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the venv: source .venv/bin/activate"
echo "Run benchmark:     ./benchmark.sh"
