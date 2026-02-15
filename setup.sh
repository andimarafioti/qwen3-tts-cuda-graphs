#!/bin/bash
# Setup script for qwen3-tts-cuda-graphs
# Installs dependencies and downloads models from HuggingFace Hub
set -e

echo "=== Qwen3-TTS CUDA Graphs Setup ==="

# Check Python
PY="${PYTHON:-python3}"
echo "Using Python: $($PY --version 2>&1)"

# Check torch
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')" || {
    echo "ERROR: PyTorch with CUDA is required. Install it first:"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
    exit 1
}

# Install Python deps
echo ""
echo "Installing dependencies..."
pip install -q transformers soundfile numpy qwen-tts 2>/dev/null || pip install -q transformers soundfile numpy

# Download models
echo ""
echo "Downloading models from HuggingFace Hub..."
$PY -c "
from huggingface_hub import snapshot_download
import os

models_dir = os.path.join(os.path.dirname(os.path.abspath('$0')), 'models')
os.makedirs(models_dir, exist_ok=True)

for model in ['Qwen3-TTS-12Hz-0.6B-Base', 'Qwen3-TTS-12Hz-1.7B-Base']:
    dest = os.path.join(models_dir, model)
    if os.path.exists(dest):
        print(f'  {model}: already downloaded')
    else:
        print(f'  {model}: downloading...')
        snapshot_download(f'Qwen/{model}', local_dir=dest)
        print(f'  {model}: done')
"

# Generate ref audio if missing
REF_AUDIO="./ref_audio.wav"
if [ ! -f "$REF_AUDIO" ]; then
    echo ""
    echo "Generating reference audio..."
    $PY -c "
import numpy as np, soundfile as sf
# Generate a simple reference audio (1s of silence + tone)
# For best quality, replace with a real speech sample
sr = 16000
t = np.linspace(0, 1.0, sr, dtype=np.float32)
audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
sf.write('$REF_AUDIO', audio, sr)
print('  Generated placeholder ref_audio.wav (replace with real speech for best quality)')
"
fi

echo ""
echo "=== Setup complete ==="
echo "Run benchmark:  ./benchmark.sh"
