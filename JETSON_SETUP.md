# Jetson AGX Orin Setup Guide

Complete setup instructions for running qwen3-tts-cuda-graphs on Jetson AGX Orin with JetPack 6.

## Quick Start

```bash
# Install Jetson PyTorch
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Install correct numpy
pip install "numpy<2"

# Clone and install local qwen-tts (avoids pip package conflicts)
git clone https://github.com/dffdeeq/Qwen3-TTS-streaming.git /tmp/Qwen3-TTS-streaming
pip install -e /tmp/Qwen3-TTS-streaming --no-deps

# Install dependencies
pip install transformers==4.57.3 soundfile librosa scikit-learn huggingface-hub

# Install this package
pip install -e .
```

## Critical Fix: transformers SDPA Patch

Jetson PyTorch 2.5.0a0 doesn't support the `enable_gqa` parameter. Apply this patch:

**File:** `.venv/lib/python3.10/site-packages/transformers/integrations/sdpa_attention.py`

**Find (around line 70):**
```python
        else:
            sdpa_kwargs = {"enable_gqa": True}
```

**Replace with:**
```python
        else:
            # Jetson PyTorch workaround
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
```

## Why These Fixes?

1. **Local qwen-tts:** Pip package has conflicting dependencies
2. **numpy <2:** Jetson PyTorch compiled against numpy 1.x
3. **transformers patch:** `enable_gqa` doesn't exist in Jetson's SDPA

## Verification

```bash
./benchmark.sh
```

Expected: RTF 1.477 (0.6B), RTF 1.219 (1.7B)

For full details and troubleshooting, see the complete guide in this file or INTEGRATION_SUMMARY.md.
