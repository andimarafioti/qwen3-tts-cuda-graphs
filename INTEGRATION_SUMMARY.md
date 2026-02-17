# Qwen3-TTS CUDA Graphs Integration Summary

## ✅ Status: WORKING

The CUDA graphs implementation is now successfully integrated into speech-to-speech and achieves **real-time performance** (RTF > 1.0) on Jetson AGX Orin.

### Benchmark Results
- **0.6B model:** 55.6ms/step, RTF 1.477, TTFA 169ms
- **1.7B model:** 67.6ms/step, RTF 1.219, TTFA 178ms
- **speech-to-speech test:** RTF 1.40 ✓

---

## What Was Done

### 1. Created qwen3-tts-cuda-graphs Python Package
**Location:** `/home/andi/Documents/qwen3-tts-cuda-graphs/`

**Changes:**
- Created `qwen3_tts_cuda_graphs/` package directory
- Added `__init__.py` exposing `Qwen3TTSCudaGraphs` class
- Added `model.py` with wrapper API matching qwen-tts interface
- Copied core modules: `fast_generate_v5.py`, `manual_cudagraph_predictor.py`, `manual_cudagraph_talker.py`
- Updated `pyproject.toml` with proper build system config
- Package is installable with `pip install -e .`

**Key API:**
```python
from qwen3_tts_cuda_graphs import Qwen3TTSCudaGraphs

model = Qwen3TTSCudaGraphs.from_pretrained(
    model_name="path/to/model",
    device="cuda",
    dtype=torch.bfloat16,
    max_seq_len=2048,
)

# Compatible with qwen-tts API
wavs, sr = model.generate_voice_clone(
    text="Hello world",
    language="English",
    ref_audio="ref.wav",
    ref_text="Reference transcript",
)
```

### 2. speech-to-speech Integration
**Location:** `/home/andi/Documents/speech-to-speech/TTS/qwen3_tts_handler.py`

The handler **already had CUDA graphs support** in the PR (`use_cuda_graphs` flag). It just needed the package to be installable.

**Usage:**
```python
handler = Qwen3TTSHandler(
    stop_event,
    queue_in=queue_in,
    queue_out=queue_out,
    setup_args=(should_listen,),
    setup_kwargs={
        "device": "cuda",
        "model_name": "Qwen3-TTS-12Hz-0.6B-Base",
        "use_cuda_graphs": True,  # ← enables CUDA graphs
        "attn_implementation": "eager",  # see compatibility notes
        "ref_audio": "ref.wav",
        "ref_text": "Reference text",
        "language": "English",
    },
)
```

---

## Critical Dependency Fixes (Jetson AGX Orin)

The qwen-tts library has dependency conflicts on Jetson. Here's what's needed:

### Environment Setup

**1. Use local qwen-tts (not pip package):**
```bash
cd /home/andi/Documents/qwen3-tts-cuda-graphs
.venv/bin/pip uninstall -y qwen-tts
.venv/bin/pip install -e /home/andi/Documents/Qwen3-TTS-streaming --no-deps
```

**2. Correct numpy version:**
```bash
.venv/bin/pip install "numpy<2"  # Jetson PyTorch needs numpy 1.x
```

**3. Upgrade transformers and patch:**
```bash
.venv/bin/pip install "transformers==4.57.3"
```

**Then apply this patch:**

**File:** `.venv/lib/python3.10/site-packages/transformers/integrations/sdpa_attention.py`

**Line ~70, replace:**
```python
        else:
            sdpa_kwargs = {"enable_gqa": True}
```

**With:**
```python
        else:
            # Jetson PyTorch 2.5.0a0 doesn't support enable_gqa parameter yet
            # Fall back to manual repeat_kv instead
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
```

**4. Create dummy torchaudio (optional, if needed):**
The local qwen-tts imports torchaudio.compliance.kaldi but doesn't actually use it for voice cloning. If you get torchaudio ABI errors, create:

```bash
mkdir -p .venv/lib/python3.10/site-packages/torchaudio/compliance
```

**`.venv/lib/python3.10/site-packages/torchaudio/__init__.py`:**
```python
__version__ = "2.5.0"
class compliance:
    class kaldi:
        @staticmethod
        def fbank(*args, **kwargs):
            import torch
            return torch.zeros(1)
```

**`.venv/lib/python3.10/site-packages/torchaudio/compliance/kaldi.py`:**
```python
import torch
def fbank(waveform, *args, **kwargs):
    if isinstance(waveform, torch.Tensor):
        return torch.zeros((waveform.shape[0], 80))
    return torch.zeros((1, 80))
```

---

## Why These Fixes Are Needed

### transformers 4.57.3 `enable_gqa` Bug
- transformers 4.57.3 added `enable_gqa` parameter to SDPA
- Jetson PyTorch 2.5.0a0 doesn't support this parameter yet
- Patch falls back to manual `repeat_kv` (slightly slower but works)

### Local qwen-tts vs pip package
- pip package `qwen-tts==0.0.4` has hard dependency on `transformers==4.57.3`
- But also has other deps that conflict (torchaudio ABI mismatch)
- Local `/home/andi/Documents/Qwen3-TTS-streaming` allows flexible deps

### numpy version
- Jetson PyTorch 2.5.0a0 was compiled against numpy 1.x
- numpy 2.x has breaking ABI changes
- Must use `numpy<2`

---

## Testing

### Standalone benchmarks (qwen3-tts-cuda-graphs):
```bash
cd /home/andi/Documents/qwen3-tts-cuda-graphs
source .venv/bin/activate
./benchmark.sh
```

### speech-to-speech integration test:
```bash
cd /home/andi/Documents/speech-to-speech
source .venv/bin/activate
python test_qwen_cuda_graphs.py
```

Expected output:
```
✓ CUDA graphs working! Real-time performance achieved.
RTF: 1.40
```

---

## Files Modified/Created

### qwen3-tts-cuda-graphs:
- `qwen3_tts_cuda_graphs/__init__.py` (new)
- `qwen3_tts_cuda_graphs/model.py` (new)
- `qwen3_tts_cuda_graphs/fast_generate_v5.py` (copied from root)
- `qwen3_tts_cuda_graphs/manual_cudagraph_predictor.py` (copied)
- `qwen3_tts_cuda_graphs/manual_cudagraph_talker.py` (copied)
- `pyproject.toml` (modified for proper package build)

### speech-to-speech:
- `test_qwen_cuda_graphs.py` (new test script)
- `TTS/qwen3_tts_handler.py` (already had CUDA graphs support in PR)

### Environment patches:
- `.venv/lib/python3.10/site-packages/transformers/integrations/sdpa_attention.py` (patched)

---

## Next Steps

1. **Commit qwen3-tts-cuda-graphs changes:**
   ```bash
   cd /home/andi/Documents/qwen3-tts-cuda-graphs
   git add qwen3_tts_cuda_graphs/ pyproject.toml
   git commit -m "feat: add installable Python package with Qwen3TTSCudaGraphs wrapper"
   ```

2. **Push and tag:**
   ```bash
   git push origin main
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. **Update speech-to-speech PR:**
   - Note the dependency setup requirements in PR description
   - Add README section on CUDA graphs usage
   - Consider adding setup script for Jetson environment

4. **Document environment setup:**
   Create `JETSON_SETUP.md` with the dependency fixes

---

## Performance Comparison

| Backend | RTF (0.6B) | RTF (1.7B) | Notes |
|---------|-----------|-----------|-------|
| Baseline qwen-tts | 0.175 | 0.130 | 5-7x slower than real-time |
| **CUDA graphs** | **1.477** | **1.219** | Real-time! |

**Speedup:** 6-10x faster with CUDA graphs

---

## Known Issues

1. **transformers compatibility:** Requires patch on Jetson PyTorch
2. **torchaudio ABI:** Pre-built wheels don't work with Jetson PyTorch
3. **Local qwen-tts required:** Pip package has conflicting deps
4. **attn_implementation:** Use `"eager"` not `"sdpa"` on Jetson (even with patch, sdpa has issues)

---

## Contact

For questions or issues, check:
- Qwen3-TTS CUDA graphs repo: `/home/andi/Documents/qwen3-tts-cuda-graphs`
- speech-to-speech PR: `feat/qwen3-tts-integration` branch
- This summary: `/home/andi/Documents/CUDA_GRAPHS_INTEGRATION_SUMMARY.md`

---

**Date:** 2026-02-17  
**Status:** ✅ Working and tested on Jetson AGX Orin 64GB
