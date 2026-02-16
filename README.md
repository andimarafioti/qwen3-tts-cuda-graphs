# Qwen3-TTS CUDA Graphs

Real-time Qwen3-TTS inference using transformers' `StaticCache` and `torch.cuda.CUDAGraph`. No Flash Attention, no vLLM, no Triton — just the model's own forward pass with the right cache. **449 lines of Python.**

## Results

Same code, four GPUs. RTF > 1.0 = faster than real-time.

### 0.6B Model

| GPU | ms/step | RTF | TTFA | TDP |
|---|---|---|---|---|
| Baseline (HF streaming) | ~330 | 0.175 | 2,572ms | — |
| Jetson AGX Orin 64GB | 54 | 1.55 | 77ms | 60W |
| DGX Spark (GB10) | 55 | 1.52 | 88ms | 100W |
| RTX 4090 | 16 | 5.06 | 36ms | 450W |
| H100 80GB HBM3 | 21 | 3.92 | 63ms | 700W |

### 1.7B Model

| GPU | ms/step | RTF | TTFA | TDP |
|---|---|---|---|---|
| Baseline (HF streaming) | ~450 | 0.130 | 2,594ms | — |
| Jetson AGX Orin 64GB | 66 | 1.24 | 77ms | 60W |
| DGX Spark (GB10) | 62 | 1.35 | 142ms | 100W |
| RTX 4090 | 19 | 4.46 | 39ms | 450W |
| H100 80GB HBM3 | 22 | 3.80 | 64ms | 700W |

The RTX 4090 beats the H100 for single-stream TTS latency. For batch=1 workloads, kernel launch overhead matters more than raw memory bandwidth.

## Quick Start

```bash
git clone https://github.com/andimarafioti/qwen3-tts-cuda-graphs
cd qwen3-tts-cuda-graphs
./setup.sh       # creates venv with uv, installs deps, downloads models
./benchmark.sh   # runs full benchmark, saves JSON + audio samples
```

Requires: Python 3.10+, NVIDIA GPU with CUDA, [uv](https://docs.astral.sh/uv/).

### Benchmark a specific model

```bash
./benchmark.sh 0.6B
./benchmark.sh 1.7B
./benchmark.sh both   # default
```

Results are saved as `bench_results_<GPU_NAME>.json` and audio samples as `sample_0.6B.wav` / `sample_1.7B.wav`.

## How It Works

Qwen3-TTS runs two autoregressive transformers per decode step:
1. **Talker** (28 layers): generates the first codebook token from text
2. **Code Predictor** (5 layers): generates 15 additional codebook tokens

A single step involves ~500 small CUDA kernel launches with Python overhead between them. The GPU spends more time waiting for the next kernel than computing.

CUDA graphs capture the entire decode step and replay it as a single GPU operation. The key insight: transformers already ships a `StaticCache` class designed for exactly this. It pre-allocates fixed-size KV tensors and uses `index_copy_` for in-place updates — no dynamic allocation, fully compatible with CUDA graph capture.

1. **`StaticCache` from transformers**: pre-allocated fixed-size KV tensors, no custom cache code needed
2. **Model's own forward pass**: no manual attention reimplementation — the model handles RoPE, masking, and GQA internally
3. **Graph capture**: `torch.cuda.CUDAGraph` wraps the model's forward for both predictor and talker
4. **`cache_position` buffer**: updated before each graph replay to shift the causal mask and RoPE

### Per-component breakdown (Jetson AGX Orin, 0.6B)

| Component | Before | After |
|---|---|---|
| Talker (28 layers) | 75ms | 12ms |
| Predictor (15 steps) | 190ms | 26ms |
| Overhead | 65ms | 16ms |
| **Total per step** | **330ms** | **54ms** |

## Voice Cloning with Precomputed Speaker Embeddings

For production use, extract the speaker embedding once and reuse it:

```bash
# 1. Extract speaker embedding from reference audio (one-time, ~10s)
python extract_speaker.py --ref_audio voice.wav --output speaker.pt

# 2. Generate speech with CUDA graphs (real-time)
python generate_xvec.py --speaker speaker.pt --text "Hello!" --language English --output en.wav
python generate_xvec.py --speaker speaker.pt --text "Bonjour!" --language French --output fr.wav
python generate_xvec.py --speaker speaker.pt --text "Hallo!" --language German --output de.wav
```

The speaker embedding is a 4KB file (2048-dim bf16 vector). In `x_vector_only` mode:
- **No accent bleed**: native pronunciation per language
- **Shorter prefill**: 10 tokens vs ~80+ in full ICL clone mode
- **No ref audio at runtime**: just the 4KB embedding file

## Comparison with Other Approaches

| | nano-qwen3tts-vllm | Qwen3-TTS-streaming | **Ours** |
|---|---|---|---|
| Lines of code | 7,289 | ~3,000 | **449** |
| Flash Attention required | Yes | No | **No** |
| Triton/torch.compile required | No | Yes | **No** |
| Runs on Jetson | No | No | **Yes** |
| RTF on H100 (1.7B) | 0.399 | N/A | **3.80** |
| TTFA | 160ms (L4) | N/A | **36ms (4090)** |

On the same H100 hardware: **~10x faster with ~16x less code** vs nano-qwen3tts-vllm.

## Files

```
manual_cudagraph_predictor.py   # Predictor graph with StaticCache (156 lines)
manual_cudagraph_talker.py      # Talker graph with StaticCache (137 lines)
fast_generate_v5.py             # Full generation loop (156 lines)
extract_speaker.py              # Extract speaker embedding from ref audio
generate_xvec.py                # End-to-end generation with precomputed speaker
bench_v5.py                     # Benchmark (throughput + TTFA + audio samples)
bench_ttft.py                   # Detailed TTFA breakdown benchmark
benchmark.sh                    # Run benchmarks
setup.sh                        # Setup venv + download models
```

Core implementation: **449 lines** of Python.

## License

MIT

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team
- [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) for inspiration on CUDA graph usage
- NVIDIA for providing the Jetson AGX Orin board
