# Qwen3-TTS CUDA Graphs

Real-time Qwen3-TTS inference using manual CUDA graph capture. No Flash Attention, no vLLM, no Triton. Just `torch.cuda.CUDAGraph`. **758 lines of Python.**

## Results

Benchmarks include tokenization + inference (apples-to-apples with baseline). RTF > 1.0 = faster than real-time.

### 0.6B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.175 | 2,572ms | 1.38 | 216ms | 7.9x |
| DGX Spark (GB10) | 1.19 | 631ms | 1.44 | 113ms | 1.2x / 5.6x |
| RTX 4090 | 1.34 | 462ms | **4.56** | **55ms** | 3.4x / 8.4x |
| H100 80GB HBM3 | 0.59 | 1,049ms | **3.47** | **100ms** | 5.9x / 10.5x |

### 1.7B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.130 | 2,594ms | 1.13 | 237ms | 8.7x |
| DGX Spark (GB10) | 0.975 | 749ms | 1.16 | 196ms | 1.2x / 3.8x |
| RTX 4090 | 1.32 | 468ms | **4.06** | **58ms** | 3.1x / 8.1x |
| H100 80GB HBM3 | 0.59 | 1,045ms | **3.30** | **104ms** | 5.6x / 10.0x |

**Note:** Baseline uses standard qwen-tts. CUDA graphs uses `Qwen3TTSCudaGraphs` wrapper with voice prompt caching. Both include text tokenization overhead for fair comparison. Speedup shows throughput improvement / TTFA improvement (e.g., "3.4x / 8.4x" = 3.4x faster generation, 8.4x lower latency).

**⚠️ H100 baseline anomaly:** Baseline RTF 0.59 is unexpectedly slow (slower than 4090, GB10, even Orin). May be hitting a generation bug or config issue. CUDA graphs results are solid (RTF 3.47/3.30, sub-105ms latency) but 4090 still wins on single-stream throughput due to higher clocks.

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

CUDA graphs capture the entire decode step and replay it as a single GPU operation:

1. **Static KV cache**: pre-allocated fixed-size tensors (no dynamic allocation)
2. **Manual attention**: direct SDPA + RoPE, bypassing HF's DynamicCache
3. **Graph capture**: `torch.cuda.CUDAGraph` for both predictor and talker
4. **Padded attention**: attention mask handles variable-length KV within fixed buffers

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
| Lines of code | 7,289 | ~3,000 | **758** |
| Flash Attention required | Yes | No | **No** |
| Triton/torch.compile required | No | Yes | **No** |
| Runs on Jetson | No | No | **Yes** |
| RTF on H100 (1.7B) | 0.399 | N/A | **3.80** |
| TTFA | 160ms (L4) | N/A | **36ms (4090)** |

On the same H100 hardware: **~10x faster with ~10x less code** vs nano-qwen3tts-vllm.

## Files

```
manual_cudagraph_predictor.py   # Predictor graph (261 lines)
manual_cudagraph_talker.py      # Talker graph (341 lines)
fast_generate_v5.py             # Full generation loop (156 lines)
extract_speaker.py              # Extract speaker embedding from ref audio
generate_xvec.py                # End-to-end generation with precomputed speaker
bench_v5.py                     # Benchmark (throughput + TTFA + audio samples)
bench_ttft.py                   # Detailed TTFA breakdown benchmark
benchmark.sh                    # Run benchmarks
setup.sh                        # Setup venv + download models
```

Core implementation: **758 lines** of Python.

## License

MIT

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team
- [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) for inspiration on CUDA graph usage
- NVIDIA for providing the Jetson AGX Orin board
