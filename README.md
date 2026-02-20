# Qwen3-TTS CUDA Graphs

Real-time Qwen3-TTS inference using CUDA graph capture. No Flash Attention, no vLLM, no Triton. Just `torch.cuda.CUDAGraph`. **1,038 lines of Python.** Supports both streaming and non-streaming generation.

## Results

Benchmarks include tokenization + inference (apples-to-apples with baseline). RTF > 1.0 = faster than real-time. TTFA measured as time to first playable audio chunk using streaming (chunk_size=8, matching baseline's default `emit_every_frames=8`).

### 0.6B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.175 | 2,572ms | 1.38 | 555ms | 7.9x / 4.6x |
| Jetson Thor | 0.803 | 862ms | 1.50 | 505ms | 1.9x / 1.7x |
| DGX Spark (GB10) | 1.19 | 631ms | 1.48 | 421ms | 1.2x / 1.5x |
| RTX 4090 | 1.34 | 462ms | **4.56** | **168ms** | 3.4x / 2.8x |
| H100 80GB HBM3 | 0.59 | 1,049ms | **3.47** | **231ms** | 5.9x / 4.5x |

### 1.7B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.130 | 2,594ms | 1.13 | 669ms | 8.7x / 3.9x |
| Jetson Thor | 0.772 | 912ms | 1.26 | 595ms | 1.6x / 1.5x |
| DGX Spark (GB10) | 0.975 | 749ms | 1.14 | 586ms | 1.2x / 1.3x |
| RTX 4090 | 1.32 | 468ms | **4.06** | **186ms** | 3.1x / 2.5x |
| H100 80GB HBM3 | 0.59 | 1,045ms | **3.30** | **245ms** | 5.6x / 4.3x |

**Note:** Baseline uses standard qwen-tts with `stream_generate_voice_clone()` (default `emit_every_frames=8`). CUDA graphs uses `generate_voice_clone_streaming(chunk_size=8)` for TTFA. Both measure time to first playable audio chunk and include text tokenization for fair comparison. Speedup shows throughput / TTFA improvement.

**GPU architecture notes:** RTX 4090 (2.5 GHz clocks) outperforms H100 (1.8 GHz) for single-stream workloads. H100's lower baseline (RTF 0.59 vs 4090's 1.34) reflects design optimization for batch processing rather than single-stream inference.

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
2. **Model's own forward**: SDPA + RoPE via the model's native attention layers
3. **Graph capture**: `torch.cuda.CUDAGraph` for both predictor and talker
4. **Padded attention**: attention mask handles variable-length KV within fixed buffers

### Per-component breakdown (Jetson AGX Orin, 0.6B)

| Component | Before | After |
|---|---|---|
| Talker (28 layers) | 75ms | 12ms |
| Predictor (15 steps) | 190ms | 26ms |
| Overhead | 65ms | 16ms |
| **Total per step** | **330ms** | **54ms** |

## Streaming

CUDA graphs support streaming output — audio chunks are yielded during generation with the same per-step performance as non-streaming mode.

### Chunk size vs performance (Jetson AGX Orin, 0.6B)

| chunk_size | TTFA | RTF | Audio per chunk |
|---|---|---|---|
| 4 | 355ms | 1.11 | 333ms |
| 8 | 555ms | 1.22 | 667ms |
| 12 | 760ms | 1.26 | 1000ms |
| Non-streaming | — | 1.36 | all at once |

Smaller chunks = lower latency but more decode overhead. `chunk_size=4` is the smallest that stays real-time on Jetson.

### Usage

```python
from qwen3_tts_cuda_graphs import Qwen3TTSCudaGraphs

model = Qwen3TTSCudaGraphs.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")

# Streaming — yields audio chunks during generation
for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
    text="Hello world!", language="English",
    ref_audio="ref.wav", ref_text="...",
    chunk_size=8,  # 8 steps ≈ 667ms of audio per chunk
):
    play(audio_chunk, sr)  # process/send each chunk immediately

# Non-streaming — returns all audio at once (unchanged API)
audio_list, sr = model.generate_voice_clone(
    text="Hello world!", language="English",
    ref_audio="ref.wav", ref_text="...",
)
```

### How it works

The CUDA graphs are unchanged — both predictor and talker graphs are replayed per step. The streaming generator yields codec ID chunks every `chunk_size` steps, and the model wrapper decodes each chunk to audio using a sliding window with 25-frame left context (matching the upstream codec's `chunked_decode` pattern) to avoid boundary artifacts.

## Voice Cloning with Precomputed Speaker Embeddings

For production use, extract the speaker embedding once and reuse it:

```bash
# 1. Extract speaker embedding from reference audio (one-time, ~10s)
python examples/extract_speaker.py --ref_audio voice.wav --output speaker.pt

# 2. Generate speech with CUDA graphs (real-time)
python examples/generate_with_embedding.py --speaker speaker.pt --text "Hello!" --language English --output en.wav
python examples/generate_with_embedding.py --speaker speaker.pt --text "Bonjour!" --language French --output fr.wav
python examples/generate_with_embedding.py --speaker speaker.pt --text "Hallo!" --language German --output de.wav
```

The speaker embedding is a 4KB file (2048-dim bf16 vector). In `x_vector_only` mode:
- **No accent bleed**: native pronunciation per language
- **Shorter prefill**: 10 tokens vs ~80+ in full ICL clone mode
- **No ref audio at runtime**: just the 4KB embedding file

## Comparison with Other Approaches

| | nano-qwen3tts-vllm | Qwen3-TTS-streaming | **Ours** |
|---|---|---|---|
| Lines of code | 7,289 | ~3,000 | **1,038** |
| Flash Attention required | Yes | No | **No** |
| Triton/torch.compile required | No | Yes | **No** |
| Streaming support | No | Yes | **Yes** |
| Runs on Jetson | No | No | **Yes** |
| RTF on H100 (1.7B) | 0.399 | N/A | **3.80** |

On the same H100 hardware: **~10x faster with ~7x less code** vs nano-qwen3tts-vllm.

## Files

```
qwen3_tts_cuda_graphs/
  model.py                        # Wrapper API (404 lines)
  generate.py                     # Non-streaming generation loop (156 lines)
  streaming.py                    # Streaming generation loop (178 lines)
  predictor_graph.py              # Predictor CUDA graph with StaticCache (156 lines)
  talker_graph.py                 # Talker CUDA graph with StaticCache (137 lines)
examples/
  extract_speaker.py              # Extract speaker embedding from ref audio
  generate_with_embedding.py      # Generate with precomputed speaker embedding
benchmarks/
  throughput.py                   # Throughput benchmark (RTF + audio samples)
  streaming.py                    # Streaming benchmark (TTFA + chunk timing)
  chunk_sweep.py                  # Chunk size sweep (RTF vs latency tradeoff)
  baseline.py                     # Baseline qwen-tts benchmark
benchmark.sh                      # Run benchmarks
setup.sh                          # Setup venv + download models
```

Core implementation: **1,038 lines** of Python.

## License

MIT

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team
- [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) for inspiration on CUDA graph usage
- NVIDIA for providing the Jetson AGX Orin board
