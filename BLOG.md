# Real-Time Qwen3-TTS: Unlocking 5x Speed on Consumer Hardware

**TL;DR:** Qwen3-TTS is an incredible open-source model, but running it at production speeds requires bypassing the Python overhead. By combining transformers' `StaticCache` with `torch.cuda.CUDAGraph`, we unlocked RTF 5.0 on an RTX 4090 and RTF 1.5 on a Jetson Orin — with streaming support — all in just 1,038 lines of pure PyTorch, with zero custom attention code.

## The Challenge: The "Reference Code" Gap

The Qwen team's technical report boasts an impressive "First-Packet Latency" of just 97ms. However, the inference code they released in their official repository is far from that.

The released code relies on a standard loop that prioritizes readability and compatibility over raw performance. On a Jetson AGX Orin, this reference implementation runs at **RTF 0.175**: 1 second of audio takes 5.7 seconds to generate. Time to first audio? **2.6 seconds.**

This isn't a flaw in the model itself — it's simply the difference between a research reference implementation and a production engine. We set out to bridge that gap and unlock the speed promised in the technical report.

## The Solution: CUDA Graphs

The bottleneck turned out to be **kernel launch overhead**. Each decode step runs ~500 small GPU operations. In a standard Python loop, the GPU spends more time waiting for the CPU's instructions than actually computing.

We solved this using PyTorch CUDA Graphs. This allows us to "record" the GPU operations once and replay them instantly, removing the Python overhead entirely.

## Results: Validating the "97ms" Promise

Our optimized implementation not only matched the Qwen team's latency claims but often exceeded them, proving how efficient this architecture truly is.

### 0.6B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.175 | 2,572ms | **1.38** | **555ms** | 7.9x / 4.6x |
| Jetson Thor | 0.803 | 862ms | 1.50 | 505ms | 1.9x / 1.7x |
| DGX Spark (GB10) | 1.19 | 631ms | 1.44 | 477ms | 1.2x / 1.3x |
| RTX 4090 | 1.34 | 462ms | **4.56** | **168ms** | 3.4x / 2.8x |
| H100 80GB HBM3 | 0.59 | 1,049ms | **3.47** | **231ms** | 5.9x / 4.5x |

### 1.7B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.130 | 2,594ms | **1.13** | **669ms** | 8.7x / 3.9x |
| Jetson Thor | 0.772 | 912ms | 1.26 | 595ms | 1.6x / 1.5x |
| DGX Spark (GB10) | 0.975 | 749ms | 1.16 | 561ms | 1.2x / 1.3x |
| RTX 4090 | 1.32 | 468ms | **4.06** | **186ms** | 3.1x / 2.5x |
| H100 80GB HBM3 | 0.59 | 1,045ms | **3.30** | **245ms** | 5.6x / 4.3x |

RTF > 1.0 = faster than real-time. TTFA = Time to First Audio, measured as time to first playable audio chunk via streaming (chunk_size=8, matching baseline's default `emit_every_frames=8`). Both include text tokenization for fair comparison. Speedup shows throughput / TTFA improvement.

**Two Stories, One Optimization:**

On **high-end GPUs (RTX 4090)**: Baseline already achieves RTF > 1.0, so CUDA graphs aren't about making it "real-time" — they're about **latency reduction and streaming**. Throughput improves 3.4x, and streaming enables real-time audio delivery.

On **edge devices (Jetson Orin)**: Baseline can't keep up (RTF 0.13–0.18). CUDA graphs deliver **7.9x–8.7x** throughput speedup, crossing the real-time threshold (RTF 1.13–1.38). Streaming TTFA drops from **2,572ms to 555ms** (4.6x).

**The 4090 Wins Single-Stream:** For batch=1 workloads, the RTX 4090 outperforms the H100. The H100's lower baseline (RTF 0.59 vs 4090's 1.34) reflects its design for batch processing. Even with CUDA graphs, the 4090's higher clocks (**2.5 GHz vs 1.8 GHz**) translate to better single-stream performance.

**Why the DGX Spark barely benefits (1.2x):** CUDA graphs eliminate kernel launch overhead — the CPU can't dispatch ~500 small kernels per step fast enough, so the GPU idles between them. The DGX Spark's Grace CPU (72 Neoverse V2 cores) is fast enough to keep up with its modest GB10 GPU, so there's little overhead to eliminate. Compare the Jetson Orin: its 12 Cortex-A78AE cores can't feed the GPU fast enough, yielding 7.9x from CUDA graphs. The H100 and 4090 have fast GPUs that outpace their CPUs, yielding 5–6x. The Spark is the most CPU/GPU-balanced system we tested — it reaches respectable absolute RTF (1.44) but gets there mostly without CUDA graphs' help.

## How We Did It (The "Magic")

We didn't rewrite the model in C++ or use a complex serving engine like vLLM. We kept it entirely within the PyTorch/Hugging Face ecosystem, using just **1,038 lines of Python** (including streaming), and we didn't reimplement a single attention layer.

The key insight: transformers already ships everything you need. Its `StaticCache` class pre-allocates fixed-size KV tensors and updates them in-place via `index_copy_` — exactly what CUDA graphs require. Instead of reimplementing 28 layers of attention, RoPE, and GQA by hand, we just call the model's own forward pass with a `StaticCache` and a `cache_position` buffer, then wrap the whole thing in `torch.cuda.CUDAGraph`.

1. **`StaticCache` from transformers**: Pre-allocated KV tensors with fixed shapes. The model's attention layers call `cache.update()` internally — no custom cache code needed.
2. **Model's own forward**: The model handles RoPE, causal masking, GQA, and layer norms. For single-token decode with `StaticCache`, all tensor shapes are fixed, making it fully CUDA-graph-compatible.
3. **Graph capture**: `torch.cuda.CUDAGraph` wraps the forward pass. Before each replay, we update the `cache_position` buffer — the model's mask and RoPE shift accordingly.

### Per-component breakdown (Jetson AGX Orin, 0.6B)

| Component | Before | After |
|---|---|---|
| Talker (28 layers) | 75ms | 12ms |
| Predictor (15 steps) | 190ms | 26ms |
| Overhead | 65ms | 16ms |
| **Total per step** | **330ms** | **54ms** |

This approach demonstrates the power of the PyTorch/transformers ecosystem: you don't need a custom inference engine or hand-rolled attention kernels. The building blocks — `StaticCache`, `cache_position`, `CUDAGraph` — are already there. You just need to connect them.

## Streaming Support

For real-time applications like voice assistants, waiting for full generation isn't an option. We added streaming output that yields audio chunks during generation — using the exact same CUDA graphs.

The streaming generator accumulates codec tokens in chunks (configurable size), decodes each chunk with left context from previous frames (matching the upstream codec's `chunked_decode` pattern), and yields playable audio. The CUDA graph replays are identical — only the control flow changes.

### Chunk size vs performance (Jetson AGX Orin, 0.6B)

| chunk_size | TTFA | Streaming RTF | Audio per chunk |
|---|---|---|---|
| 4 | 355ms | 1.11 | 333ms |
| 8 | 555ms | 1.22 | 667ms |
| 12 | 760ms | 1.26 | 1000ms |
| Non-streaming | — | 1.36 | all at once |

`chunk_size=4` is the smallest that stays real-time on Jetson. On faster GPUs, even `chunk_size=1` should remain above RTF 1.0.

## Code

We've open-sourced this implementation to help the community deploy Qwen3-TTS in production environments:

**[github.com/andimarafioti/qwen3-tts-cuda-graphs](https://github.com/andimarafioti/qwen3-tts-cuda-graphs)**

```bash
git clone https://github.com/andimarafioti/qwen3-tts-cuda-graphs
cd qwen3-tts-cuda-graphs
./setup.sh       # creates venv with uv, installs deps, downloads models
./benchmark.sh   # runs full benchmark, saves JSON + audio samples
```

Core implementation:
- `predictor_graph.py` (156 lines)
- `talker_graph.py` (137 lines)
- `generate.py` (156 lines) — non-streaming
- `streaming.py` (178 lines) — streaming
- `model.py` (404 lines) — wrapper API

No Flash Attention. No Triton. No vLLM. No custom attention code. Just the model's own forward pass, `StaticCache`, and `CUDAGraph`.

### What we tried first (and what didn't work)

Before CUDA graphs, we systematically tried everything else:

- **Attention backends** (eager, SDPA, Flash Attention 2): all identical RTF. Attention is not the bottleneck.
- **Custom CUDA kernels** (fused RMSNorm 8.4x faster, fused SiLU 2.2x): only 1.25x end-to-end. These ops are ~4% of compute.
- **torch.compile**: we patched three Triton incompatibilities to get it working on Jetson for the first time. Zero speedup — dynamic KV-cache shapes defeat the compiler.
- **Porting nano-qwen3tts-vllm** (7,289 lines): KV cache block allocator breaks on Jetson's unified memory.
- **Manual attention reimplementation** (previous version of this repo): 758 lines with hand-rolled RoPE, GQA, and KV cache. Worked, but unnecessary — `StaticCache` already does all of this inside the model's own forward pass.

## Conclusion

Qwen3-TTS is a beast of a model. By leveraging the `StaticCache` API already available in transformers and wrapping the model's own forward pass in CUDA graphs, we can reveal its true speed — without reimplementing a single layer. With streaming support, audio starts playing within 555ms on a Jetson (4.6x faster than baseline). Whether you are running on a $30,000 H100 or a $1,000 Jetson, this model is ready for real-time prime time.


---

*Model: Qwen3-TTS-12Hz (0.6B and 1.7B). Benchmarked on Jetson AGX Orin 64GB (JetPack 6, PyTorch 2.5.0a0), Jetson Thor (PyTorch 2.10.0+cu130), DGX Spark (GB10, PyTorch 2.11.0+cu130), RTX 4090 (PyTorch 2.10.0+cu128), and H100 80GB (PyTorch 2.10.0+cu128). NVIDIA provided the Jetson AGX Orin board used in this work.*
