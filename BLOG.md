# Real-Time Qwen3-TTS: Unlocking 5x Speed on Consumer Hardware

**TL;DR:** Qwen3-TTS is an incredible open-source model, but running it at production speeds requires bypassing the Python overhead. By combining transformers' `StaticCache` with `torch.cuda.CUDAGraph`, we unlocked RTF 5.0 on an RTX 4090 and RTF 1.5 on a Jetson Orin — all in just 449 lines of pure PyTorch, with zero custom attention code.

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

| GPU | Baseline RTF | CUDA Graphs RTF | Speedup | TTFA |
|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.175 | **1.55** | 8.8x | 77ms |
| DGX Spark (GB10) | — | **1.52** | — | 88ms |
| RTX 4090 | — | **5.06** | — | **36ms** |
| H100 80GB HBM3 | — | **3.92** | — | 63ms |

### 1.7B Model

| GPU | Baseline RTF | CUDA Graphs RTF | Speedup | TTFA |
|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.130 | **1.24** | 9.5x | 77ms |
| DGX Spark (GB10) | — | **1.35** | — | 142ms |
| RTX 4090 | — | **4.46** | — | **39ms** |
| H100 80GB HBM3 | — | **3.80** | — | 64ms |

RTF > 1.0 = faster than real-time. TTFA = Time to First Audio, measured as time to first playable audio chunk.

**Verified Latency:** On the RTX 4090, we achieved **36ms** latency — well under the 97ms benchmark from the tech report. Even the Jetson Orin hit 77ms, making it viable for real-time edge voice interaction.

**The 4090 Surprise:** For single-user (batch=1) workloads, the RTX 4090 actually outperformed the H100. At batch=1, each operation's matrices are too small to saturate the H100's massive compute and memory bandwidth. In this latency-bound regime, the 4090's ~38% higher clock speed (2.5 GHz vs 1.8 GHz) translates directly into faster per-step execution.

**Edge Efficiency:** The Jetson delivers RTF 1.2–1.5 at 60W while the H100 delivers RTF 3.8–3.9 at 700W. That's ~2.5x more RTF per watt on the edge device, which matters for always-on applications like robotics or embedded assistants.

## How We Did It (The "Magic")

We didn't rewrite the model in C++ or use a complex serving engine like vLLM. We kept it entirely within the PyTorch/Hugging Face ecosystem, using just **449 lines of Python**, and we didn't reimplement a single attention layer.

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
- `manual_cudagraph_predictor.py` (156 lines)
- `manual_cudagraph_talker.py` (137 lines)
- `fast_generate_v5.py` (156 lines)

No Flash Attention. No Triton. No vLLM. No custom attention code. Just the model's own forward pass, `StaticCache`, and `CUDAGraph`.

### What we tried first (and what didn't work)

Before CUDA graphs, we systematically tried everything else:

- **Attention backends** (eager, SDPA, Flash Attention 2): all identical RTF. Attention is not the bottleneck.
- **Custom CUDA kernels** (fused RMSNorm 8.4x faster, fused SiLU 2.2x): only 1.25x end-to-end. These ops are ~4% of compute.
- **torch.compile**: we patched three Triton incompatibilities to get it working on Jetson for the first time. Zero speedup — dynamic KV-cache shapes defeat the compiler.
- **Porting nano-qwen3tts-vllm** (7,289 lines): KV cache block allocator breaks on Jetson's unified memory.
- **Manual attention reimplementation** (previous version of this repo): 758 lines with hand-rolled RoPE, GQA, and KV cache. Worked, but unnecessary — `StaticCache` already does all of this inside the model's own forward pass.

## Conclusion

Qwen3-TTS is a beast of a model. By leveraging the `StaticCache` API already available in transformers and wrapping the model's own forward pass in CUDA graphs, we can reveal its true speed — without reimplementing a single layer. Whether you are running on a $30,000 H100 or a $1,000 Jetson, this model is ready for real-time prime time.


---

*Model: Qwen3-TTS-12Hz (0.6B and 1.7B). Benchmarked on Jetson AGX Orin 64GB (JetPack 6, PyTorch 2.5.0a0), DGX Spark (GB10, PyTorch 2.11.0+cu130), RTX 4090 (PyTorch 2.10.0+cu128), and H100 80GB (PyTorch 2.10.0+cu128). NVIDIA provided the Jetson AGX Orin board used in this work.*
