# Qwen3-TTS: 5.6x Real-Time on an RTX 4090

**TL;DR:** Qwen3-TTS is an incredible open-source model, but running it at production speeds requires bypassing the Python overhead. By combining transformers' `StaticCache` with `torch.cuda.CUDAGraph`, we unlocked RTF 5.6 on an RTX 4090 and RTF 4.2 on an H100 — with streaming support — all in just 1,038 lines of pure PyTorch, with zero custom attention code.

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
| Jetson AGX Orin 64GB | 0.175 | 2,572ms | **1.57** | **556ms** | 9.0x / 4.6x |
| Jetson Thor | 0.803 | 862ms | 1.50 | 505ms | 1.9x / 1.7x |
| DGX Spark (GB10) | 1.19 | 631ms | 2.26 | 364ms | 1.9x / 1.7x |
| RTX 4090 | 1.34 | 462ms | **5.56** | **152ms** | 4.1x / 3.0x |
| H100 80GB HBM3 | 0.59 | 1,049ms | **4.19** | **224ms** | 7.1x / 4.7x |

### 1.7B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.130 | 2,594ms | **1.27** | **650ms** | 9.8x / 4.0x |
| Jetson Thor | 0.772 | 912ms | 1.26 | 595ms | 1.6x / 1.5x |
| DGX Spark (GB10) | 0.975 | 749ms | 1.66 | 464ms | 1.7x / 1.6x |
| RTX 4090 | 1.32 | 468ms | **4.85** | **170ms** | 3.7x / 2.8x |
| H100 80GB HBM3 | 0.59 | 1,045ms | **3.98** | **236ms** | 6.7x / 4.4x |

RTF > 1.0 = faster than real-time. TTFA = Time to First Audio, measured as time to first playable audio chunk via streaming (chunk_size=8). Baseline TTFA values come from the community `Qwen3-TTS-streaming` fork (which adds streaming). The official `Qwen3-TTS` repo does **not** currently support streaming, so its “TTFA” is effectively **time-to-full-audio** — with RTF near 1.0 you must wait for the whole sentence/paragraph to finish before you hear anything. Both include text tokenization for fair comparison. Speedup shows throughput / TTFA improvement. The streaming fork reports additional speedups that appear tied to `torch.compile`; we couldn’t reproduce those on Jetson-class devices where `torch.compile` isn’t available.

**For production deployment:** On the **RTX 4090** — a standard consumer GPU — throughput reaches **RTF 5.56** with streaming TTFA of just **152ms** (4.1x / 3.0x over baseline). On the **H100**, the **7.1x throughput speedup** is the largest we measured, reaching **RTF 4.19** — ready for large-scale serving. The 4090 outperforms the H100 in absolute single-stream RTF thanks to its higher boost clocks (**2.5 GHz vs 1.8 GHz**); the H100's strength lies in batch processing, not single-stream inference.

**For embedded and robotics:** The **Jetson AGX Orin** sees the most dramatic transformation — from RTF 0.175 (1 second of audio takes 5.7 seconds) to **RTF 1.57** (9.0x). Streaming TTFA drops from **2.6 seconds to 556ms**, making on-device voice synthesis viable where cloud latency isn't an option.

**Why do speedups range from 1.2x to 8.7x?** CUDA graphs eliminate kernel dispatch overhead: each decode step launches ~500 small GPU operations, and in a standard Python loop the GPU idles between them while the CPU prepares each launch. The speedup scales with the CPU/GPU imbalance. When the GPU finishes kernels faster than the CPU can dispatch them — whether because the CPU is slow (Jetson Orin's 12 Cortex-A78AE cores) or because the GPU is fast (4090, H100) — there's idle time to recover, and CUDA graphs recover it. This is the common case: most real-world GPU setups have this imbalance, yielding **3–9x improvements**.

The two exceptions in our benchmarks are NVIDIA's **Jetson Thor** and **DGX Spark**, both of which pair unusually powerful CPUs with more moderate GPUs. Thor's next-gen CPU pushes baseline to RTF 0.80 (4.5x faster than Orin without any optimization), and the Spark's 72-core Grace CPU reaches baseline RTF 1.19 — already real-time. With less dispatch overhead to eliminate, CUDA graphs add a modest 1.2–1.9x. The Spark is a particularly clean demonstration of the mechanism: its Grace CPU dispatches kernels efficiently enough that baseline Python overhead is minimal, confirming that CUDA graphs specifically target the CPU-to-GPU dispatch gap. These are purpose-built machines with exceptional CPU/GPU balance — on standard hardware, expect the larger speedups.

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

## A Small but High-Leverage Optimization

While profiling, we found a surprising hotspot: the **repetition penalty** logic in the decode loop. It was a small Python loop that indexed GPU tensors per token, which triggered CPU↔GPU syncs and cost a few milliseconds per step. On fast GPUs, that overhead becomes a meaningful fraction of the total decode time.

The fix was simple: **vectorize the repetition penalty**. Instead of looping token-by-token, we gather the unique tokens and apply the penalty in one fused GPU op via `torch.where`. This doesn’t change model behavior, but it removes the Python overhead and measurably improves throughput on every GPU we tested.

**Before (per-token Python loop):**

```python
if repetition_penalty != 1.0 and len(all_codec_ids) > 0:
    n_recent = min(50, len(all_codec_ids))
    recent = torch.stack([c[0] for c in all_codec_ids[-n_recent:]])
    for prev_tok in recent.unique():
        s = logits[0, 0, prev_tok]
        logits[0, 0, prev_tok] = s / repetition_penalty if s > 0 else s * repetition_penalty
```

**After (vectorized):**

```python
if repetition_penalty != 1.0 and len(all_codec_ids) > 0:
    n_recent = min(50, len(all_codec_ids))
    recent = torch.stack([c[0] for c in all_codec_ids[-n_recent:]])
    unique_toks = recent.unique()
    tok_logits = logits[0, 0, unique_toks]
    logits[0, 0, unique_toks] = torch.where(
        tok_logits > 0, tok_logits / repetition_penalty, tok_logits * repetition_penalty
    )
```

## Streaming Support

For real-time applications like voice assistants, waiting for full generation isn't an option. We added streaming output that yields audio chunks during generation — using the exact same CUDA graphs.

The streaming generator accumulates codec tokens in chunks (configurable size), decodes each chunk with left context from previous frames (matching the upstream codec's `chunked_decode` pattern), and yields playable audio. The CUDA graph replays are identical — only the control flow changes.

### Chunk size vs performance (Jetson AGX Orin, 0.6B)

| chunk_size | TTFA | Streaming RTF | Audio per chunk |
|---|---|---|---|
| 1 | 240ms | 0.750 | 83ms |
| 2 | 266ms | 1.042 | 167ms |
| 4 | 362ms | 1.251 | 333ms |
| 8 | 556ms | 1.384 | 667ms |
| 12 | 753ms | 1.449 | 1000ms |
| Non-streaming | — | 1.36 | all at once |

`chunk_size=2` is the smallest that stays real-time on Jetson. On faster GPUs, even `chunk_size=1` should remain above RTF 1.0.

## An Unexpected Discovery: The ICL Phoneme Artifact

While testing voice cloning, we noticed something odd: every generated sample started with a brief, consistent sound that wasn't in the target text — something like a "thumbs" or a "comes" sound depending on the reference audio used. It appeared reliably at the very start of every generation, regardless of what we asked the model to say.

Tracking it down required reading through the upstream `qwen_tts` library's model code to understand exactly what the prefill sequence looks like.

Qwen3-TTS voice cloning operates in **ICL (In-Context Learning) mode**: rather than extracting a static speaker embedding, it feeds the reference audio's raw codec tokens directly into the transformer's context. This gives the model a rich, frame-level representation of the target voice. The prefill sequence looks roughly like this:

```
[text role tokens] [speaker embedding] [codec BOS]
[ref_text_tok₀ + ref_code₀] [ref_text_tok₁ + ref_code₁] ... [ref_text_tokₙ + ref_codeₙ]
                                                               ↑
                                              last position before generation starts
```

Text and codec embeddings are **summed by position** across the reference audio's length. The last position in the prefill is the last codec token of the reference audio — and the model's first generated token is predicted from exactly there.

The consequence: whatever phoneme the reference audio ends on directly conditions the model's first output token. If the reference ends on a consonant cluster like the "mz" in "thumbs", the model treats that as the acoustic context it's continuing from, and generates a token that completes or continues that phoneme before transitioning to the target text. The effect is small but clearly audible — especially noticeable in conversation applications where clean turn boundaries matter.

The fix is a single operation: **append 0.5 seconds of silence to the reference audio before encoding it**. When the last codec tokens represent silence, the model's starting context is acoustic silence, and it generates the target speech cleanly from the first frame.

```python
audio, sr = sf.read(ref_audio_path, dtype="float32", always_2d=False)
silence = np.zeros(int(0.5 * sr), dtype=np.float32)
ref_audio_input = (np.concatenate([audio, silence]), sr)
```

This is now applied automatically in `_prepare_generation()` before the reference audio is passed to `create_voice_clone_prompt()`, so it works transparently regardless of what reference recording the user provides.

## Code

We've open-sourced this implementation to help the community deploy Qwen3-TTS in production environments:

**[github.com/andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)**

```bash
git clone https://github.com/andimarafioti/faster-qwen3-tts
cd faster-qwen3-tts
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

Qwen3-TTS is a beast of a model. By leveraging the `StaticCache` API already available in transformers and wrapping the model's own forward pass in CUDA graphs, we can reveal its true speed — without reimplementing a single layer. On an RTX 4090, audio generates at 5.6x real-time with 152ms time-to-first-audio. On a Jetson Orin, streaming TTFA drops from 2.6 seconds to 556ms. Whether you're serving from an H100 or running on-device on a Jetson, this model is ready for real-time prime time.


---

*Model: Qwen3-TTS-12Hz (0.6B and 1.7B). Benchmarked on Jetson AGX Orin 64GB (JetPack 6, PyTorch 2.5.0a0), Jetson Thor (PyTorch 2.10.0+cu130), DGX Spark (GB10, PyTorch 2.11.0+cu130), RTX 4090 (PyTorch 2.10.0+cu128), and H100 80GB (PyTorch 2.10.0+cu128). NVIDIA provided the Jetson AGX Orin board used in this work.*
