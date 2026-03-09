# Faster Qwen3-TTS

Real-time Qwen3-TTS inference using CUDA graph capture. No Flash Attention, no vLLM, no Triton. Just `torch.cuda.CUDAGraph`. Supports both streaming and non-streaming generation.

## Install

Requires: Python 3.10+, PyTorch 2.5.1+, NVIDIA GPU with CUDA.

```bash
pip install faster-qwen3-tts
```

**PyTorch compatibility note:** CUDA-graph capture in the fast path is not reliable on `torch<=2.5.0` for this project (capture can fail with "operation not permitted when stream is capturing"). We validated `2.5.1+` as working and set that as the minimum supported version.

## Quick Start

### Python

```python
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
ref_audio = "ref_audio.wav"
ref_text = (
    "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up "
    "reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach "
    "of training on verifiable outcomes is doomed."
)

# Streaming — yields audio chunks during generation
for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
    text="What do you mean that I'm not real?", language="English",
    ref_audio=ref_audio, ref_text=ref_text,
    chunk_size=8,  # 8 steps ≈ 667ms of audio per chunk
):
    play(audio_chunk, sr)  # process/send each chunk immediately

# Non-streaming — returns all audio at once
audio_list, sr = model.generate_voice_clone(
    text="Hello world!", language="English",
    ref_audio=ref_audio, ref_text=ref_text,
)
```

### CLI

Voice cloning (reference audio):

```bash
faster-qwen3-tts clone \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --text "What do you mean that I'm not real?" \
  --language English \
  --ref-audio ref_audio.wav \
  --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
  --output out.wav
```

CustomVoice (predefined speaker IDs):

```bash
faster-qwen3-tts custom --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --list-speakers
faster-qwen3-tts custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --text "What do you mean that I'm not real?" \
  --language English \
  --output out.wav
```

VoiceDesign (instruction-based):

```bash
faster-qwen3-tts design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instruct "Warm, confident narrator with slight British accent" \
  --text "Welcome to the show." \
  --language English \
  --output out.wav
```

Streaming (prints RTF after write):

```bash
faster-qwen3-tts custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --text "What do you mean that I'm not real?" \
  --language English \
  --output out.wav \
  --streaming
```

Server mode (keep model hot, stop with `exit`):

```bash
faster-qwen3-tts serve \
  --mode custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --language English \
  --streaming
```

### Demo UI

A minimal web UI that streams audio in real time and shows TTFA and RTF live:

```bash
pip install -e ".[demo]"
python demo/server.py
# open http://localhost:7860
```

Features: voice clone (upload any WAV or use your microphone), voice design (1.7B-VoiceDesign model), streaming/non-streaming toggle, adjustable chunk size, live TTFA/RTF metrics, WAV download.

### OpenAI-compatible API server

`examples/openai_server.py` exposes a `POST /v1/audio/speech` endpoint that follows the OpenAI TTS API contract, so it works out of the box with OpenWebUI, llama-swap, and any other OpenAI-compatible client.

```bash
pip install "faster-qwen3-tts[demo]"
python examples/openai_server.py \
    --ref-audio ref_audio.wav \
    --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
    --language English --port 8000
```

```bash
curl http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model": "tts-1", "input": "Hello world.", "voice": "alloy", "response_format": "wav"}' \
    --output speech.wav
```

To expose multiple voices, pass a JSON file mapping names to reference audio configs — each `voice` value in a request will be routed to the matching entry (`--voices voices.json`). WAV and PCM formats stream chunks as they are generated; MP3 requires `pydub`.

## Results

Benchmarks include tokenization + inference (apples-to-apples with baseline). RTF > 1.0 = faster than real-time. TTFA measured as time to first playable audio chunk using streaming (chunk_size=8).

### 0.6B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.179 | 3,641ms | 1.307 | 597ms | 7.3x / 6.1x |
| DGX Spark (GB10) | 1.17 | 567ms | 2.56 | 280ms | 2.2x / 2.0x |
| RTX 4090 | 0.82 | 800ms | **4.78** | **156ms** | 5.8x / 5.1x |
| RTX 4060 (Windows) | 0.23 | 2,697ms | **2.26** | **413ms** | 9.8x / 6.5x |
| H100 80GB HBM3 | 0.435 | 1,474ms | **3.884** | **228ms** | 8.9x / 6.5x |

### 1.7B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.183 | 3,573ms | 1.089 | 693ms | 6.0x / 5.2x |
| DGX Spark (GB10) | 1.01 | 661ms | 1.87 | 400ms | 1.9x / 1.7x |
| RTX 4090 | 0.82 | 850ms | **4.22** | **174ms** | 5.1x / 4.9x |
| RTX 4060 (Windows) | 0.23 | 2,905ms | **1.83** | **460ms** | 7.9x / 6.3x |
| H100 80GB HBM3 | 0.439 | 1,525ms | **3.304** | **241ms** | 7.5x / 6.3x |

**Note:** Baseline TTFA values are **streaming TTFA** from the community `Qwen3-TTS-streaming` fork (which adds streaming) or from our **dynamic-cache parity streaming** path (no CUDA graphs) where available. The official `Qwen3-TTS` repo does **not** currently support streaming, so without a streaming baseline TTFA would be **time-to-full-audio**. CUDA graphs uses `generate_voice_clone_streaming(chunk_size=8)` for TTFA. Both include text tokenization for fair comparison. Speedup shows throughput / TTFA improvement. The streaming fork reports additional speedups that appear tied to `torch.compile`; we couldn’t reproduce those on Jetson-class devices where `torch.compile` isn’t available.

**GPU architecture notes:** RTX 4090 (2.5 GHz clocks) outperforms H100 (1.8 GHz) for single-stream workloads. H100's lower baseline (RTF 0.59 vs 4090's 0.82) reflects design optimization for batch processing rather than single-stream inference.

### Benchmark your hardware

Benchmarks run from source. You only need [uv](https://docs.astral.sh/uv/) and `./setup.sh`:

**Linux / macOS / WSL:**

```bash
git clone https://github.com/andimarafioti/faster-qwen3-tts
cd faster-qwen3-tts
./setup.sh
./benchmark.sh # or ./benchmark.sh 0.6B or ./benchmark.sh 1.7B for a single model
```

**Windows (Native):**

```cmd
git clone https://github.com/andimarafioti/faster-qwen3-tts
cd faster-qwen3-tts
setup_windows.bat
benchmark_windows.bat   # or benchmark_windows.bat 0.6B / 1.7B / both
```

Results are saved as `bench_results_<GPU_NAME>.json` and audio samples as `sample_0.6B.wav` / `sample_1.7B.wav`.

## Streaming

CUDA graphs support streaming output — audio chunks are yielded during generation with the same per-step performance as non-streaming mode.

### Chunk size vs performance (Jetson AGX Orin, 0.6B)

| chunk_size | TTFA | RTF | Audio per chunk |
|---|---|---|---|
| 1 | 240ms | 0.750 | 83ms |
| 2 | 266ms | 1.042 | 167ms |
| 4 | 362ms | 1.251 | 333ms |
| 8 | 556ms | 1.384 | 667ms |
| 12 | 753ms | 1.449 | 1000ms |
| Non-streaming | — | 1.57 | all at once |

Smaller chunks = lower latency but more decode overhead. `chunk_size=2` is the smallest that stays real-time on Jetson.

**Model seed:** All the different model modes are effectively the same speed. The first time you clone a voice, it takes longer, but later it's cached. Use `benchmarks/compare_modes.py` to reproduce. Example on 0.6B, `chunk_size=8`:

| Mode | TTFA (ms) | RTF | ms/step |
| ---- | --------- | --- | ------- |
| VoiceClone xvec | 152 ± 11 | 5.470 ± 0.032 | 15.2 ± 0.1 |
| VoiceClone full ICL | 149 ± 1 | 5.497 ± 0.026 | 15.2 ± 0.1 |
| CustomVoice | 148 ± 1 | 5.537 ± 0.020 | 15.0 ± 0.1 |

### How streaming works

The CUDA graphs are unchanged — both predictor and talker graphs are replayed per step. The streaming generator yields codec ID chunks every `chunk_size` steps, and the model wrapper decodes each chunk to audio using a sliding window with 25-frame left context (matching the upstream codec's `chunked_decode` pattern) to avoid boundary artifacts.

## Voice Cloning Quality

### Cloning modes

`generate_voice_clone` exposes two modes via `xvec_only`:

| Mode | `xvec_only` | Notes |
|---|---|---|
| Simple (x-vector) | `True` (default) | Speaker embedding only — shorter prefill, clean language switching, no `ref_text` needed |
| Advanced (ICL) | `False` | Full reference audio in context — requires accurate `ref_text`, may produce a brief artifact at the start since it literally continues the sentence `ref_wav` you use |

Simple mode is the default and generally produces clean results. Advanced (ICL) mode can more closely match the reference timbre but requires an accurate transcript and sometimes has a rough start on the first word.

### Decoder context (ICL mode)

The 12 Hz codec uses a causal `chunked_decode`: each frame is reconstructed using prior frames as acoustic context. In ICL mode the reference audio codec tokens are prepended to the generated tokens before decoding, then the reference portion is trimmed from the output. Without this, the codec decoder starts cold with no voice context — the model generates the right tokens but they get reconstructed in the wrong voice. This is handled automatically.

### Text input streaming vs Non-streaming quality

The original Qwen3TTS implementation supports two mode of generation. It either takes the full input text and prepares the utterance, or it feeds the text progressively. This is the `non_streaming_mode` parameter in the generation methods. The name is maintained from the Qwen3TTS implementation, but I understand it might bring some headaches since here we also have general audio output streaming.
`generate_voice_clone` defaults to `non_streaming_mode=True` to put the **full target text** into the prefill before any audio is generated. This improves prosody/consistency for non‑streaming use cases.
`generate_voice_clone_streaming` also defaults to `non_streaming_mode=True`, which pre-fills the full target text before streaming decode. Set it to `False` to match the upstream step‑by‑step text feeding behavior.

**Performance impact (RTX 4090, 1.7B, ICL, chunk_size=8):** TTFA is unchanged (≈159ms ± 1ms), and RTF is effectively the same (nsm=False: 4.87 ± 0.01, nsm=True: 4.85 ± 0.01).

### ICL Phoneme Artifact

In ICL mode the model's prefill ends with the last codec token of the reference audio, so the first generated token is conditioned on whatever phoneme the reference ends on. If the reference ends mid-word, that phoneme bleeds into the generated speech.

**The fix is applied by default.** The wrapper appends 0.5 s of silence to the reference audio before encoding it, giving the model a clean starting point regardless of how the recording ends. Set `append_silence=False` to match the upstream behavior exactly.

## Quality Samples

### Quality Comparison: Qwen3TTS vs FasterQwen3TTS

We provide side‑by‑side audio samples to compare **Qwen3TTS** (dynamic cache) against **FasterQwen3TTS** (static cache) for both CustomVoice and ICL/voice‑clone. The algorithms are equivalent, but the kernels and reduction order differ, so results are not bit‑identical; the samples let you judge the perceptual impact directly. All samples use the **1.7B** models and cap generation at ~14 seconds so the model can finish naturally.

- `samples/parity/README.md` describes the prompts and model details
- `samples/parity/*.wav` contain 2 voices × 2 prompts × {static,dynamic}

**CustomVoice (aiden) – Prompt 1**

<audio controls src="samples/parity/custom_aiden_gen1_static.wav"></audio>
<audio controls src="samples/parity/custom_aiden_gen1_dynamic.wav"></audio>

**CustomVoice (aiden) – Prompt 2**

<audio controls src="samples/parity/custom_aiden_gen2_static.wav"></audio>
<audio controls src="samples/parity/custom_aiden_gen2_dynamic.wav"></audio>

**CustomVoice (serena) – Prompt 1**

<audio controls src="samples/parity/custom_serena_gen1_static.wav"></audio>
<audio controls src="samples/parity/custom_serena_gen1_dynamic.wav"></audio>

**CustomVoice (serena) – Prompt 2**

<audio controls src="samples/parity/custom_serena_gen2_static.wav"></audio>
<audio controls src="samples/parity/custom_serena_gen2_dynamic.wav"></audio>

**ICL (ref_audio.wav) – Prompt 1**

<audio controls src="samples/parity/icl_ref_audio_gen1_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_gen1_dynamic.wav"></audio>

**ICL (ref_audio.wav) – Prompt 2**

<audio controls src="samples/parity/icl_ref_audio_gen2_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_gen2_dynamic.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 1**

<audio controls src="samples/parity/icl_ref_audio_2_gen1_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_2_gen1_dynamic.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 2**

<audio controls src="samples/parity/icl_ref_audio_2_gen2_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_2_gen2_dynamic.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 1**

<audio controls src="samples/parity/icl_ref_audio_3_gen1_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_3_gen1_dynamic.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 2**

<audio controls src="samples/parity/icl_ref_audio_3_gen2_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_3_gen2_dynamic.wav"></audio>

### non_streaming_mode Comparison (ICL)

We provide side‑by‑side samples comparing **non_streaming_mode=False** vs **True** for ICL voice cloning.
All samples use the **1.7B** model with `xvec_only=False`.

- `samples/non_streaming_mode/README.md` describes prompts, settings, and filenames
- `samples/non_streaming_mode/*.wav` contain 3 references × 2 prompts × {nsm_false,nsm_true}

**ICL (ref_audio.wav) – Prompt 1**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen1_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen1_nsm_true.wav"></audio>

**ICL (ref_audio.wav) – Prompt 2**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen2_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen2_nsm_true.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 1**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen1_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen1_nsm_true.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 2**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen2_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen2_nsm_true.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 1**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen1_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen1_nsm_true.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 2**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen2_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen2_nsm_true.wav"></audio>

## Parity

We maintain parity with upstream Qwen3‑TTS in two layers, and document where (and why) the fast path can differ numerically. When we say **Qwen3TTS vs FasterQwen3TTS**, we are comparing the upstream dynamic‑cache path against our static‑cache CUDA‑graph path.

- **Fast path (static cache + CUDA graphs):** Streaming and non‑streaming share the same decode core and match upstream for the initial window where artifacts are most audible. Tests enforce this prefix parity deterministically.
- **Parity mode (dynamic cache, tests only):** A dynamic‑cache decode path (no CUDA graphs) that calls `talker.generate(...)` is used in tests to prove exact token‑level equality against upstream for all model types.

**Why can static cache differ from dynamic cache?** The math is equivalent, but the kernel path is not. Static cache uses a fixed max‑length KV buffer and an explicit attention mask, which often selects a different SDPA kernel than the dynamic cache path (shorter K/V, `is_causal=True`, mask‑free). In BF16/TF32, different kernel/reduction orders are not bit‑exact, so the outputs can differ slightly even when the algorithm is the same.

**Parity streaming note:** The dynamic‑cache parity streaming path is intentionally slow. On an RTX 4090 it measured ~0.77s TTFA (chunk_size=8) and ~1.17s TTFA (chunk_size=12), versus ~0.16–0.18s TTFA in the fast CUDA‑graph path. Use parity streaming only for validation, not performance.

Tests live in `tests/test_e2e_parity.py` and cover:

- Voice clone (x‑vector) prefix parity vs upstream
- Streaming vs non‑streaming parity (fast path)
- CustomVoice full equality (parity mode)
- VoiceDesign full equality (parity mode)
- Voice clone ICL full equality (parity mode)

You can control the model IDs used by tests via environment variables:

```
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
QWEN_TTS_CUSTOM_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_TTS_VOICE_DESIGN_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```
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

You can now pass this precomputed prompt directly to the public APIs:

```python
import torch
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
spk_emb = torch.load("speaker.pt", weights_only=True).to(model.device)

voice_clone_prompt = {
    "ref_code": [None],
    "ref_spk_embedding": [spk_emb],
    "x_vector_only_mode": [True],
    "icl_mode": [False],
}

audio_list, sr = model.generate_voice_clone(
    text="Hello world!",
    language="English",
    ref_audio="ignored_when_voice_clone_prompt_is_set.wav",
    ref_text="",
    voice_clone_prompt=voice_clone_prompt,
)
```

When `voice_clone_prompt` is provided, prompt extraction from `ref_audio` is skipped.

## License

MIT

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team
- [Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) for ideas and code we adapted for streaming
- [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) for inspiration on CUDA graph usage
- NVIDIA for providing the Jetson AGX Orin board
