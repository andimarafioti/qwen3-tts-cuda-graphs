# Faster Qwen3-TTS

Real-time Qwen3-TTS inference using CUDA graph capture. No Flash Attention, no vLLM, no Triton. Just `torch.cuda.CUDAGraph`. Supports both streaming and non-streaming generation.

## Results

Benchmarks include tokenization + inference (apples-to-apples with baseline). RTF > 1.0 = faster than real-time. TTFA measured as time to first playable audio chunk using streaming (chunk_size=8).

### 0.6B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.175 | 2,572ms | 1.57 | 556ms | 9.0x / 4.6x |
| Jetson Thor | 0.803 | 862ms | 1.50 | 505ms | 1.9x / 1.7x |
| DGX Spark (GB10) | 1.19 | 631ms | 2.26 | 364ms | 1.9x / 1.7x |
| RTX 4090 | 1.34 | 462ms | **5.56** | **152ms** | 4.1x / 3.0x |
| H100 80GB HBM3 | 0.59 | 1,049ms | **4.19** | **224ms** | 7.1x / 4.7x |

### 1.7B Model

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.130 | 2,594ms | 1.27 | 650ms | 9.8x / 4.0x |
| Jetson Thor | 0.772 | 912ms | 1.26 | 595ms | 1.6x / 1.5x |
| DGX Spark (GB10) | 0.975 | 749ms | 1.66 | 464ms | 1.7x / 1.6x |
| RTX 4090 | 1.32 | 468ms | **4.85** | **170ms** | 3.7x / 2.8x |
| H100 80GB HBM3 | 0.59 | 1,045ms | **3.98** | **236ms** | 6.7x / 4.4x |

**Note:** Baseline TTFA values are **streaming TTFA** from the community `Qwen3-TTS-streaming` fork (which adds streaming). The official `Qwen3-TTS` repo does **not** currently support streaming, so its “TTFA” is effectively **time-to-full-audio**. With RTF near 1.0, that means waiting for the entire sentence/paragraph to finish speaking before you hear anything. CUDA graphs uses `generate_voice_clone_streaming(chunk_size=8)` for TTFA. Both include text tokenization for fair comparison. Speedup shows throughput / TTFA improvement. The streaming fork reports additional speedups that appear tied to `torch.compile`; we couldn’t reproduce those on Jetson-class devices where `torch.compile` isn’t available.

**GPU architecture notes:** RTX 4090 (2.5 GHz clocks) outperforms H100 (1.8 GHz) for single-stream workloads. H100's lower baseline (RTF 0.59 vs 4090's 1.34) reflects design optimization for batch processing rather than single-stream inference.

## Quick Start

```bash
git clone https://github.com/andimarafioti/faster-qwen3-tts
cd faster-qwen3-tts
./setup.sh       # creates venv with uv, installs deps, downloads models
./benchmark.sh   # runs full benchmark, saves JSON + audio samples
```

Requires: Python 3.10+, NVIDIA GPU with CUDA, [uv](https://docs.astral.sh/uv/).

### Install (PyPI)

```bash
pip install faster-qwen3-tts
```

Note: This installs the `qwen-tts` PyPI package (`>=0.1.1`).

Install from source:

```bash
pip install -e .
```

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
| 1 | 240ms | 0.750 | 83ms |
| 2 | 266ms | 1.042 | 167ms |
| 4 | 362ms | 1.251 | 333ms |
| 8 | 556ms | 1.384 | 667ms |
| 12 | 753ms | 1.449 | 1000ms |
| Non-streaming | — | 1.36 | all at once |

Smaller chunks = lower latency but more decode overhead. `chunk_size=2` is the smallest that stays real-time on Jetson.

### Usage

```python
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")

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

### CLI

Voice cloning (reference audio):

```bash
faster-qwen3-tts clone \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --text "Hello world!" \
  --language English \
  --ref-audio ref.wav \
  --ref-text "Reference transcript" \
  --output out.wav
```

CustomVoice (predefined speaker IDs):

```bash
faster-qwen3-tts custom --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --list-speakers
faster-qwen3-tts custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --text "Hello!" \
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
  --text "Hello!" \
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

### How it works

The CUDA graphs are unchanged — both predictor and talker graphs are replayed per step. The streaming generator yields codec ID chunks every `chunk_size` steps, and the model wrapper decodes each chunk to audio using a sliding window with 25-frame left context (matching the upstream codec's `chunked_decode` pattern) to avoid boundary artifacts.

## Voice Cloning: ICL Phoneme Artifact

In ICL (In-Context Learning) mode — the default voice cloning path — the model's prefill sequence ends with the last codec token of the reference audio. The model conditions its **first generated token** on whatever phoneme the reference audio happens to end on. If the reference ends mid-word or on a consonant cluster, that phoneme bleeds into the very start of the generated speech.

**The fix is applied automatically.** The wrapper appends 0.5 seconds of silence to the reference audio before encoding it. This ensures the last codec tokens in the prefill represent silence, giving the model a clean starting point regardless of how the reference recording ends — no changes to your calling code required.

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

## License

MIT

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team
- [Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) for ideas and code we adapted for streaming
- [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) for inspiration on CUDA graph usage
- NVIDIA for providing the Jetson AGX Orin board
