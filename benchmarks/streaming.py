#!/usr/bin/env python3
"""Benchmark streaming vs non-streaming generation with CUDA graphs."""
import torch
import time
import os
import numpy as np
import soundfile as sf
from qwen3_tts_cuda_graphs import Qwen3TTSCudaGraphs

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SIZE = os.environ.get('MODEL_SIZE', '0.6B')
MODEL_ID = f'Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base'
text = "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it. The robots have officially won. Please remain calm."
ref_audio = os.path.join(PROJECT_DIR, 'ref_audio.wav')
ref_text = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."

print("Loading model...")
model = Qwen3TTSCudaGraphs.from_pretrained(
    MODEL_ID,
    device='cuda',
    dtype=torch.bfloat16,
    attn_implementation='eager',
    max_seq_len=2048,
)

# Warmup (includes CUDA graph capture)
print("\nWarmup run...")
start = time.perf_counter()
audio_list, sr = model.generate_voice_clone(
    text=text[:50],
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
    max_new_tokens=20,
)
print(f"Warmup: {time.perf_counter() - start:.2f}s")

# === STREAMING TTFA ===
print("\n=== Streaming TTFA (time to first audio chunk) ===")
for chunk_size in [4, 8, 12]:
    ttfa_results = []
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = model.generate_voice_clone_streaming(
            text=text,
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
            chunk_size=chunk_size,
        )
        first_chunk, sr, timing = next(gen)
        torch.cuda.synchronize()
        ttfa_ms = (time.perf_counter() - t0) * 1000
        ttfa_results.append(ttfa_ms)
        gen.close()

    mean_ttfa = np.mean(ttfa_results)
    std_ttfa = np.std(ttfa_results)
    audio_ms = chunk_size / 12.0 * 1000
    print(f"  chunk_size={chunk_size:2d}: TTFA={mean_ttfa:.0f}ms ± {std_ttfa:.0f}ms "
          f"(first {audio_ms:.0f}ms of audio)")

# === FULL STREAMING RUN ===
print("\n=== Full streaming run (chunk_size=12) ===")
for run in range(3):
    chunks = []
    chunk_timings = []
    torch.cuda.synchronize()
    t_total = time.perf_counter()

    for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
        chunk_size=12,
    ):
        t_chunk = time.perf_counter() - t_total
        chunks.append(audio_chunk)
        chunk_timings.append({
            'wall_time_s': t_chunk,
            'chunk_steps': timing['chunk_steps'],
            'decode_ms': timing['decode_ms'],
        })

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t_total

    full_audio = np.concatenate(chunks)
    audio_duration = len(full_audio) / sr
    rtf = audio_duration / total_time

    print(f"\nRun {run+1}: {len(chunks)} chunks, "
          f"audio={audio_duration:.1f}s, time={total_time:.1f}s, RTF={rtf:.3f}")

    # Per-chunk timing
    for i, ct in enumerate(chunk_timings):
        inter = ct['wall_time_s'] - (chunk_timings[i-1]['wall_time_s'] if i > 0 else 0)
        label = "TTFA" if i == 0 else f"  #{i+1}"
        print(f"    {label}: {inter*1000:.0f}ms ({ct['chunk_steps']} steps, "
              f"decode={ct['decode_ms']:.0f}ms)")

# === NON-STREAMING COMPARISON ===
print("\n=== Non-streaming comparison ===")
for run in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    audio_list, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    audio = audio_list[0]
    audio_duration = len(audio) / sr
    rtf = audio_duration / total_time

    print(f"Run {run+1}: audio={audio_duration:.1f}s, time={total_time:.1f}s, RTF={rtf:.3f}")

# === SAVE AUDIO SAMPLES ===
print("\n=== Saving audio samples ===")

# Streaming
chunks = []
for audio_chunk, sr, _ in model.generate_voice_clone_streaming(
    text=text, language="English",
    ref_audio=ref_audio, ref_text=ref_text, chunk_size=12,
):
    chunks.append(audio_chunk)
streaming_audio = np.concatenate(chunks)
sf.write(os.path.join(PROJECT_DIR, f'sample_streaming_{MODEL_SIZE}.wav'), streaming_audio, sr)
print(f"Saved sample_streaming_{MODEL_SIZE}.wav ({len(streaming_audio)/sr:.1f}s)")

# Non-streaming
audio_list, sr = model.generate_voice_clone(
    text=text, language="English",
    ref_audio=ref_audio, ref_text=ref_text,
)
sf.write(os.path.join(PROJECT_DIR, f'sample_nonstreaming_{MODEL_SIZE}.wav'), audio_list[0], sr)
print(f"Saved sample_nonstreaming_{MODEL_SIZE}.wav ({len(audio_list[0])/sr:.1f}s)")

print("\nDone!")
