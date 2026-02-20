#!/usr/bin/env python3
"""Benchmark custom-voice generation with CUDA graphs."""
import torch
import time
import os
import numpy as np
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SIZE = os.environ.get('MODEL_SIZE', '0.6B')
MODEL_ID = f'Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-CustomVoice'
text = os.environ.get(
    'TEXT',
    "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it."
)
language = os.environ.get('LANGUAGE', 'English')
instruct = os.environ.get('INSTRUCT', '')

print("Loading model...")
model = FasterQwen3TTS.from_pretrained(
    MODEL_ID,
    device='cuda',
    dtype=torch.bfloat16,
    attn_implementation='eager',
    max_seq_len=2048,
)

speakers = model.model.get_supported_speakers() or []
if not speakers:
    raise RuntimeError("No speakers reported by custom voice model")

speaker = os.environ.get('SPEAKER', speakers[0])
print(f"Using speaker: {speaker}")

print("\nWarmup run (includes CUDA graph capture)...")
start = time.perf_counter()
model.generate_custom_voice(
    text=text[:50],
    speaker=speaker,
    language=language,
    instruct=instruct,
    max_new_tokens=20,
)
warmup_time = time.perf_counter() - start
print(f"Warmup: {warmup_time:.2f}s")

# TTFA (Time to First Audio) via streaming
CHUNK_SIZES = [4, 8, 12]
PRIMARY_CHUNK_SIZE = 8
print("\nMeasuring streaming TTFA (5 runs per chunk size)...")
all_ttfa = {}
for chunk_size in CHUNK_SIZES:
    ttfas = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = model.generate_custom_voice_streaming(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            chunk_size=chunk_size,
            max_new_tokens=512,
        )
        try:
            chunk, sr, timing = next(gen)
        finally:
            gen.close()
        torch.cuda.synchronize()
        ttfa = (time.perf_counter() - t0) * 1000
        ttfas.append(ttfa)
    all_ttfa[chunk_size] = (np.mean(ttfas), np.std(ttfas))
    if chunk_size == PRIMARY_CHUNK_SIZE:
        print(f"  Run 1: {ttfas[0]:.1f}ms")
        for i in range(1, 5):
            print(f"  Run {i+1}: {ttfas[i]:.1f}ms")
    print(f"  chunk_size={chunk_size:2d}: TTFA={all_ttfa[chunk_size][0]:.0f}ms ± {all_ttfa[chunk_size][1]:.0f}ms" + (" <<" if chunk_size == PRIMARY_CHUNK_SIZE else ""))

# Throughput benchmark
print("\nBenchmark runs...")
rtfs = []
ms_per_steps = []
for run in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    audio_list, sr = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        instruct=instruct,
        max_new_tokens=512,
    )
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    audio = audio_list[0]
    audio_dur = len(audio) / sr
    rtf = audio_dur / total
    rtfs.append(rtf)
    n_steps = int(round(audio_dur * 12))
    ms_per_step = (total / max(n_steps, 1)) * 1000
    ms_per_steps.append(ms_per_step)
    print(f"Run {run+1}: {n_steps} steps, {ms_per_step:.1f}ms/step, audio={audio_dur:.1f}s, time={total:.1f}s, RTF={rtf:.3f}")

print(f"\n=== {MODEL_SIZE} CustomVoice Average: {np.mean(ms_per_steps):.1f}ms/step, RTF={np.mean(rtfs):.3f}, TTFA={all_ttfa[PRIMARY_CHUNK_SIZE][0]:.0f}ms ===")

out_path = os.path.join(PROJECT_DIR, f'sample_custom_voice_{MODEL_SIZE}.wav')
sf.write(out_path, audio_list[0], sr)
print(f"\nSaved sample audio to {out_path}")
