#!/usr/bin/env python3
"""Sweep chunk sizes to find RTF vs latency tradeoff."""
import torch
import time
import os
import numpy as np
import soundfile as sf
from qwen3_tts_cuda_graphs import Qwen3TTSCudaGraphs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SIZE = os.environ.get('MODEL_SIZE', '0.6B')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', f'Qwen3-TTS-12Hz-{MODEL_SIZE}-Base')
text = "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it. The robots have officially won. Please remain calm."
ref_audio = os.path.join(SCRIPT_DIR, 'ref_audio.wav')
ref_text = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."

print("Loading model...")
model = Qwen3TTSCudaGraphs.from_pretrained(
    MODEL_PATH, device='cuda', dtype=torch.bfloat16,
    attn_implementation='eager', max_seq_len=2048,
)

# Warmup
print("Warmup...")
model.generate_voice_clone(
    text=text[:50], language="English",
    ref_audio=ref_audio, ref_text=ref_text, max_new_tokens=20,
)

# Non-streaming baseline
print("\n=== Non-streaming baseline (3 runs) ===")
ns_rtfs = []
for run in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    audio_list, sr = model.generate_voice_clone(
        text=text, language="English",
        ref_audio=ref_audio, ref_text=ref_text,
    )
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    dur = len(audio_list[0]) / sr
    rtf = dur / total
    ns_rtfs.append(rtf)
    print(f"  Run {run+1}: audio={dur:.1f}s, time={total:.1f}s, RTF={rtf:.3f}")
print(f"  Average RTF: {np.mean(ns_rtfs):.3f}")

# Chunk size sweep
print("\n=== Chunk size sweep (3 runs each) ===")
print(f"{'chunk':>6} {'TTFA_ms':>8} {'RTF':>8} {'chunks':>7} {'audio_s':>8} {'time_s':>7}")
print("-" * 52)

for chunk_size in [1, 2, 4, 8, 12]:
    rtfs = []
    ttfas = []
    last_chunks = None
    last_sr = None

    for run in range(3):
        chunks = []
        torch.cuda.synchronize()
        t_total = time.perf_counter()
        t_first = None

        for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
            text=text, language="English",
            ref_audio=ref_audio, ref_text=ref_text,
            chunk_size=chunk_size,
        ):
            if t_first is None:
                torch.cuda.synchronize()
                t_first = time.perf_counter() - t_total
            chunks.append(audio_chunk)

        torch.cuda.synchronize()
        total = time.perf_counter() - t_total

        full_audio = np.concatenate(chunks)
        dur = len(full_audio) / sr
        rtf = dur / total
        rtfs.append(rtf)
        ttfas.append(t_first * 1000)
        last_chunks = chunks
        last_sr = sr

    avg_rtf = np.mean(rtfs)
    avg_ttfa = np.mean(ttfas)
    n_chunks = len(last_chunks)
    dur = sum(len(c) for c in last_chunks) / last_sr

    marker = " <-- below real-time!" if avg_rtf < 1.0 else ""
    print(f"{chunk_size:>6} {avg_ttfa:>8.0f} {avg_rtf:>8.3f} {n_chunks:>7} {dur:>8.1f} {np.mean([len(np.concatenate(last_chunks))/last_sr / r for r in rtfs]):>7.1f}{marker}")

    # Save audio for chunk_size=1
    if chunk_size == 1:
        full_audio = np.concatenate(last_chunks)
        out_path = os.path.join(SCRIPT_DIR, f'sample_streaming_chunk1_{MODEL_SIZE}.wav')
        sf.write(out_path, full_audio, last_sr)
        print(f"  --> Saved {out_path} ({len(full_audio)/last_sr:.1f}s)")

print(f"\nNon-streaming baseline RTF: {np.mean(ns_rtfs):.3f}")
print("Done!")
