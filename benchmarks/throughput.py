#!/usr/bin/env python3
"""Benchmark throughput: CUDA graphs using the Qwen3TTSCudaGraphs wrapper."""
import torch
import time
import os
import numpy as np
import soundfile as sf
from qwen3_tts_cuda_graphs import Qwen3TTSCudaGraphs

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SIZE = os.environ.get('MODEL_SIZE', '0.6B')
MODEL_PATH = os.path.join(PROJECT_DIR, 'models', f'Qwen3-TTS-12Hz-{MODEL_SIZE}-Base')
text = "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it. The robots have officially won. Please remain calm."
ref_audio = os.path.join(PROJECT_DIR, 'ref_audio.wav')
ref_text = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."

print("Loading model...")
model = Qwen3TTSCudaGraphs.from_pretrained(
    MODEL_PATH,
    device='cuda',
    dtype=torch.bfloat16,
    attn_implementation='eager',
    max_seq_len=2048,
)

print("\nWarmup run (includes CUDA graph capture)...")
start = time.perf_counter()
audio_list, sr = model.generate_voice_clone(
    text=text[:50],  # Short warmup
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
    max_new_tokens=20,
)
warmup_time = time.perf_counter() - start
print(f"Warmup: {warmup_time:.2f}s")

# TTFA (Time to First Audio) measurement
print("\nMeasuring TTFA (5 runs)...")
ttfa_results = []
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    audio_list, sr = model.generate_voice_clone(
        text=text[:30],
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
        max_new_tokens=1,
    )
    torch.cuda.synchronize()
    ttfa_ms = (time.perf_counter() - t0) * 1000
    ttfa_results.append(ttfa_ms)
    print(f"  Run {i+1}: {ttfa_ms:.1f}ms")

ttfa_mean = np.mean(ttfa_results)
ttfa_std = np.std(ttfa_results)
print(f"  TTFA: {ttfa_mean:.1f}ms ± {ttfa_std:.1f}ms")

# Full benchmark runs
print("\nBenchmark runs...")
results = []
for run in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    audio_list, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
        max_new_tokens=2048,
    )
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    
    audio = audio_list[0]
    audio_duration = len(audio) / sr
    rtf = audio_duration / total_time
    
    # Estimate steps (12 Hz codec)
    n_steps = int(audio_duration * 12)
    ms_per_step = (total_time * 1000) / n_steps if n_steps > 0 else 0
    
    print(f"Run {run+1}: {n_steps} steps, {ms_per_step:.1f}ms/step, "
          f"audio={audio_duration:.1f}s, time={total_time:.1f}s, RTF={rtf:.3f}")
    
    results.append({
        'steps': n_steps,
        'ms_per_step': ms_per_step,
        'rtf': rtf,
        'total_time': total_time,
        'audio_duration': audio_duration,
    })

if results:
    avg_ms = np.mean([r['ms_per_step'] for r in results])
    avg_rtf = np.mean([r['rtf'] for r in results])
    print(f"\n=== {MODEL_SIZE} Average: {avg_ms:.1f}ms/step, RTF={avg_rtf:.3f}, TTFA={ttfa_mean:.0f}ms ===")
    
    # Save audio from last run
    try:
        out_wav = os.path.join(PROJECT_DIR, f'sample_{MODEL_SIZE}.wav')
        sf.write(out_wav, audio, sr)
        print(f"\nSaved sample audio to {out_wav}")
    except Exception as e:
        print(f"Audio save failed: {e}")
    
    # Save results as JSON
    import json
    import subprocess
    gpu_name = "Unknown"
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                      stderr=subprocess.DEVNULL, text=True)
        gpu_name = out.strip().split('\n')[0].replace(' ', '_')
    except:
        pass
    
    bench_data = {
        'model': MODEL_SIZE,
        'gpu': gpu_name,
        'avg_ms_per_step': avg_ms,
        'avg_rtf': avg_rtf,
        'ttfa_ms': ttfa_mean,
        'ttfa_std_ms': ttfa_std,
        'runs': results,
    }
    
    json_path = f'bench_results_{gpu_name}.json'
    with open(json_path, 'r+' if os.path.exists(json_path) else 'w') as f:
        if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
            f.seek(0)
            try:
                data = json.load(f)
            except:
                data = {}
        else:
            data = {}
        data[MODEL_SIZE] = bench_data
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {json_path}")
