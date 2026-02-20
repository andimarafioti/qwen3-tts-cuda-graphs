# Baseline benchmark - copy paste and run
# Usage: python baseline_bench.py
import torch, time, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_tts import Qwen3TTSModel

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SIZE = os.environ.get('MODEL_SIZE', '0.6B')
MODEL_ID = f'Qwen/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base'
text = "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it. The robots have officially won. Please remain calm."
ref_audio = os.path.join(PROJECT_DIR, 'ref_audio.wav')
ref_text = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs."

print(f"=== {MODEL_SIZE} Baseline Benchmark ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Loading model...", flush=True)
model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map='cuda:0', dtype=torch.bfloat16)

# Pre-compute voice clone prompt once (avoids re-encoding ref audio every run)
print("Building voice clone prompt...", flush=True)
voice_clone_prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)

# Warmup (3 runs)
print("Warmup...", flush=True)
for _ in range(3):
    _ = model.generate_voice_clone(text=text, voice_clone_prompt=voice_clone_prompt, max_new_tokens=20)
    torch.cuda.synchronize()

# TTFA (streaming)
print("\nTTFA (5 runs, streaming)...", flush=True)
ttfa_times = []
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen = model.stream_generate_voice_clone(text=text, voice_clone_prompt=voice_clone_prompt)
    try:
        chunk, sr = next(gen)
    finally:
        gen.close()
    ttfa = (time.perf_counter() - t0) * 1000
    ttfa_times.append(ttfa)
    print(f"  Run {i+1}: {ttfa:.0f}ms", flush=True)
    torch.cuda.synchronize()
print(f"  TTFA: {np.mean(ttfa_times):.0f}ms")

# Throughput (3 runs)
print("\nThroughput (3 runs)...", flush=True)
for i in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Cap at 512 tokens (~42s of audio at 12Hz). The vanilla generate_voice_clone
    # doesn't suppress the top-1024 special token IDs (unlike the CUDA graphs version),
    # so the model occasionally samples those and never produces EOS, especially
    # with the 1.7B variant. This cap prevents runaway generation.
    wav, sr = model.generate_voice_clone(text=text, voice_clone_prompt=voice_clone_prompt, max_new_tokens=512)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    audio_dur = len(wav[0]) / sr
    rtf = audio_dur / elapsed
    print(f"  Run {i+1}: audio={audio_dur:.1f}s, time={elapsed:.1f}s, RTF={rtf:.3f}", flush=True)
