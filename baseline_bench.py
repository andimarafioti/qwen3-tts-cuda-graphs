# Baseline benchmark - copy paste and run
# Usage: python baseline_bench.py
import torch, time, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_tts import Qwen3TTSModel

MODEL_SIZE = os.environ.get('MODEL_SIZE', '0.6B')
MODEL_PATH = f'./models/Qwen3-TTS-12Hz-{MODEL_SIZE}-Base'
text = "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it. The robots have officially won. Please remain calm."
ref_audio = './ref_audio.wav'
ref_text = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs."

print(f"=== {MODEL_SIZE} Baseline Benchmark ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")
model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map='cuda:0', dtype=torch.bfloat16)

# Warmup (3 runs)
print("Warmup...")
for _ in range(3):
    _ = model.generate_voice_clone(text=text, ref_audio=ref_audio, ref_text=ref_text, max_new_tokens=20)
    torch.cuda.synchronize()

# TTFA (streaming)
print("\nTTFA (5 runs, streaming)...")
ttfa_times = []
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for chunk in model.stream_generate_voice_clone(text=text, ref_audio=ref_audio, ref_text=ref_text):
        ttfa = (time.perf_counter() - t0) * 1000
        ttfa_times.append(ttfa)
        print(f"  Run {i+1}: {ttfa:.0f}ms")
        break
print(f"  TTFA: {np.mean(ttfa_times):.0f}ms")

# Throughput (3 runs)
print("\nThroughput (3 runs)...")
for i in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wav, sr = model.generate_voice_clone(text=text, ref_audio=ref_audio, ref_text=ref_text)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    audio_dur = len(wav) / sr
    rtf = audio_dur / elapsed
    print(f"  Run {i+1}: audio={audio_dur:.1f}s, time={elapsed:.1f}s, RTF={rtf:.3f}")
