#!/bin/bash
# Portable Qwen3-TTS CUDA Graph Benchmark
# Run on any system with PyTorch + CUDA + the Qwen3-TTS models
#
# Usage:
#   ./bench_portable.sh /path/to/qwen3-tts
#
# Expects:
#   - Python 3.10+ with torch (CUDA) installed
#   - Models at $BASE/models/Qwen3-TTS-12Hz-0.6B-Base and 1.7B-Base
#   - The cuda_graphs/ directory from this repo
#
# Output: prints benchmark table and saves results to bench_results_<gpu>.json

set -e

BASE="${1:-$(dirname "$0")/..}"
cd "$BASE"

# Auto-detect venv or system python
if [ -f .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python3
fi

echo "=== Qwen3-TTS CUDA Graph Benchmark ==="
echo "Python: $($PY --version 2>&1)"
echo "PyTorch: $($PY -c 'import torch; print(torch.__version__)')"
echo "CUDA: $($PY -c 'import torch; print(torch.version.cuda)')"
GPU_NAME=$($PY -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo "unknown")
echo "GPU: $GPU_NAME"
echo ""

# Kill ollama if running (avoid GPU contention)
pkill -f ollama 2>/dev/null || true
sleep 1

$PY -c "
import torch, time, sys, json, numpy as np
sys.path.insert(0, 'cuda_graphs')

gpu_name = torch.cuda.get_device_name(0)
results = {'gpu': gpu_name, 'models': {}}

for model_size in ['0.6B', '1.7B']:
    model_path = f'./models/Qwen3-TTS-12Hz-{model_size}-Base'
    print(f'\\n=== {model_size} Model ===')
    print(f'Loading {model_path}...')

    try:
        # Import here to allow reloading
        from fast_generate_v5 import FastGenerateV5

        fg = FastGenerateV5(model_path)

        # Build inputs
        text = 'The quick brown fox jumps over the lazy dog while the sun shines brightly in the clear blue sky.'
        ref_audio_path = './ref_audio.wav'

        import soundfile as sf
        if not __import__('os').path.exists(ref_audio_path):
            print(f'Warning: {ref_audio_path} not found, skipping')
            continue

        tie = fg.build_inputs(text, ref_audio_path)
        print(f'Input shape: {tie.shape}, prefill_len: {tie.shape[1]}')

        # Setup CUDA graphs
        print('Capturing CUDA graphs...')
        fg.capture(prefill_len=tie.shape[1], num_warmup=3)

        # Warmup
        print('Warmup run...')
        _ = fg.generate(tie, max_steps=20)
        torch.cuda.synchronize()

        # TTFT measurement
        print('Measuring TTFT...')
        ttft_times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = fg.generate(tie, max_steps=1)
            torch.cuda.synchronize()
            ttft_times.append((time.perf_counter() - t0) * 1000)
        ttft_ms = np.mean(ttft_times)

        # Benchmark
        print('Benchmarking (3 runs)...')
        run_results = []
        for i in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = fg.generate(tie, max_steps=500)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            n_steps = output['num_steps']
            audio_dur = output['audio_duration']
            ms_per_step = (elapsed / n_steps) * 1000
            rtf = audio_dur / elapsed

            run_results.append({
                'steps': n_steps,
                'ms_per_step': round(ms_per_step, 1),
                'audio_s': round(audio_dur, 2),
                'gen_s': round(elapsed, 2),
                'rtf': round(rtf, 3)
            })
            print(f'  Run {i+1}: {n_steps} steps, {ms_per_step:.1f}ms/step, audio={audio_dur:.1f}s, time={elapsed:.1f}s, RTF={rtf:.3f}')

        avg_ms = np.mean([r['ms_per_step'] for r in run_results])
        avg_rtf = np.mean([r['rtf'] for r in run_results])
        print(f'  Average: {avg_ms:.1f}ms/step, RTF={avg_rtf:.3f}, TTFT={ttft_ms:.0f}ms')

        results['models'][model_size] = {
            'runs': run_results,
            'avg_ms_per_step': round(avg_ms, 1),
            'avg_rtf': round(avg_rtf, 3),
            'ttft_ms': round(ttft_ms, 1)
        }

        # Cleanup
        del fg
        torch.cuda.empty_cache()

    except Exception as e:
        print(f'  Error: {e}')
        import traceback
        traceback.print_exc()
        results['models'][model_size] = {'error': str(e)}

# Print summary
print(f'\\n=== Summary ({gpu_name}) ===')
print(f'{\"Model\":<8} {\"ms/step\":<10} {\"RTF\":<8} {\"TTFT\":<8} {\"Real-time?\":<10}')
print('-' * 44)
for size, data in results['models'].items():
    if 'error' not in data:
        rt = 'YES' if data['avg_rtf'] > 1.0 else 'no'
        print(f'{size:<8} {data[\"avg_ms_per_step\"]:<10} {data[\"avg_rtf\"]:<8} {data[\"ttft_ms\"]:<8.0f}ms {rt:<10}')

# Save results
gpu_slug = gpu_name.replace(' ', '_').replace('/', '_')
out_path = f'cuda_graphs/bench_results_{gpu_slug}.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nResults saved to {out_path}')
"

echo ""
echo "Done! Copy the JSON file and send it back."
