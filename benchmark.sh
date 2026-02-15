#!/bin/bash
# Benchmark Qwen3-TTS with CUDA Graphs
# Usage: ./benchmark.sh [--model 0.6B|1.7B|both]
set -e

MODEL="${1:-both}"
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PY="${PYTHON:-python3}"

# Check deps
$PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: PyTorch with CUDA required. Run setup.sh first."
    exit 1
}

# Kill ollama if running
pkill -f ollama 2>/dev/null || true

GPU_NAME=$($PY -c 'import torch; print(torch.cuda.get_device_name(0))')
echo "=== Qwen3-TTS CUDA Graph Benchmark ==="
echo "GPU: $GPU_NAME"
echo "PyTorch: $($PY -c 'import torch; print(torch.__version__)')"
echo "CUDA: $($PY -c 'import torch; print(torch.version.cuda)')"
echo ""

$PY - "$MODEL" <<'PYEOF'
import torch, time, sys, json, os
import numpy as np

model_arg = sys.argv[1] if len(sys.argv) > 1 else "both"
gpu_name = torch.cuda.get_device_name(0)

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or '.')

from qwen_tts import Qwen3TTSModel
from transformers import PretrainedConfig
from manual_cudagraph_predictor import ManualPredictorGraph
from manual_cudagraph_talker import ManualTalkerGraph
from fast_generate_v5 import fast_generate_v5

TEXT = "The quick brown fox jumps over the lazy dog while the sun shines brightly in the clear blue sky."
REF_AUDIO = "./ref_audio.wav"

models_to_test = []
if model_arg in ("0.6B", "both"):
    models_to_test.append("0.6B")
if model_arg in ("1.7B", "both"):
    models_to_test.append("1.7B")

results = {"gpu": gpu_name, "pytorch": torch.__version__, "cuda": torch.version.cuda, "models": {}}

for model_size in models_to_test:
    model_path = f"./models/Qwen3-TTS-12Hz-{model_size}-Base"
    if not os.path.exists(model_path):
        print(f"\n{model_size}: model not found at {model_path}, skipping")
        continue

    print(f"\n{'='*50}")
    print(f"{model_size} Model")
    print(f"{'='*50}")

    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    config = PretrainedConfig.from_pretrained(model_path)

    # Build inputs
    import soundfile as sf
    ref_audio, sr = sf.read(REF_AUDIO)
    ref_audio_tensor = torch.tensor(ref_audio, dtype=torch.float32).unsqueeze(0)
    if ref_audio_tensor.dim() == 1:
        ref_audio_tensor = ref_audio_tensor.unsqueeze(0)

    # Use model's tokenizer/processor to build inputs
    # This depends on the specific model API - adapt as needed
    print("Building inputs...")
    try:
        talker_input_embeds, attention_mask, trailing_text_hiddens, tts_pad_embed = model.prepare_tts_inputs(
            text=TEXT, ref_audio=REF_AUDIO
        )
    except AttributeError:
        # Fallback: use bench_v5.py's approach
        exec(open("bench_v5.py").read().split("# Setup CUDA graphs")[0])
        print("  (used bench_v5 input preparation)")

    prefill_len = talker_input_embeds.shape[1]
    print(f"Input shape: {talker_input_embeds.shape}, prefill_len: {prefill_len}")

    # Setup CUDA graphs
    print("Capturing CUDA graphs...")
    predictor_graph = ManualPredictorGraph(model.tts_predictor, model.tts_predictor.config)
    talker_graph = ManualTalkerGraph(model.talker, model.talker.config)

    # Warmup predictor
    predictor_graph.warmup(num_runs=3)
    predictor_graph.capture()
    print("  Predictor graph captured")

    # Warmup talker
    talker_graph.warmup(prefill_len=prefill_len, num_runs=3)
    talker_graph.capture()
    print("  Talker graph captured")

    # Warmup generation
    print("Warmup run...")
    _ = fast_generate_v5(
        model.talker, talker_input_embeds, attention_mask,
        trailing_text_hiddens, tts_pad_embed, config,
        predictor_graph, talker_graph, max_new_tokens=20
    )
    torch.cuda.synchronize()

    # TTFT
    print("Measuring TTFT...")
    ttft_times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fast_generate_v5(
            model.talker, talker_input_embeds, attention_mask,
            trailing_text_hiddens, tts_pad_embed, config,
            predictor_graph, talker_graph, max_new_tokens=1
        )
        torch.cuda.synchronize()
        ttft_times.append((time.perf_counter() - t0) * 1000)
    ttft_ms = np.mean(ttft_times)
    print(f"  TTFT: {ttft_ms:.1f}ms (±{np.std(ttft_times):.1f}ms)")

    # Benchmark
    print("Benchmarking (3 runs)...")
    runs = []
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = fast_generate_v5(
            model.talker, talker_input_embeds, attention_mask,
            trailing_text_hiddens, tts_pad_embed, config,
            predictor_graph, talker_graph, max_new_tokens=500
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        n_steps = output["num_steps"]
        audio_dur = output["audio_duration"]
        ms_step = (elapsed / n_steps) * 1000
        rtf = audio_dur / elapsed

        runs.append({"steps": n_steps, "ms_per_step": round(ms_step, 1),
                      "audio_s": round(audio_dur, 2), "gen_s": round(elapsed, 2),
                      "rtf": round(rtf, 3)})
        print(f"  Run {i+1}: {n_steps} steps, {ms_step:.1f}ms/step, audio={audio_dur:.1f}s, time={elapsed:.1f}s, RTF={rtf:.3f}")

    avg_ms = np.mean([r["ms_per_step"] for r in runs])
    avg_rtf = np.mean([r["rtf"] for r in runs])

    results["models"][model_size] = {
        "runs": runs,
        "avg_ms_per_step": round(avg_ms, 1),
        "avg_rtf": round(avg_rtf, 3),
        "ttft_ms": round(ttft_ms, 1)
    }

    print(f"\n  >>> {model_size} Average: {avg_ms:.1f}ms/step, RTF={avg_rtf:.3f}, TTFT={ttft_ms:.0f}ms")

    del model, predictor_graph, talker_graph
    torch.cuda.empty_cache()

# Summary
print(f"\n{'='*50}")
print(f"SUMMARY — {gpu_name}")
print(f"{'='*50}")
print(f"{'Model':<8} {'ms/step':<10} {'RTF':<8} {'TTFT':<10} {'Real-time?':<10}")
print("-" * 46)
for size, data in results["models"].items():
    if "error" not in data:
        rt = "YES" if data["avg_rtf"] > 1.0 else "no"
        print(f"{size:<8} {data['avg_ms_per_step']:<10} {data['avg_rtf']:<8} {data['ttft_ms']:.0f}ms{'':>4} {rt:<10}")

gpu_slug = gpu_name.replace(" ", "_").replace("/", "_")
out_path = f"bench_results_{gpu_slug}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
PYEOF
