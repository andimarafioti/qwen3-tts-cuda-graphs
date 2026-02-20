#!/bin/bash
# Benchmark Qwen3-TTS with CUDA Graphs
# Usage: ./benchmark.sh [0.6B|1.7B|both|custom]
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

MODEL="${1:-both}"
PY="$DIR/.venv/bin/python"

if [ ! -f "$PY" ]; then
    echo "ERROR: venv not found. Run ./setup.sh first."
    exit 1
fi

$PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: PyTorch with CUDA required. Check your venv."
    exit 1
}

GPU_NAME=$($PY -c 'import torch; print(torch.cuda.get_device_name(0))')
echo "=== Faster Qwen3-TTS Benchmark ==="
echo "GPU: $GPU_NAME"
echo "PyTorch: $($PY -c 'import torch; print(torch.__version__)')"
echo "CUDA: $($PY -c 'import torch; print(torch.version.cuda)')"
echo ""

run_model() {
    local size=$1
    echo "--- Benchmarking $size ---"
    MODEL_SIZE="$size" $PY "$DIR/benchmarks/throughput.py"
    echo ""
}

run_custom() {
    local size=$1
    echo "--- Benchmarking $size (CustomVoice) ---"
    MODEL_SIZE="$size" $PY "$DIR/benchmarks/custom_voice.py"
    echo ""
}

case "$MODEL" in
    0.6B) run_model "0.6B" ;;
    1.7B) run_model "1.7B" ;;
    custom)
        run_custom "0.6B"
        run_custom "1.7B"
        ;;
    both)
        run_model "0.6B"
        run_model "1.7B"
        ;;
    *)
        echo "Usage: ./benchmark.sh [0.6B|1.7B|both|custom]"
        exit 1
        ;;
esac

echo "Done. Results saved as bench_results_*.json"
