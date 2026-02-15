#!/usr/bin/env python3
"""Benchmark v5: CUDA graphs for both predictor and talker."""
import torch, time, sys, json, numpy as np
sys.path.insert(0, '/home/andi/Documents/Qwen3-TTS-streaming')
sys.path.insert(0, '/home/andi/Documents/qwen3-tts/cuda_graphs')

from qwen_tts import Qwen3TTSModel
from transformers import PretrainedConfig
from manual_cudagraph_predictor import ManualPredictorGraph
from manual_cudagraph_talker import ManualTalkerGraph
from fast_generate_v5 import fast_generate_v5

MODEL_PATH = './models/Qwen3-TTS-12Hz-0.6B-Base'
text = 'The quick brown fox jumps over the lazy dog. It was a sunny afternoon and the birds were singing in the trees.'
ref_audio = '/home/andi/Documents/qwen3-tts/ref_audio.wav'
ref_text = 'This is a reference audio sample.'
MAX_SEQ = 2048

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map='cuda:0', dtype=torch.bfloat16)
talker = model.model.talker
config = model.model.config.talker_config

with open(f'{MODEL_PATH}/config.json') as f:
    fc = json.load(f)
pred_config = PretrainedConfig(**fc['talker_config']['code_predictor_config'])
talker_cfg = PretrainedConfig(**fc['talker_config'])

@torch.inference_mode()
def build_inputs():
    input_texts = [f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"]
    input_ids = []
    for t in input_texts:
        inp = model.processor(text=t, return_tensors="pt", padding=True)
        iid = inp["input_ids"].to(model.device)
        input_ids.append(iid.unsqueeze(0) if iid.dim() == 1 else iid)
    prompt_items = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    vcp = model._prompt_items_to_voice_clone_prompt(prompt_items)
    ref_ids = []
    rt = prompt_items[0].ref_text
    if rt:
        ref_ids.append(model._tokenize_texts([f"<|im_start|>assistant\n{rt}<|im_end|>\n"])[0])
    m = model.model
    return m._build_talker_inputs(
        input_ids=input_ids, instruct_ids=None, ref_ids=ref_ids,
        voice_clone_prompt=vcp, languages=["Auto"], speakers=None, non_streaming_mode=False,
    )

print("Building inputs...")
tie, tam, tth, tpe = build_inputs()
print(f"Input embeds shape: {tie.shape}, prefill_len: {tie.shape[1]}")

print("\nSetting up CUDA graphs...")

# Predictor graph
predictor = talker.code_predictor
mpg = ManualPredictorGraph(predictor, pred_config, fc['talker_config']['hidden_size'])
mpg.capture(num_warmup=3)

# Talker graph
mtg = ManualTalkerGraph(talker.model, talker_cfg, max_seq_len=MAX_SEQ)
mtg.capture(prefill_len=tie.shape[1], num_warmup=3)

# Warmup generation
print("\nWarmup run...")
talker.rope_deltas = None
codec_ids, timing = fast_generate_v5(
    talker, tie, tam, tth, tpe, config, mpg, mtg,
    temperature=0.9, top_k=50, do_sample=True, max_new_tokens=20,
)
print(f"Warmup: {timing['steps']} steps, {timing['ms_per_step']:.1f}ms/step")

# Benchmark
print("\nBenchmark runs...")
results = []
for run in range(3):
    talker.rope_deltas = None
    
    codec_ids, timing = fast_generate_v5(
        talker, tie, tam, tth, tpe, config, mpg, mtg,
        temperature=0.9, top_k=50, do_sample=True, max_new_tokens=2048,
    )
    
    if codec_ids is not None:
        n_steps = timing['steps']
        audio_duration = n_steps / 12.0  # 12 Hz codec
        total_time = timing['prefill_ms']/1000 + timing['decode_s']
        rtf = audio_duration / total_time
        
        print(f"Run {run+1}: {n_steps} steps, {timing['ms_per_step']:.1f}ms/step, "
              f"audio={audio_duration:.1f}s, time={total_time:.1f}s, RTF={rtf:.3f}")
        results.append({
            'steps': n_steps,
            'ms_per_step': timing['ms_per_step'],
            'rtf': rtf,
            'prefill_ms': timing['prefill_ms'],
            'decode_s': timing['decode_s'],
        })

if results:
    avg_ms = np.mean([r['ms_per_step'] for r in results])
    avg_rtf = np.mean([r['rtf'] for r in results])
    print(f"\n=== Average: {avg_ms:.1f}ms/step, RTF={avg_rtf:.3f} ===")
    
    # Also try to decode audio
    try:
        best = results[0]
        print(f"\nSaving audio from first run...")
        from qwen_tts.utils.codec import Codec
        codec = Codec(f'{MODEL_PATH}/codec', 'cuda:0')
        audio = codec.decode(codec_ids.unsqueeze(0))
        import soundfile as sf
        sf.write('/tmp/qwen_tts_v5.wav', audio.cpu().numpy().flatten(), 24000)
        print("Saved to /tmp/qwen_tts_v5.wav")
    except Exception as e:
        print(f"Audio decode failed: {e}")
