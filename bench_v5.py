#!/usr/bin/env python3
"""Benchmark v5: CUDA graphs for both predictor and talker."""
import torch, time, sys, json, os, numpy as np

# Ensure repo root is on path for local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_tts import Qwen3TTSModel
from transformers import PretrainedConfig
from qwen3_tts_cuda_graphs.manual_cudagraph_predictor import ManualPredictorGraph
from qwen3_tts_cuda_graphs.manual_cudagraph_talker import ManualTalkerGraph
from qwen3_tts_cuda_graphs.fast_generate_v5 import fast_generate_v5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SIZE = os.environ.get('MODEL_SIZE', '0.6B')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', f'Qwen3-TTS-12Hz-{MODEL_SIZE}-Base')
text = "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it. The robots have officially won. Please remain calm."
ref_audio = os.path.join(SCRIPT_DIR, 'ref_audio.wav')
ref_text = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."
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

# TTFA (Time to First Audio) measurement
print("\nMeasuring TTFA (5 runs)...")
ttfa_results = []
for i in range(5):
    talker.rope_deltas = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    codec_ids_ttfa, timing_ttfa = fast_generate_v5(
        talker, tie, tam, tth, tpe, config, mpg, mtg,
        temperature=0.9, top_k=50, do_sample=True, max_new_tokens=1,
    )
    torch.cuda.synchronize()
    ttfa_ms = (time.perf_counter() - t0) * 1000
    ttfa_results.append(ttfa_ms)
    print(f"  Run {i+1}: {ttfa_ms:.1f}ms (prefill={timing_ttfa['prefill_ms']:.1f}ms)")

ttfa_mean = np.mean(ttfa_results)
ttfa_std = np.std(ttfa_results)
print(f"  TTFA: {ttfa_mean:.1f}ms ± {ttfa_std:.1f}ms")

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
    print(f"\n=== {MODEL_SIZE} Average: {avg_ms:.1f}ms/step, RTF={avg_rtf:.3f}, TTFA={ttfa_mean:.0f}ms ===")
    
    # Decode audio from last run
    try:
        print(f"\nSaving audio from last run...")
        speech_tokenizer = model.model.speech_tokenizer
        audio_list, sr = speech_tokenizer.decode({"audio_codes": codec_ids.unsqueeze(0)})
        import soundfile as sf
        out_wav = os.path.join(SCRIPT_DIR, f'sample_{MODEL_SIZE}.wav')
        sf.write(out_wav, audio_list[0].flatten(), sr)
        print(f"Saved to {out_wav}")
    except Exception as e:
        print(f"Audio decode failed: {e}")
