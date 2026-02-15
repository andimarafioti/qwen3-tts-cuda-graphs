#!/usr/bin/env python3
"""Measure Time to First Audio Chunk (TTFT) for Qwen3-TTS v5 pipeline."""
import torch, time, sys, json, numpy as np
sys.path.insert(0, '/home/andi/Documents/Qwen3-TTS-streaming')
sys.path.insert(0, '/home/andi/Documents/qwen3-tts/cuda_graphs')

from qwen_tts import Qwen3TTSModel
from transformers import PretrainedConfig
from manual_cudagraph_predictor import ManualPredictorGraph
from manual_cudagraph_talker import ManualTalkerGraph
from fast_generate_v5 import fast_generate_v5, _sample
import torch.nn.functional as F

MODEL_PATH = '/home/andi/Documents/qwen3-tts/models/Qwen3-TTS-12Hz-0.6B-Base'
text = 'The quick brown fox jumps over the lazy dog. It was a sunny afternoon and the birds were singing in the trees.'
ref_audio = '/home/andi/Documents/qwen3-tts/ref_audio.wav'
ref_text = 'This is a reference audio sample.'
MAX_SEQ = 2048

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map='cuda:0', dtype=torch.bfloat16, local_files_only=True)
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
predictor = talker.code_predictor
mpg = ManualPredictorGraph(predictor, pred_config, fc['talker_config']['hidden_size'])
mpg.capture(num_warmup=3)
mtg = ManualTalkerGraph(talker.model, talker_cfg, max_seq_len=MAX_SEQ)
mtg.capture(prefill_len=tie.shape[1], num_warmup=3)

# Speech tokenizer for codec decode
speech_tokenizer = model.model.speech_tokenizer

# Warmup
print("\nWarmup run...")
talker.rope_deltas = None
codec_ids, _ = fast_generate_v5(
    talker, tie, tam, tth, tpe, config, mpg, mtg,
    temperature=0.9, top_k=50, do_sample=True, max_new_tokens=20,
)
# Warmup codec decode too
if codec_ids is not None:
    speech_tokenizer.decode({"audio_codes": codec_ids[:1]})
torch.cuda.synchronize()

@torch.inference_mode()
def measure_ttft():
    """Run one TTFT measurement: prefill + 1 decode step + codec decode of first chunk."""
    talker.rope_deltas = None
    
    eos_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    vocab_size = config.vocab_size
    device = tie.device

    suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for i in range(vocab_size - 1024, vocab_size):
        if i != eos_id:
            suppress_mask[i] = True

    talker_codec_embed = talker.get_input_embeddings()
    predictor_codec_embeds = predictor.get_input_embeddings()
    talker_codec_head = talker.codec_head

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # === PREFILL ===
    out = talker.forward(
        inputs_embeds=tie, attention_mask=tam, use_cache=True,
        output_hidden_states=True, return_dict=True,
        trailing_text_hidden=tth, tts_pad_embed=tpe,
        generation_step=None, past_hidden=None, past_key_values=None,
    )
    talker_past_kv = out.past_key_values
    past_hidden = out.past_hidden
    gen_step = out.generation_step
    logits = out.logits[:, -1, :]
    token = _sample(logits, 0.9, 50, True, suppress_mask)
    prefill_len = mtg.prefill_kv(talker_past_kv)

    torch.cuda.synchronize()
    t_prefill = time.perf_counter()

    # === FIRST DECODE STEP ===
    last_id_hidden = talker_codec_embed(token.unsqueeze(1))
    pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
    codebook_token_ids = mpg.run(pred_input)

    all_cb = torch.cat([token.view(1), codebook_token_ids])  # [16]

    # Build input for talker (not needed for TTFT but included for completeness of the step)
    codec_hiddens = [last_id_hidden]
    for i in range(num_code_groups - 1):
        codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0)))
    inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
    if gen_step < tth.shape[1]:
        inputs_embeds = inputs_embeds + tth[:, gen_step].unsqueeze(1)
    else:
        inputs_embeds = inputs_embeds + tpe
    
    hidden_states = mtg.run(inputs_embeds, position=prefill_len)

    torch.cuda.synchronize()
    t_decode1 = time.perf_counter()

    # === CODEC DECODE (first token → PCM audio) ===
    # audio_codes shape for 12Hz: [batch, num_frames, num_codebooks] = [1, 1, 16]
    audio, sr = speech_tokenizer.decode({"audio_codes": all_cb.unsqueeze(0)})

    torch.cuda.synchronize()
    t_audio = time.perf_counter()

    return {
        'prefill_ms': (t_prefill - t0) * 1000,
        'first_decode_ms': (t_decode1 - t_prefill) * 1000,
        'codec_decode_ms': (t_audio - t_decode1) * 1000,
        'total_ttft_ms': (t_audio - t0) * 1000,
    }


# === Run measurements ===
print("\n" + "="*60)
print("TTFT Measurement (5 runs)")
print("="*60)

results = []
for i in range(5):
    r = measure_ttft()
    results.append(r)
    print(f"Run {i+1}: prefill={r['prefill_ms']:.1f}ms, decode1={r['first_decode_ms']:.1f}ms, "
          f"codec={r['codec_decode_ms']:.1f}ms, TTFT={r['total_ttft_ms']:.1f}ms")

print("\n" + "="*60)
print("Summary (mean ± std over 5 runs)")
print("="*60)
for key in ['prefill_ms', 'first_decode_ms', 'codec_decode_ms', 'total_ttft_ms']:
    vals = [r[key] for r in results]
    label = key.replace('_', ' ').replace('ms', '').strip()
    print(f"  {label:>20s}: {np.mean(vals):7.1f} ± {np.std(vals):5.1f} ms")
