#!/usr/bin/env python3
"""
Non-streaming generation loop using CUDA graphs for both predictor and talker.
"""
import torch
import torch.nn.functional as F
import time
from typing import Optional, Tuple
from .predictor_graph import PredictorGraph
from .talker_graph import TalkerGraph


@torch.inference_mode()
def fast_generate(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: PredictorGraph,
    talker_graph: TalkerGraph,
    max_new_tokens: int = 2048,
    temperature: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
) -> Tuple[Optional[torch.Tensor], dict]:
    """
    Fast autoregressive generation with CUDA-graphed predictor and talker.
    """
    eos_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    vocab_size = config.vocab_size
    device = talker_input_embeds.device
    
    suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for i in range(vocab_size - 1024, vocab_size):
        if i != eos_id:
            suppress_mask[i] = True
    
    predictor = talker.code_predictor
    talker_model = talker.model
    talker_codec_embed = talker.get_input_embeddings()
    talker_codec_head = talker.codec_head
    predictor_codec_embeds = predictor.get_input_embeddings()
    
    # === PREFILL (still uses HF forward for variable-length prefill) ===
    t_start = time.time()
    
    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
    )
    
    talker_past_kv = out.past_key_values
    past_hidden = out.past_hidden
    gen_step = out.generation_step
    
    logits = out.logits[:, -1, :]
    token = _sample(logits, temperature, top_k, do_sample, suppress_mask)
    
    # Copy prefill KV cache into talker graph's static cache
    prefill_len = talker_graph.prefill_kv(talker_past_kv)
    
    torch.cuda.synchronize()
    t_prefill = time.time() - t_start
    
    # === DECODE LOOP ===
    t_decode_start = time.time()
    all_codec_ids = []
    
    for step_idx in range(max_new_tokens):
        if token.item() == eos_id:
            break
        
        # --- CUDA-Graphed Code Predictor ---
        last_id_hidden = talker_codec_embed(token.unsqueeze(1))  # [1, 1, H]
        pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)  # [1, 2, H]
        codebook_token_ids = predictor_graph.run(pred_input)  # [15] long tensor
        
        # Build full codec: [first_cb, cb1, ..., cb15]
        all_cb = torch.cat([token.view(1), codebook_token_ids])  # [16]
        all_codec_ids.append(all_cb.detach())
        
        # --- Build input embedding for talker ---
        codec_hiddens = [last_id_hidden]
        for i in range(num_code_groups - 1):
            codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0)))
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
        
        if gen_step < trailing_text_hiddens.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hiddens[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed
        
        # --- CUDA-Graphed Talker decode step ---
        current_pos = prefill_len + step_idx
        if current_pos >= talker_graph.max_seq_len - 1:
            # Stop if we exceed max_seq_len
            break
        
        hidden_states = talker_graph.run(inputs_embeds, position=current_pos)
        # hidden_states is the static output buffer - use it immediately
        
        logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)
        
        if repetition_penalty != 1.0 and len(all_codec_ids) > 0:
            n_recent = min(50, len(all_codec_ids))
            recent = torch.stack([c[0] for c in all_codec_ids[-n_recent:]])
            unique_toks = recent.unique()
            tok_logits = logits[0, 0, unique_toks]
            logits[0, 0, unique_toks] = torch.where(
                tok_logits > 0, tok_logits / repetition_penalty, tok_logits * repetition_penalty
            )
        
        token = _sample(logits.squeeze(0), temperature, top_k, do_sample, suppress_mask)
        past_hidden = hidden_states[:, -1:, :].clone()  # clone since it's the static buffer
        gen_step += 1
    
    torch.cuda.synchronize()
    t_decode = time.time() - t_decode_start
    
    n_steps = len(all_codec_ids)
    timing = {
        'prefill_ms': t_prefill * 1000,
        'decode_s': t_decode,
        'steps': n_steps,
        'ms_per_step': (t_decode / n_steps * 1000) if n_steps > 0 else 0,
        'steps_per_s': (n_steps / t_decode) if t_decode > 0 else 0,
    }
    
    if all_codec_ids:
        return torch.stack(all_codec_ids), timing
    return None, timing


def _sample(logits, temperature, top_k, do_sample, suppress_mask):
    logits = logits.clone()
    logits[..., suppress_mask] = float('-inf')
    if not do_sample:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < topk_vals[..., -1:]] = float('-inf')
    return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
