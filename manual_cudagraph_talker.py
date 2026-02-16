#!/usr/bin/env python3
"""
Manual CUDA graph capture for the talker's single-token decode step.

The talker has 28 transformer layers with:
- hidden_size=1024, num_heads=16, num_kv_heads=8, head_dim=128
- GQA (2 groups), interleaved multimodal RoPE (sections [24,20,20])
- SiLU-gated MLP (1024→3072→1024)

Strategy:
- Pre-allocate static KV cache padded to max_seq_len
- Pre-compute RoPE cos/sin for all positions
- Use attention mask to handle variable KV length
- Capture single-token decode as CUDA graph
- Update position and mask between replays
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def _rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _precompute_interleaved_mrope(inv_freq, max_seq_len, mrope_section, attention_scaling, device, dtype):
    """
    Pre-compute interleaved multimodal RoPE cos/sin for all positions.
    For text-only (talker decode), all 3 position dims are identical.
    
    Returns cos, sin of shape [max_seq_len, head_dim] in target dtype.
    """
    # position_ids: [3, 1, max_seq_len] - all same for text
    positions = torch.arange(max_seq_len, device=device).float()
    # inv_freq shape: [dim//2] where dim = head_dim//2 = 64
    
    # For interleaved mrope with sections [24, 20, 20]:
    # The rotary embedding produces cos/sin of shape [3, 1, seq, dim] 
    # where dim = sum(sections) = 64 (half of head_dim)
    # Then interleaved: take modality 0 for indices 0,3,6,...; modality 1 for 1,4,7,...; etc.
    
    # Compute freqs for each of the 3 modalities (all same positions for text)
    # inv_freq: [dim] where dim = 64
    # freqs: [3, seq, dim]
    inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1)  # [1, 64, 1]
    pos_expanded = positions[None, None, :].float()  # [1, 1, seq]
    freqs = (inv_freq_expanded @ pos_expanded).squeeze(0).transpose(0, 1)  # [seq, 64]
    
    # For text, all 3 modalities have same positions, so freqs are same for all 3
    # emb before interleaving: [3, seq, 64]
    freqs_3 = freqs.unsqueeze(0).expand(3, -1, -1)  # [3, seq, 64]
    
    # Apply interleaved mrope
    # mrope_section = [24, 20, 20], modality_num = 3
    # For interleaved: take chunks of sections, interleave modalities
    modality_num = len(mrope_section)
    
    # The interleaved version creates a single tensor where:
    # indices 0,3,6,... come from modality 0
    # indices 1,4,7,... come from modality 1  
    # indices 2,5,8,... come from modality 2
    # But since all modalities have same freqs for text, it doesn't matter!
    # The cos/sin will be the same regardless of interleaving.
    
    # Just compute standard: emb = cat(freqs, freqs) -> cos/sin
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq, 128]
    cos = (emb.cos() * attention_scaling).to(dtype)
    sin = (emb.sin() * attention_scaling).to(dtype)
    
    return cos, sin  # [max_seq_len, head_dim=128]


class ManualTalkerGraph:
    """
    Captures the talker's single-token decode step as a CUDA graph.
    
    The graph processes one token through 28 layers, using a static
    padded KV cache with attention masking.
    """
    
    def __init__(self, talker_model, talker_config, device='cuda:0', dtype=torch.bfloat16,
                 max_seq_len=512):
        self.device = device
        self.dtype = dtype
        self.num_layers = talker_config.num_hidden_layers  # 28
        self.hidden_size = talker_config.hidden_size  # 1024
        self.num_heads = talker_config.num_attention_heads  # 16
        self.num_kv_heads = talker_config.num_key_value_heads  # 8
        self.head_dim = getattr(talker_config, 'head_dim', self.hidden_size // self.num_heads)  # 128
        self.kv_groups = self.num_heads // self.num_kv_heads  # 2
        self.intermediate_size = talker_config.intermediate_size  # 3072
        self.rms_eps = talker_config.rms_norm_eps
        self.max_seq_len = max_seq_len
        self.rope_scaling = talker_config.rope_scaling
        
        # Extract model weights (references, not copies)
        self.layers = talker_model.layers
        self.norm = talker_model.norm
        self.rotary_emb = talker_model.rotary_emb
        
        # Pre-compute RoPE for all positions
        inv_freq = self.rotary_emb.inv_freq
        attention_scaling = self.rotary_emb.attention_scaling
        self.rope_cos, self.rope_sin = _precompute_interleaved_mrope(
            inv_freq, max_seq_len, self.rope_scaling['mrope_section'],
            attention_scaling, device, dtype
        )
        # Shape: [max_seq_len, head_dim] -> we'll index [pos] for each step
        
        # Static KV cache: [num_layers, 2(k/v), batch=1, kv_heads, max_seq, head_dim]
        self.kv_cache = torch.zeros(
            self.num_layers, 2, 1, self.num_kv_heads, max_seq_len, self.head_dim,
            dtype=dtype, device=device
        )
        
        # Static attention mask: [1, 1, 1, max_seq_len]
        # Filled with -inf, we unmask positions 0..current_pos
        self.attn_mask = torch.full(
            (1, 1, 1, max_seq_len), float('-inf'), dtype=dtype, device=device
        )
        
        # Static input/output buffers
        self.input_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        self.output_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        
        # Position index tensor (for scatter into KV cache) - shape [1] for index_copy_
        self.pos_idx = torch.zeros(1, dtype=torch.long, device=device)
        
        # Static RoPE buffers for current position (copied before graph replay)
        self.cur_cos = torch.zeros(1, 1, 1, self.head_dim, dtype=dtype, device=device)
        self.cur_sin = torch.zeros(1, 1, 1, self.head_dim, dtype=dtype, device=device)
        
        self.graph = None
        self.captured = False
    
    def _decode_step(self):
        """Single-token decode through all 28 layers."""
        h = self.input_buf  # [1, 1, 1024]
        
        # Use pre-copied RoPE values (set before graph replay)
        cos = self.cur_cos  # [1, 1, 1, head_dim]
        sin = self.cur_sin  # [1, 1, 1, head_dim]
        
        for li in range(self.num_layers):
            layer = self.layers[li]
            
            # Input layernorm
            residual = h
            h = layer.input_layernorm(h)
            
            # Attention projections
            attn = layer.self_attn
            q = attn.q_proj(h).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [1, 16, 1, 128]
            k = attn.k_proj(h).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, 8, 1, 128]
            v = attn.v_proj(h).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, 8, 1, 128]
            
            # QK norms
            q = attn.q_norm(q)
            k = attn.k_norm(k)
            
            # Apply RoPE
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            
            # Store in KV cache at current position
            # kv_cache: [num_layers, 2, 1, kv_heads, max_seq, head_dim]
            # k is [1, 8, 1, 128], cache is [layers, 2, 1, 8, max_seq, 128]
            # Use index_copy_ to write at pos_idx position (no CPU sync needed)
            # kv_cache[li, 0, 0] is [8, max_seq, 128], k[0] is [8, 1, 128]
            self.kv_cache[li, 0, 0].index_copy_(1, self.pos_idx, k[0])
            self.kv_cache[li, 1, 0].index_copy_(1, self.pos_idx, v[0])
            
            # Get full KV (padded to max_seq_len)
            k_full = self.kv_cache[li, 0]  # [1, 8, max_seq, 128]
            v_full = self.kv_cache[li, 1]  # [1, 8, max_seq, 128]
            
            # Expand KV heads for GQA
            k_full = k_full.repeat_interleave(self.kv_groups, dim=1)  # [1, 16, max_seq, 128]
            v_full = v_full.repeat_interleave(self.kv_groups, dim=1)  # [1, 16, max_seq, 128]
            
            # Scaled dot-product attention with mask
            attn_out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=self.attn_mask)
            attn_out = attn_out.transpose(1, 2).reshape(1, 1, self.num_heads * self.head_dim)
            h = residual + attn.o_proj(attn_out)
            
            # MLP
            residual = h
            h = layer.post_attention_layernorm(h)
            h = layer.mlp(h)
            h = residual + h
        
        self.output_buf.copy_(self.norm(h))
    
    @torch.inference_mode()
    def capture(self, prefill_len=100, num_warmup=3):
        """
        Capture CUDA graph for single-token decode.
        prefill_len: simulated prefill length for warmup (graph is position-independent).
        """
        print(f"Warming up talker graph ({num_warmup} runs)...")
        
        # Set up as if we've done prefill of prefill_len tokens
        self.attn_mask.fill_(float('-inf'))
        self.attn_mask[:, :, :, :prefill_len+1] = 0.0
        self.pos_idx[0] = prefill_len
        self.cur_cos[0, 0, 0] = self.rope_cos[prefill_len]
        self.cur_sin[0, 0, 0] = self.rope_sin[prefill_len]
        
        for _ in range(num_warmup):
            self._decode_step()
        torch.cuda.synchronize()
        
        print("Capturing CUDA graph for talker decode...")
        self.graph = torch.cuda.CUDAGraph()
        
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # Warmup in capture stream
            self._decode_step()
            torch.cuda.synchronize()
            
            with torch.cuda.graph(self.graph):
                self._decode_step()
        
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.captured = True
        print("Talker CUDA graph captured!")
    
    def reset(self, prefill_len: int):
        """Reset for new sequence after prefill of prefill_len tokens."""
        self.kv_cache.zero_()
        self.attn_mask.fill_(float('-inf'))
        # After prefill, positions 0..prefill_len-1 are filled
        # Next decode will be at position prefill_len
        self.attn_mask[:, :, :, :prefill_len] = 0.0
    
    def prefill_kv(self, past_key_values):
        """
        Copy HF DynamicCache from prefill into our static KV cache.
        past_key_values: DynamicCache with 28 layers of [1, 8, prefill_len, 128]
        """
        for li in range(self.num_layers):
            k, v = past_key_values[li]  # each [1, 8, seq_len, 128]
            seq_len = k.shape[2]
            self.kv_cache[li, 0, 0, :, :seq_len, :] = k[0]
            self.kv_cache[li, 1, 0, :, :seq_len, :] = v[0]
        # Set mask
        self.attn_mask.fill_(float('-inf'))
        self.attn_mask[:, :, :, :seq_len] = 0.0
        return seq_len
    
    @torch.inference_mode()
    def run(self, input_embeds: torch.Tensor, position: int) -> torch.Tensor:
        """
        Run one decode step.
        input_embeds: [1, 1, 1024]
        position: current sequence position
        Returns: [1, 1, 1024] hidden states
        """
        # Update static buffers
        self.input_buf.copy_(input_embeds)
        self.pos_idx[0] = position
        # Copy RoPE for this position into static buffers
        self.cur_cos[0, 0, 0] = self.rope_cos[position]
        self.cur_sin[0, 0, 0] = self.rope_sin[position]
        # Unmask current position (previous positions already unmasked)
        self.attn_mask[:, :, :, position] = 0.0
        
        # Replay graph
        self.graph.replay()
        
        return self.output_buf  # NOTE: this is the static buffer, caller should use immediately or clone


if __name__ == "__main__":
    import sys, time, numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qwen_tts import Qwen3TTSModel
    
    MODEL_PATH = './models/Qwen3-TTS-12Hz-0.6B-Base'
    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map='cuda:0', dtype=torch.bfloat16)
    
    talker = model.model.talker
    talker_model = talker.model
    
    import json
    with open(f'{MODEL_PATH}/config.json') as f:
        fc = json.load(f)
    from transformers import PretrainedConfig
    talker_config = PretrainedConfig(**fc['talker_config'])
    
    MAX_SEQ = 512
    
    with torch.inference_mode():
        mtg = ManualTalkerGraph(talker_model, talker_config, max_seq_len=MAX_SEQ)
        
        # Test: compare eager vs graph at position 150
        test_input = torch.randn(1, 1, 1024, device='cuda', dtype=torch.bfloat16)
        
        # Eager test
        mtg.attn_mask.fill_(float('-inf'))
        mtg.attn_mask[:, :, :, :151] = 0.0
        mtg.pos_idx[0] = 150
        mtg.cur_cos[0, 0, 0] = mtg.rope_cos[150]
        mtg.cur_sin[0, 0, 0] = mtg.rope_sin[150]
        mtg.input_buf.copy_(test_input)
        mtg._decode_step()
        eager_out = mtg.output_buf.clone()
        
        # Benchmark eager
        N = 20
        times = []
        for _ in range(N):
            torch.cuda.synchronize(); t0 = time.time()
            mtg.input_buf.copy_(test_input)
            mtg.cur_cos[0, 0, 0] = mtg.rope_cos[150]
            mtg.cur_sin[0, 0, 0] = mtg.rope_sin[150]
            mtg._decode_step()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        print(f"Eager: {np.mean(times)*1000:.1f}ms avg, {np.min(times)*1000:.1f}ms min")
        
        # Capture graph
        mtg.capture(prefill_len=150, num_warmup=3)
        
        # Benchmark graph
        times = []
        for _ in range(N):
            torch.cuda.synchronize(); t0 = time.time()
            out = mtg.run(test_input, position=150)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        print(f"Graph: {np.mean(times)*1000:.1f}ms avg, {np.min(times)*1000:.1f}ms min")
        print(f"Max diff: {(eager_out - out).abs().max().item()}")
