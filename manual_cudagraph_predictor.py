#!/usr/bin/env python3
"""
Manual CUDA graph capture for the code predictor's 15-step decode loop.

The predictor generates 15 codebooks autoregressively:
- Step 0: prefill with 2 tokens (past_hidden + first_codebook_embed), get logits[0]
- Steps 1-14: decode 1 token at a time using previous codebook token's embedding

Total: 15 forward passes through 5-layer transformer (2-token + 14×1-token)

Strategy for CUDA graph capture:
1. Pre-allocate static KV cache [5 layers, 2(k/v), 17 max_seq, 8 kv_heads, 64 head_dim]
2. Use sequence length counter (updated via in-place ops)
3. Unroll the full loop so shapes are deterministic
4. Use SDPA for attention (handles variable-length via slicing)

Note: We capture the FULL 15-step predictor as a single CUDA graph.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def _rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    # cos, sin: [1, seq_len, head_dim] -> unsqueeze for heads dim
    cos = cos.unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [1, 1, seq_len, head_dim]
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class ManualPredictorGraph:
    """
    Captures the full predictor 15-step loop as a CUDA graph.
    
    Usage:
        mpg = ManualPredictorGraph(code_predictor, talker_config)
        mpg.capture()
        codebook_tokens = mpg.run(pred_input)  # pred_input: [1, 2, H]
    """
    
    def __init__(self, code_predictor, pred_config, talker_hidden_size, device='cuda:0', dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.num_layers = pred_config.num_hidden_layers
        self.hidden_size = pred_config.hidden_size
        self.num_heads = pred_config.num_attention_heads
        self.num_kv_heads = pred_config.num_key_value_heads
        self.head_dim = getattr(pred_config, 'head_dim', self.hidden_size // self.num_heads)
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.num_code_groups = pred_config.num_code_groups
        self.num_codebooks = self.num_code_groups - 1  # 15
        self.max_seq = 2 + self.num_codebooks  # 17
        self.rms_eps = pred_config.rms_norm_eps
        
        # Extract model weights (keep references, don't copy)
        cp = code_predictor
        self.small_to_mtp = cp.small_to_mtp_projection
        self.layers = cp.model.layers
        self.norm = cp.model.norm
        self.lm_heads = cp.lm_head  # ModuleList[15]
        self.codec_embeds = cp.model.codec_embedding  # ModuleList[15] for codebooks 0-14
        self.rotary_emb = cp.model.rotary_emb
        
        # Static buffers
        self.kv_cache = torch.zeros(
            self.num_layers, 2, self.max_seq, self.num_kv_heads, self.head_dim,
            dtype=dtype, device=device
        )
        
        # Input/output buffers
        self.input_buf = torch.zeros(1, 2, talker_hidden_size, dtype=dtype, device=device)
        self.output_tokens = torch.zeros(self.num_codebooks, dtype=torch.long, device=device)
        
        self.graph = None
        self.captured = False
    
    def _single_layer_forward(self, hidden_states, layer_idx, seq_start, seq_len):
        """Forward through one transformer layer, updating KV cache."""
        layer = self.layers[layer_idx]
        
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        
        attn = layer.self_attn
        bsz, qlen, _ = hidden_states.shape
        
        q = attn.q_proj(hidden_states).view(bsz, qlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(bsz, qlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(bsz, qlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if hasattr(attn, 'q_norm') and attn.q_norm is not None:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        
        # RoPE - compute for positions [seq_start, seq_start+seq_len)
        positions = torch.arange(seq_start, seq_start + seq_len, device=self.device).unsqueeze(0)  # [1, seq_len]
        cos, sin = self.rotary_emb(hidden_states, positions)
        q, k = _apply_rope(q, k, cos, sin)
        
        # Store in KV cache: k is [1, num_kv_heads, seq_len, head_dim], cache is [seq, kv_heads, head_dim]
        self.kv_cache[layer_idx, 0, seq_start:seq_start+seq_len] = k[0].transpose(0, 1)
        self.kv_cache[layer_idx, 1, seq_start:seq_start+seq_len] = v[0].transpose(0, 1)
        
        # Get full KV up to current position: [seq, kv_heads, head_dim] -> [1, kv_heads, seq, head_dim]
        total_len = seq_start + seq_len
        k_full = self.kv_cache[layer_idx, 0, :total_len].transpose(0, 1).unsqueeze(0)
        v_full = self.kv_cache[layer_idx, 1, :total_len].transpose(0, 1).unsqueeze(0)
        
        # Expand KV heads for GQA
        if self.kv_groups > 1:
            k_full = k_full.repeat_interleave(self.kv_groups, dim=1)
            v_full = v_full.repeat_interleave(self.kv_groups, dim=1)
        
        attn_out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=(qlen > 1))
        attn_out = attn_out.transpose(1, 2).reshape(bsz, qlen, -1)
        hidden_states = attn.o_proj(attn_out)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def _full_loop(self):
        """The full 15-step predictor loop on static buffers."""
        # Project input
        h = self.small_to_mtp(self.input_buf)  # [1, 2, hidden]
        
        # Prefill: 2 tokens through all layers
        for li in range(self.num_layers):
            h = self._single_layer_forward(h, li, seq_start=0, seq_len=2)
        h = self.norm(h)
        
        # First codebook: logits from last position
        logits = self.lm_heads[0](h[:, -1:, :])  # [1, 1, vocab]
        tok = torch.argmax(logits[:, 0, :], dim=-1)  # [1]
        self.output_tokens[0] = tok[0]
        
        # Remaining 14 codebooks
        for cb_idx in range(1, self.num_codebooks):
            # Embed previous token
            emb = self.codec_embeds[cb_idx - 1](tok.unsqueeze(0))  # [1, 1, codec_hidden]
            emb = self.small_to_mtp(emb)  # [1, 1, hidden]
            
            # Single-token decode through all layers
            h = emb
            for li in range(self.num_layers):
                h = self._single_layer_forward(h, li, seq_start=1+cb_idx, seq_len=1)
            h = self.norm(h)
            
            logits = self.lm_heads[cb_idx](h[:, -1:, :])
            tok = torch.argmax(logits[:, 0, :], dim=-1)
            self.output_tokens[cb_idx] = tok[0]
        
        return self.output_tokens
    
    @torch.inference_mode()
    def capture(self, num_warmup=3):
        """Warmup and capture the CUDA graph."""
        print(f"Warming up predictor ({num_warmup} runs)...")
        for _ in range(num_warmup):
            self.kv_cache.zero_()
            self._full_loop()
        torch.cuda.synchronize()
        
        print("Capturing CUDA graph for predictor...")
        self.kv_cache.zero_()
        
        # Side-pool: capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.graph = torch.cuda.CUDAGraph()
            # Warmup in capture stream
            self.kv_cache.zero_()
            self._full_loop()
            torch.cuda.synchronize()
            
            self.kv_cache.zero_()
            with torch.cuda.graph(self.graph):
                self._full_loop()
        
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.captured = True
        print("CUDA graph captured!")
    
    @torch.inference_mode()
    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        """
        Run the captured graph.
        pred_input: [1, 2, talker_hidden_size] (past_hidden cat first_codebook_embed)
        Returns: [15] long tensor of codebook tokens
        """
        self.input_buf.copy_(pred_input)
        self.kv_cache.zero_()
        self.graph.replay()
        return self.output_tokens.clone()


if __name__ == "__main__":
    import sys, time, json, numpy as np
    sys.path.insert(0, '/home/andi/Documents/Qwen3-TTS-streaming')
    from qwen_tts import Qwen3TTSModel
    
    MODEL_PATH = './models/Qwen3-TTS-12Hz-0.6B-Base'
    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map='cuda:0', dtype=torch.bfloat16)
    
    predictor = model.model.talker.code_predictor
    
    with open(f'{MODEL_PATH}/config.json') as f:
        fc = json.load(f)
    
    from transformers import PretrainedConfig
    pred_config = PretrainedConfig(**fc['talker_config']['code_predictor_config'])
    talker_hidden = fc['talker_config']['hidden_size']
    
    with torch.inference_mode():
        mpg = ManualPredictorGraph(predictor, pred_config, talker_hidden)
        
        # Test eager first
        test_input = torch.randn(1, 2, talker_hidden, device='cuda', dtype=torch.bfloat16)
        mpg.input_buf.copy_(test_input)
        mpg.kv_cache.zero_()
        result_eager = mpg._full_loop().clone()
        print(f"Eager result: {result_eager}")
        
        # Benchmark eager
        N = 20
        times = []
        for _ in range(N):
            mpg.kv_cache.zero_()
            torch.cuda.synchronize(); t0 = time.time()
            mpg._full_loop()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        print(f"Eager: {np.mean(times)*1000:.1f}ms avg, {np.min(times)*1000:.1f}ms min")
        
        # Capture graph
        mpg.capture(num_warmup=3)
        
        # Benchmark graph
        times = []
        for _ in range(N):
            torch.cuda.synchronize(); t0 = time.time()
            result = mpg.run(test_input)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        print(f"Graph: {np.mean(times)*1000:.1f}ms avg, {np.min(times)*1000:.1f}ms min")
        print(f"Graph result: {result}")
