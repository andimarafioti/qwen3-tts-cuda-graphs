#!/usr/bin/env python3
"""
CUDA graph capture for the talker's single-token decode step,
using transformers StaticCache.

The talker has 28 transformer layers. Instead of reimplementing the
forward pass manually, we use the model's own forward with StaticCache.
The StaticCache provides fixed-size KV tensors compatible with CUDA graphs.

Strategy:
- Use transformers StaticCache for KV cache management
- Use the model's forward method (handles mask, RoPE, attention internally)
- Capture the single-token decode as a CUDA graph
- Update cache_position buffer between replays
"""
import torch
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


class TalkerGraph:
    """
    Captures the talker's single-token decode step as a CUDA graph,
    using the model's own forward with transformers StaticCache.
    """

    def __init__(self, talker_model, talker_config, device='cuda:0', dtype=torch.bfloat16,
                 max_seq_len=512):
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.hidden_size = talker_config.hidden_size
        self.num_layers = talker_config.num_hidden_layers

        # Keep reference to the inner model (transformer backbone)
        self.model = talker_model

        # Transformers StaticCache — handles index_copy_ and fixed-size KV internally
        self.static_cache = StaticCache(config=talker_config, max_cache_len=max_seq_len)

        # Static I/O buffers for CUDA graph
        self.input_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        self.output_buf = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)

        # Cache position buffer — updated before each graph replay
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)

        self.graph = None
        self.captured = False
        self.attn_mask = None
        self.attn_mask_table = None

    def _init_cache_layers(self):
        """Force lazy initialization of StaticCache layers before graph capture."""
        config = self.model.config
        num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=self.dtype, device=self.device)
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _build_attention_masks(self):
        dummy = torch.zeros(1, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        max_len = self.max_seq_len
        self.attn_mask_table = [None] * max_len

        mask_fn = create_causal_mask if self.model.config.sliding_window is None else create_sliding_window_causal_mask

        for i in range(max_len):
            pos = torch.tensor([i], device=self.device)
            full = mask_fn(
                config=self.model.config,
                input_embeds=dummy,
                attention_mask=None,
                cache_position=pos,
                past_key_values=self.static_cache,
            )
            self.attn_mask_table[i] = full

        self.attn_mask = self.attn_mask_table[0].clone()

    def _set_attention_mask(self, position: int):
        self.attn_mask.copy_(self.attn_mask_table[position])

    def _decode_step(self):
        """Single-token decode through the model's forward."""
        out = self.model(
            inputs_embeds=self.input_buf,
            attention_mask=self.attn_mask,
            past_key_values=self.static_cache,
            cache_position=self.cache_position,
            use_cache=True,
        )
        self.output_buf.copy_(out.last_hidden_state)

    @torch.inference_mode()
    def capture(self, prefill_len=100, num_warmup=3):
        """
        Capture CUDA graph for single-token decode.
        prefill_len: simulated prefill length for warmup (graph is position-independent).
        """
        print(f"Warming up talker graph ({num_warmup} runs)...")

        # Force cache initialization before graph capture
        self._init_cache_layers()
        self._build_attention_masks()

        # Set cache_position for warmup
        self.cache_position[0] = prefill_len
        self._set_attention_mask(prefill_len)

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
        """Reset cache for new sequence."""
        self.static_cache.reset()

    def prefill_kv(self, past_key_values):
        """
        Copy HF DynamicCache from prefill into our StaticCache.
        past_key_values: DynamicCache with num_layers layers of [1, kv_heads, seq_len, head_dim]
        """
        self.static_cache.reset()
        seq_len = 0
        for li in range(self.num_layers):
            k, v = past_key_values[li]  # each [1, kv_heads, seq_len, head_dim]
            seq_len = k.shape[2]
            cache_pos = torch.arange(seq_len, device=self.device)
            self.static_cache.update(k, v, li, {"cache_position": cache_pos})
        return seq_len

    @torch.inference_mode()
    def run(self, input_embeds: torch.Tensor, position: int) -> torch.Tensor:
        """
        Run one decode step.
        input_embeds: [1, 1, hidden_size]
        position: current sequence position
        Returns: [1, 1, hidden_size] hidden states
        """
        self.input_buf.copy_(input_embeds)
        self.cache_position[0] = position
        self._set_attention_mask(position)

        self.graph.replay()

        return self.output_buf  # static buffer — caller should use immediately or clone
