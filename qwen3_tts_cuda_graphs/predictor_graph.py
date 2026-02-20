#!/usr/bin/env python3
"""
CUDA graph capture for the code predictor's 15-step decode loop,
using transformers StaticCache.

The predictor generates 15 codebooks autoregressively:
- Step 0: prefill with 2 tokens (past_hidden + first_codebook_embed), get logits[0]
- Steps 1-14: decode 1 token at a time using previous codebook token's embedding

Strategy:
- Use transformers StaticCache for KV cache management
- Use the predictor's inner model forward (handles mask, RoPE, attention internally)
- Unroll the full 15-step loop for deterministic shapes
- Capture the entire loop as a single CUDA graph
"""
import torch
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


class PredictorGraph:
    """
    Captures the full predictor 15-step loop as a CUDA graph,
    using the model's forward with transformers StaticCache.

    Usage:
        mpg = PredictorGraph(code_predictor, pred_config, talker_hidden_size)
        mpg.capture()
        codebook_tokens = mpg.run(pred_input)  # pred_input: [1, 2, H]
    """

    def __init__(self, code_predictor, pred_config, talker_hidden_size, device='cuda:0', dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.num_layers = pred_config.num_hidden_layers
        self.hidden_size = pred_config.hidden_size
        self.num_code_groups = pred_config.num_code_groups
        self.num_codebooks = self.num_code_groups - 1  # 15
        self.max_seq = 2 + self.num_codebooks  # 17

        # Extract model components (references, not copies)
        cp = code_predictor
        self.small_to_mtp = cp.small_to_mtp_projection
        self.pred_model = cp.model  # Inner transformer model (5 layers)
        self.lm_heads = cp.lm_head  # ModuleList[15]
        self.codec_embeds = cp.model.codec_embedding  # ModuleList[15]
        self.has_sliding_layers = "sliding_attention" in getattr(self.pred_model.config, "layer_types", [])

        # Transformers StaticCache for the predictor
        self.static_cache = StaticCache(config=pred_config, max_cache_len=self.max_seq)

        # Pre-allocate cache_position tensors for each step (avoids CPU→GPU in graph)
        self.prefill_cache_pos = torch.arange(2, device=device)
        self.decode_cache_positions = [
            torch.tensor([2 + i], device=device) for i in range(self.num_codebooks - 1)
        ]

        # I/O buffers
        self.input_buf = torch.zeros(1, 2, talker_hidden_size, dtype=dtype, device=device)
        self.output_tokens = torch.zeros(self.num_codebooks, dtype=torch.long, device=device)

        self.graph = None
        self.captured = False
        self.prefill_attn = None
        self.decode_attn = None

    def _init_cache_layers(self):
        """Force lazy initialization of StaticCache layers before graph capture."""
        config = self.pred_model.config
        num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=self.dtype, device=self.device)
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _make_attn_mask(self, input_embeds: torch.Tensor, cache_position: torch.Tensor):
        mask = create_causal_mask(
            config=self.pred_model.config,
            input_embeds=input_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=self.static_cache,
        )
        if self.has_sliding_layers:
            sliding = create_sliding_window_causal_mask(
                config=self.pred_model.config,
                input_embeds=input_embeds,
                attention_mask=None,
                cache_position=cache_position,
                past_key_values=self.static_cache,
            )
            return {"full_attention": mask, "sliding_attention": sliding}
        return {"full_attention": mask}

    def _build_attention_masks(self):
        dummy_prefill = torch.zeros(1, 2, self.hidden_size, dtype=self.dtype, device=self.device)
        dummy_decode = torch.zeros(1, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        self.prefill_attn = self._make_attn_mask(dummy_prefill, self.prefill_cache_pos)
        self.decode_attn = []
        for pos in self.decode_cache_positions:
            self.decode_attn.append(self._make_attn_mask(dummy_decode, pos))

    def _full_loop(self):
        """The full 15-step predictor loop on static buffers."""
        # Project input from talker hidden size to predictor hidden size
        h = self.small_to_mtp(self.input_buf)  # [1, 2, hidden]

        # Prefill: 2 tokens through all layers
        out = self.pred_model(
            inputs_embeds=h,
            attention_mask=self.prefill_attn,
            past_key_values=self.static_cache,
            cache_position=self.prefill_cache_pos,
            use_cache=True,
        )
        h = out.last_hidden_state  # [1, 2, hidden] — already normalized

        # First codebook: logits from last position
        logits = self.lm_heads[0](h[:, -1:, :])  # [1, 1, vocab]
        tok = torch.argmax(logits[:, 0, :], dim=-1)  # [1]
        self.output_tokens[0] = tok[0]

        # Remaining 14 codebooks
        for cb_idx in range(1, self.num_codebooks):
            # Embed previous token using codebook-specific embedding
            emb = self.codec_embeds[cb_idx - 1](tok.unsqueeze(0))  # [1, 1, codec_hidden]
            emb = self.small_to_mtp(emb)  # [1, 1, hidden]

            # Single-token decode through all layers
            out = self.pred_model(
                inputs_embeds=emb,
                attention_mask=self.decode_attn[cb_idx - 1],
                past_key_values=self.static_cache,
                cache_position=self.decode_cache_positions[cb_idx - 1],
                use_cache=True,
            )
            h = out.last_hidden_state

            logits = self.lm_heads[cb_idx](h[:, -1:, :])
            tok = torch.argmax(logits[:, 0, :], dim=-1)
            self.output_tokens[cb_idx] = tok[0]

        return self.output_tokens

    @torch.inference_mode()
    def capture(self, num_warmup=3):
        """Warmup and capture the CUDA graph."""
        print(f"Warming up predictor ({num_warmup} runs)...")

        # Force cache initialization before graph capture
        self._init_cache_layers()
        self._build_attention_masks()

        for _ in range(num_warmup):
            self.static_cache.reset()
            self._full_loop()
        torch.cuda.synchronize()

        print("Capturing CUDA graph for predictor...")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.graph = torch.cuda.CUDAGraph()
            # Warmup in capture stream
            self.static_cache.reset()
            self._full_loop()
            torch.cuda.synchronize()

            self.static_cache.reset()
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
        self.static_cache.reset()
        self.graph.replay()
        return self.output_tokens.clone()
