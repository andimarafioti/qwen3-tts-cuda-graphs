import gc
import os
import random

import numpy as np
import pytest
import torch

from qwen_tts import Qwen3TTSModel

from faster_qwen3_tts import FasterQwen3TTS


MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
CUSTOM_MODEL_ID = os.environ.get("QWEN_TTS_CUSTOM_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
VOICE_DESIGN_MODEL_ID = os.environ.get("QWEN_TTS_VOICE_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

# The test phrase "Short parity test." has a natural EOS at ~84 tokens in xvec mode
# and ~16 tokens in ICL mode (with correct ref_text). 256 gives ample headroom.
_MAX_NEW_TOKENS = 256

# Correct transcript for ref_audio.wav (from demo/samples/parity/icl_transcripts.txt).
# ICL voice cloning aligns (text tokens ↔ codec frames) position-by-position.
# A mismatched transcript produces wrong alignment and the model loops indefinitely.
_ICL_REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same time "
    "are bullish on scaling up reinforcement learning atop LLMs. If we're actually "
    "close to a human-like learner, then this whole approach of training on verifiable "
    "outcomes is doomed."
)


def _seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _assert_codec_output_valid(fast_codes, config, max_new_tokens, label="",
                               check_natural_eos=True):
    """Structural validity checks for output from fast_generate.

    Verifies:
    - output is non-None and non-empty
    - correct number of codebooks (16 for Qwen3-TTS)
    - first-codebook tokens are in the un-suppressed range [0, vocab_size - 1024)
    - EOS token is absent (the generation loop strips it before appending)
    - all token values are non-negative (includes predictor codebooks 1-15)
    - (if check_natural_eos) generation terminated before budget exhaustion

    check_natural_eos=False is appropriate for ICL mode where the output length
    depends on the reference audio duration and may legitimately exceed _MAX_NEW_TOKENS.
    """
    pfx = f"[{label}] " if label else ""
    eos_id = config.codec_eos_token_id
    vocab_size = config.vocab_size
    suppress_start = max(0, vocab_size - 1024)

    assert fast_codes is not None, (
        f"{pfx}fast_generate returned None — EOS at prefill step or early crash"
    )
    assert fast_codes.shape[0] > 0, f"{pfx}empty codec output"
    assert fast_codes.shape[1] == config.num_code_groups, (
        f"{pfx}expected {config.num_code_groups} codebooks, got {fast_codes.shape[1]}"
    )

    first_cb = fast_codes[:, 0].cpu()

    # EOS is never appended: the loop does `if token == eos_id: break` before append.
    assert (first_cb != eos_id).all(), (
        f"{pfx}EOS token ({eos_id}) leaked into first codebook — "
        "generation loop may be broken"
    )

    # fast_generate suppresses tokens in [suppress_start, vocab_size) except eos_id,
    # so all sampled first-codebook tokens must lie in [0, suppress_start).
    if suppress_start > 0:
        bad = first_cb[first_cb >= suppress_start]
        assert len(bad) == 0, (
            f"{pfx}suppressed-range tokens in first codebook: {bad.tolist()} "
            f"(valid range is [0, {suppress_start}))"
        )

    # Predictor outputs (codebooks 1-15) must also be non-negative.
    assert (fast_codes >= 0).all(), f"{pfx}negative token values in codec output"

    if check_natural_eos:
        # Natural termination: EOS was generated before the budget ran out.
        # If shape[0] == max_new_tokens the loop hit the hard limit, which means EOS
        # was never produced — a bug or insufficient budget.
        assert fast_codes.shape[0] < max_new_tokens, (
            f"{pfx}generation used the full budget ({max_new_tokens} steps) — "
            "EOS was never produced; model may be looping or inputs are misconfigured. "
            f"Natural stop for the test phrase is ~84 tokens in xvec mode."
        )

    assert fast_codes.shape[0] >= 5, (
        f"{pfx}suspiciously short output ({fast_codes.shape[0]} steps) — "
        "possible premature termination or misconfigured inputs"
    )


def _assert_icl_codes_match(upstream_codes, fast_codes_cpu, label=""):
    """Prefix parity check for ICL fast-path vs upstream.

    ICL mode conditions on the reference audio and can generate sequences much
    longer than _MAX_NEW_TOKENS.  When the budget is exhausted, HF generate appends
    pad_token_id (= eos_token_id) as the last token; upstream post-processing detects
    this fake EOS and strips it, yielding budget-1 tokens.  The fast path runs for
    exactly budget steps (all legitimate codec tokens) and returns budget tokens.
    This gives a predictable 1-token length difference.

    We verify:
    - All tokens in the shorter (upstream) sequence match the corresponding fast tokens
    - The length difference is at most 1 (the fake-EOS strip artifact)
    """
    pfx = f"[{label}] " if label else ""
    min_len = min(upstream_codes.shape[0], fast_codes_cpu.shape[0])
    for i in range(min_len):
        if not torch.equal(upstream_codes[i], fast_codes_cpu[i]):
            pytest.fail(
                f"{pfx}first mismatch at step {i}: "
                f"upstream={upstream_codes[i].tolist()}, "
                f"fast={fast_codes_cpu[i].tolist()}"
            )
    length_diff = abs(upstream_codes.shape[0] - fast_codes_cpu.shape[0])
    assert length_diff <= 1, (
        f"{pfx}length diff {length_diff} > 1: upstream={upstream_codes.shape[0]}, "
        f"fast={fast_codes_cpu.shape[0]}. Expected at most 1 (HF fake-EOS strip)."
    )


def _assert_codes_match(upstream_codes, fast_codes_cpu, label=""):
    """Exact token-for-token parity check with a helpful first-mismatch message."""
    pfx = f"[{label}] " if label else ""
    min_len = min(upstream_codes.shape[0], fast_codes_cpu.shape[0])
    for i in range(min_len):
        if not torch.equal(upstream_codes[i], fast_codes_cpu[i]):
            pytest.fail(
                f"{pfx}first mismatch at step {i}: "
                f"upstream={upstream_codes[i].tolist()}, "
                f"fast={fast_codes_cpu[i].tolist()}"
            )
    assert upstream_codes.shape[0] == fast_codes_cpu.shape[0], (
        f"{pfx}length mismatch: upstream={upstream_codes.shape[0]} steps, "
        f"fast={fast_codes_cpu.shape[0]} steps"
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
#
# scope="class" ensures each fixture is torn down (models deleted, GPU memory
# freed) before the next class's fixture is set up.  With scope="module" all
# fixtures stay alive until the end of the module, causing 8 model instances to
# accumulate in GPU memory simultaneously and triggering OOM on 24 GB GPUs.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def parity_fixture():
    """bfloat16 xvec-only fixture.

    Used for:
    - Layer-2 bfloat16 functional tests (production dtype)
    - DynamicCache parity tests (both sides use the same code path)
    - Regression tests that don't need a specific dtype
    """
    _seed_all(0)

    device = "cuda"
    dtype = torch.bfloat16

    ref_audio = "ref_audio.wav"
    text = "Short parity test."

    # Both base and fast must use the same attn_implementation so that parity tests
    # (which compare fast_generate(parity_mode=True) against base.model.generate())
    # remain valid.  sdpa is required for bfloat16 CUDA-graph correctness: with
    # StaticCache padded to max_seq_len the eager BF16 GEMM kernel accumulates
    # differently for different K-sequence lengths (2048 vs the actual prefill length),
    # causing hidden-state divergence that grows step-by-step.  sdpa's tiled kernel
    # skips fully-masked K blocks, giving identical results to DynamicCache regardless
    # of StaticCache padding length.
    base = Qwen3TTSModel.from_pretrained(
        MODEL_ID, device_map=device, dtype=dtype, attn_implementation="sdpa"
    )
    fast = FasterQwen3TTS.from_pretrained(
        MODEL_ID, device=device, dtype=dtype, attn_implementation="sdpa"
    )

    prompt_items = base.create_voice_clone_prompt(ref_audio=ref_audio, ref_text="", x_vector_only_mode=True)
    vcp = base._prompt_items_to_voice_clone_prompt(prompt_items)

    input_ids_base = base._tokenize_texts([base._build_assistant_text(text)])
    input_ids_fast = fast.model._tokenize_texts([fast.model._build_assistant_text(text)])

    tie, tam, tth, tpe = fast._build_talker_inputs_local(
        m=fast.model.model,
        input_ids=input_ids_fast,
        ref_ids=[None],
        voice_clone_prompt=vcp,
        languages=["English"],
        speakers=None,
        non_streaming_mode=False,
    )

    if not fast._warmed_up:
        # Force deterministic predictor sampling so the functional tests are
        # reproducible (the predictor CUDA graph captures this greedy policy).
        fast.predictor_graph.do_sample = False
        fast.predictor_graph.top_k = 0
        fast.predictor_graph.top_p = 1.0
        fast.predictor_graph.temperature = 1.0
        fast._warmup(tie.shape[1])

    data = dict(
        base=base,
        fast=fast,
        vcp=vcp,
        input_ids_base=input_ids_base,
        input_ids_fast=input_ids_fast,
        tie=tie,
        tam=tam,
        tth=tth,
        tpe=tpe,
    )
    yield data
    del data["base"]
    del data["fast"]
    torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="class")
def parity_fixture_fp32():
    """float32 + TF32-off fixture for hardware-portable exact parity tests.

    Problem: bfloat16 tensor-core arithmetic differs between GPU microarchitectures
    (Blackwell sm_121, Ada Lovelace sm_89, integrated Ampere sm_87), so the argmax
    result of identical logits can diverge across hardware. This makes token-level
    parity assertions fragile (they require hardware-specific thresholds).

    Solution: float32 with TF32 disabled gives 24-bit mantissa precision that is
    consistent across these architectures. The extra mantissa bits provide enough
    margin that the static-cache vs dynamic-cache attention mask shape difference
    (the structural source of divergence) no longer flips any argmax within
    _MAX_NEW_TOKENS steps for the test phrase.

    TF32 is re-enabled after the fixture tears down; bfloat16 tests are unaffected
    since they never use float32 matmuls.
    """
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    _seed_all(0)

    device = "cuda"
    dtype = torch.float32

    ref_audio = "ref_audio.wav"
    text = "Short parity test."

    base = Qwen3TTSModel.from_pretrained(
        MODEL_ID, device_map=device, dtype=dtype, attn_implementation="eager"
    )
    fast = FasterQwen3TTS.from_pretrained(
        MODEL_ID, device=device, dtype=dtype, attn_implementation="eager"
    )

    prompt_items = base.create_voice_clone_prompt(ref_audio=ref_audio, ref_text="", x_vector_only_mode=True)
    vcp = base._prompt_items_to_voice_clone_prompt(prompt_items)

    input_ids_base = base._tokenize_texts([base._build_assistant_text(text)])
    input_ids_fast = fast.model._tokenize_texts([fast.model._build_assistant_text(text)])

    tie, tam, tth, tpe = fast._build_talker_inputs_local(
        m=fast.model.model,
        input_ids=input_ids_fast,
        ref_ids=[None],
        voice_clone_prompt=vcp,
        languages=["English"],
        speakers=None,
        non_streaming_mode=False,
    )

    if not fast._warmed_up:
        fast.predictor_graph.do_sample = False
        fast.predictor_graph.top_k = 0
        fast.predictor_graph.top_p = 1.0
        fast.predictor_graph.temperature = 1.0
        fast._warmup(tie.shape[1])

    data = dict(
        base=base,
        fast=fast,
        vcp=vcp,
        input_ids_base=input_ids_base,
        input_ids_fast=input_ids_fast,
        tie=tie,
        tam=tam,
        tth=tth,
        tpe=tpe,
    )
    yield data
    del data["base"]
    del data["fast"]
    torch.backends.cuda.matmul.allow_tf32 = prev_matmul
    torch.backends.cudnn.allow_tf32 = prev_cudnn
    torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="class")
def custom_voice_fixture():
    _seed_all(0)

    device = "cuda"
    dtype = torch.bfloat16
    text = "Short parity test."
    language = "English"

    base = Qwen3TTSModel.from_pretrained(
        CUSTOM_MODEL_ID, device_map=device, dtype=dtype, attn_implementation="eager"
    )
    fast = FasterQwen3TTS.from_pretrained(
        CUSTOM_MODEL_ID, device=device, dtype=dtype, attn_implementation="eager"
    )

    speakers = base.get_supported_speakers()
    assert speakers, "CustomVoice model returned no supported speakers"
    speaker = speakers[0]

    input_ids_base = base._tokenize_texts([base._build_assistant_text(text)])
    input_ids_fast = fast.model._tokenize_texts([fast.model._build_assistant_text(text)])

    _, _, _, tie, tam, tth, tpe = fast._prepare_generation_custom(
        text=text,
        language=language,
        speaker=speaker,
        instruct=None,
    )

    data = dict(
        base=base,
        fast=fast,
        speaker=speaker,
        language=language,
        input_ids_base=input_ids_base,
        input_ids_fast=input_ids_fast,
        tie=tie,
        tam=tam,
        tth=tth,
        tpe=tpe,
    )
    yield data
    del data["base"]
    del data["fast"]
    torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="class")
def voice_design_fixture():
    _seed_all(0)

    device = "cuda"
    dtype = torch.bfloat16
    text = "Short parity test."
    instruct = "Warm, calm voice."
    language = "English"

    base = Qwen3TTSModel.from_pretrained(
        VOICE_DESIGN_MODEL_ID, device_map=device, dtype=dtype, attn_implementation="eager"
    )
    fast = FasterQwen3TTS.from_pretrained(
        VOICE_DESIGN_MODEL_ID, device=device, dtype=dtype, attn_implementation="eager"
    )

    input_ids_base = base._tokenize_texts([base._build_assistant_text(text)])
    input_ids_fast = fast.model._tokenize_texts([fast.model._build_assistant_text(text)])

    _, _, _, tie, tam, tth, tpe = fast._prepare_generation_custom(
        text=text,
        language=language,
        speaker=None,
        instruct=instruct,
    )

    data = dict(
        base=base,
        fast=fast,
        language=language,
        instruct=instruct,
        input_ids_base=input_ids_base,
        input_ids_fast=input_ids_fast,
        tie=tie,
        tam=tam,
        tth=tth,
        tpe=tpe,
    )
    yield data
    del data["base"]
    del data["fast"]
    torch.cuda.empty_cache()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — FLOAT32 EXACT PARITY TESTS
#
# Goal: algorithmic correctness — does the fast path (CUDA graph + StaticCache)
# produce identical tokens to the upstream (HF generate + DynamicCache)?
#
# Why float32?  In bfloat16 the static-cache and dynamic-cache attention mask
# shapes differ ([1,1,1,max_seq_len] vs [1,1,1,current_len]), causing different
# CUDA softmax kernel dispatch and different FP accumulation order.  On Ada
# Lovelace (4090) this rarely flips an argmax, but on Blackwell (GB10) and
# integrated Ampere (Orin) the bfloat16 tensor-core arithmetic is different
# enough that the first mismatch occurs within the first ~10 steps.  Float32
# with TF32 disabled provides enough mantissa bits that these small accumulated
# differences no longer flip any argmax for the test phrase.
# ─────────────────────────────────────────────────────────────────────────────

class TestFP32Parity:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for parity test.")
    def test_voice_clone_token_parity_xvec_only(self, parity_fixture_fp32):
        """xvec-only fast path (CUDA graph + StaticCache) must exactly match upstream.

        In float32 with TF32 off, the fast and upstream paths produce identical
        token sequences for the test phrase despite using structurally different
        attention caches.  This test is portable across GPU architectures.
        """
        from faster_qwen3_tts.generate import fast_generate

        f = parity_fixture_fp32
        base, fast = f["base"], f["fast"]
        vcp = f["vcp"]

        assert torch.equal(f["input_ids_base"][0].cpu(), f["input_ids_fast"][0].cpu())

        fast_codes, _ = fast_generate(
            talker=fast.model.model.talker,
            talker_input_embeds=f["tie"],
            attention_mask=f["tam"],
            trailing_text_hiddens=f["tth"],
            tts_pad_embed=f["tpe"],
            config=fast.model.model.config.talker_config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
        )

        talker_codes_list, _ = base.model.generate(
            input_ids=f["input_ids_base"],
            ref_ids=[None],
            voice_clone_prompt=vcp,
            languages=["English"],
            non_streaming_mode=False,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            subtalker_dosample=False,
            subtalker_top_k=0,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
        )

        upstream_codes = talker_codes_list[0].detach().cpu()
        fast_codes_cpu = fast_codes.detach().cpu()
        _assert_codes_match(upstream_codes, fast_codes_cpu, label="xvec_only/fp32")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for parity test.")
    def test_voice_clone_icl_prefix_parity_fast_path(self, parity_fixture_fp32):
        """ICL fast path (CUDA graph + StaticCache) matches upstream in float32.

        ICL voice cloning requires ref_text to match the spoken content exactly — a
        mismatched transcript breaks the text↔codec alignment and the model loops
        indefinitely.  We use _ICL_REF_TEXT (the actual transcript of ref_audio.wav)
        and load audio via _load_ref_audio_with_silence (adds 0.5 s trailing silence,
        same as the production pipeline) so the model terminates naturally at ~16 tokens.

        In float32 with TF32 disabled the fast and upstream paths produce the same
        tokens on all tested hardware.  A length difference of ≤1 is tolerated because
        HF generate injects a fake EOS token when the budget is exhausted and
        upstream post-processing strips it; with natural termination this does not
        arise in practice but we keep the tolerance for robustness.
        """
        from faster_qwen3_tts.generate import fast_generate

        f = parity_fixture_fp32
        base, fast = f["base"], f["fast"]

        ref_audio = "ref_audio.wav"
        text = "Short parity test."

        with torch.inference_mode():
            ref_audio_input = fast._load_ref_audio_with_silence(ref_audio)
            prompt_items = base.create_voice_clone_prompt(
                ref_audio=ref_audio_input,
                ref_text=_ICL_REF_TEXT,
                x_vector_only_mode=False,
            )
            vcp = base._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_text = prompt_items[0].ref_text

            input_ids_base = base._tokenize_texts([base._build_assistant_text(text)])
            input_ids_fast = fast.model._tokenize_texts([fast.model._build_assistant_text(text)])
            ref_ids = [base._tokenize_texts([base._build_ref_text(ref_text)])[0]]

            tie, tam, tth, tpe = fast._build_talker_inputs_local(
                m=fast.model.model,
                input_ids=input_ids_fast,
                ref_ids=ref_ids,
                voice_clone_prompt=vcp,
                languages=["English"],
                speakers=None,
                non_streaming_mode=True,
            )

        if not fast._warmed_up:
            fast.predictor_graph.do_sample = False
            fast.predictor_graph.top_k = 0
            fast.predictor_graph.top_p = 1.0
            fast.predictor_graph.temperature = 1.0
            fast._warmup(tie.shape[1])

        fast.model.model.talker.rope_deltas = None
        fast_codes, _ = fast_generate(
            talker=fast.model.model.talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=fast.model.model.config.talker_config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
        )

        talker_codes_list, _ = base.model.generate(
            input_ids=input_ids_base,
            ref_ids=ref_ids,
            voice_clone_prompt=vcp,
            languages=["English"],
            non_streaming_mode=True,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            subtalker_dosample=False,
            subtalker_top_k=0,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
        )

        upstream_codes = talker_codes_list[0].detach().cpu()
        fast_codes_cpu = fast_codes.detach().cpu()
        _assert_icl_codes_match(upstream_codes, fast_codes_cpu, label="icl_fast_path/fp32")


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — BFLOAT16 FUNCTIONAL TESTS
#
# Goal: production behavior — does the fast path (CUDA graph + StaticCache) in
# bfloat16 (the actual inference dtype) generate structurally valid codec output?
#
# These tests do NOT compare fast vs upstream token-for-token.  Instead they
# check structural invariants that must hold regardless of which exact tokens
# the model picks:
#   - tokens are in the valid (un-suppressed) codec range
#   - EOS was generated and stripped correctly
#   - generation terminates naturally before the budget is exhausted
#   - output has a plausible length for the test phrase
#
# These tests are portable across GPU architectures because they never require a
# specific token value, only structural properties.
# ─────────────────────────────────────────────────────────────────────────────

class TestBF16Parity:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
    def test_voice_clone_xvec_bf16_generates_valid_tokens(self, parity_fixture):
        """xvec-only fast path produces structurally valid codec output in bfloat16."""
        from faster_qwen3_tts.generate import fast_generate

        _seed_all(0)
        fast = parity_fixture["fast"]
        config = fast.model.model.config.talker_config

        fast_codes, _ = fast_generate(
            talker=fast.model.model.talker,
            talker_input_embeds=parity_fixture["tie"],
            attention_mask=parity_fixture["tam"],
            trailing_text_hiddens=parity_fixture["tth"],
            tts_pad_embed=parity_fixture["tpe"],
            config=config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
        )

        _assert_codec_output_valid(fast_codes, config, _MAX_NEW_TOKENS, label="xvec/bf16")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
    def test_voice_clone_icl_bf16_generates_valid_tokens(self, parity_fixture):
        """ICL fast path produces structurally valid codec output in bfloat16."""
        from faster_qwen3_tts.generate import fast_generate

        _seed_all(0)
        fast = parity_fixture["fast"]

        ref_audio = "ref_audio.wav"
        text = "Short parity test."

        with torch.inference_mode():
            ref_audio_input = fast._load_ref_audio_with_silence(ref_audio)
            prompt_items = fast.model.create_voice_clone_prompt(
                ref_audio=ref_audio_input,
                ref_text=_ICL_REF_TEXT,
                x_vector_only_mode=False,
            )
            vcp = fast.model._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_text = prompt_items[0].ref_text
            input_ids_fast = fast.model._tokenize_texts([fast.model._build_assistant_text(text)])
            ref_ids = [fast.model._tokenize_texts([fast.model._build_ref_text(ref_text)])[0]]
            tie, tam, tth, tpe = fast._build_talker_inputs_local(
                m=fast.model.model,
                input_ids=input_ids_fast,
                ref_ids=ref_ids,
                voice_clone_prompt=vcp,
                languages=["English"],
                speakers=None,
                non_streaming_mode=True,
            )

        config = fast.model.model.config.talker_config

        fast.model.model.talker.rope_deltas = None
        fast_codes, _ = fast_generate(
            talker=fast.model.model.talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
        )

        _assert_codec_output_valid(fast_codes, config, _MAX_NEW_TOKENS, label="icl/bf16",
                                   check_natural_eos=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
    def test_streaming_bf16_produces_valid_chunks(self, parity_fixture):
        """Streaming fast path produces non-empty, valid chunks in bfloat16."""
        from faster_qwen3_tts.streaming import fast_generate_streaming

        _seed_all(0)
        fast = parity_fixture["fast"]
        config = fast.model.model.config.talker_config

        chunks = []
        for chunk, info in fast_generate_streaming(
            talker=fast.model.model.talker,
            talker_input_embeds=parity_fixture["tie"],
            attention_mask=parity_fixture["tam"],
            trailing_text_hiddens=parity_fixture["tth"],
            tts_pad_embed=parity_fixture["tpe"],
            config=config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            chunk_size=8,
        ):
            assert chunk.shape[0] > 0, f"chunk {info['chunk_index']} is empty"
            assert chunk.shape[1] == config.num_code_groups, (
                f"chunk {info['chunk_index']} has wrong codebook count: {chunk.shape[1]}"
            )
            chunks.append(chunk)

        assert len(chunks) > 0, "streaming produced no chunks"

        all_codes = torch.cat(chunks, dim=0)
        _assert_codec_output_valid(all_codes, config, _MAX_NEW_TOKENS, label="streaming/bf16")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for streaming parity test.")
    def test_streaming_matches_non_streaming_prefix(self, parity_fixture):
        """Streaming and non-streaming fast paths produce identical tokens.

        Both call the same CUDA graph with the same inputs; they must agree on every
        token.  Using _MAX_NEW_TOKENS ensures both paths reach the natural EOS and
        we verify the full output, not a truncated prefix.
        """
        from faster_qwen3_tts.generate import fast_generate
        from faster_qwen3_tts.streaming import fast_generate_streaming

        _seed_all(0)
        fast = parity_fixture["fast"]

        full_codes, _ = fast_generate(
            talker=fast.model.model.talker,
            talker_input_embeds=parity_fixture["tie"],
            attention_mask=parity_fixture["tam"],
            trailing_text_hiddens=parity_fixture["tth"],
            tts_pad_embed=parity_fixture["tpe"],
            config=fast.model.model.config.talker_config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
        )

        chunks = []
        for chunk, _ in fast_generate_streaming(
            talker=fast.model.model.talker,
            talker_input_embeds=parity_fixture["tie"],
            attention_mask=parity_fixture["tam"],
            trailing_text_hiddens=parity_fixture["tth"],
            tts_pad_embed=parity_fixture["tpe"],
            config=fast.model.model.config.talker_config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            chunk_size=8,
        ):
            chunks.append(chunk)

        stream_codes = torch.cat(chunks, dim=0)
        _assert_codes_match(full_codes.cpu(), stream_codes.cpu(), label="streaming_vs_non_streaming")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ICL parity test.")
    def test_voice_clone_icl_full_parity_dynamic_cache(self, parity_fixture):
        """fast._build_talker_inputs_local matches upstream for ICL mode (DynamicCache)."""
        from faster_qwen3_tts.generate import fast_generate

        _seed_all(0)

        ref_audio = "ref_audio.wav"
        text = "Short parity test."

        base = parity_fixture["base"]
        fast = parity_fixture["fast"]

        with torch.inference_mode():
            ref_audio_input = fast._load_ref_audio_with_silence(ref_audio)
            prompt_items = base.create_voice_clone_prompt(
                ref_audio=ref_audio_input,
                ref_text=_ICL_REF_TEXT,
                x_vector_only_mode=False,
            )
            vcp = base._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_text = prompt_items[0].ref_text

            input_ids_base = base._tokenize_texts([base._build_assistant_text(text)])
            input_ids_fast = fast.model._tokenize_texts([fast.model._build_assistant_text(text)])
            ref_ids = [base._tokenize_texts([base._build_ref_text(ref_text)])[0]]

            tie, tam, tth, tpe = fast._build_talker_inputs_local(
                m=fast.model.model,
                input_ids=input_ids_fast,
                ref_ids=ref_ids,
                voice_clone_prompt=vcp,
                languages=["English"],
                speakers=None,
                non_streaming_mode=True,
            )

        if not fast._warmed_up:
            fast.predictor_graph.do_sample = False
            fast.predictor_graph.top_k = 0
            fast.predictor_graph.top_p = 1.0
            fast.predictor_graph.temperature = 1.0
            fast._warmup(tie.shape[1])

        fast.model.model.talker.rope_deltas = None
        fast_codes, _ = fast_generate(
            talker=fast.model.model.talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=fast.model.model.config.talker_config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            subtalker_dosample=False,
            subtalker_top_k=0,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
            parity_mode=True,
        )

        talker_codes_list, _ = base.model.generate(
            input_ids=input_ids_base,
            ref_ids=ref_ids,
            voice_clone_prompt=vcp,
            languages=["English"],
            non_streaming_mode=True,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            subtalker_dosample=False,
            subtalker_top_k=0,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
        )

        upstream_codes = talker_codes_list[0].detach().cpu()
        fast_codes_cpu = fast_codes.detach().cpu()
        _assert_codes_match(upstream_codes, fast_codes_cpu, label="icl/dynamic_cache")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ICL parity test.")
    def test_icl_build_talker_inputs_outside_inference_mode(self, parity_fixture):
        """Regression test: _build_talker_inputs_local must work with ICL ref_code tensors
        even when called outside of torch.inference_mode().

        create_voice_clone_prompt() runs under @torch.inference_mode(), producing inference
        tensors. Without the .clone() fix these tensors trigger a RuntimeError when passed
        to nn.Embedding outside an inference_mode context.
        """
        base = parity_fixture["base"]
        fast = parity_fixture["fast"]

        ref_audio = "ref_audio.wav"
        text = "Short parity test."

        prompt_items = base.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=_ICL_REF_TEXT,
            x_vector_only_mode=False,
        )
        vcp = base._prompt_items_to_voice_clone_prompt(prompt_items)
        assert vcp["ref_code"][0] is not None and vcp["ref_code"][0].is_inference(), (
            "pre-condition: ref_code must be an inference tensor for this test to be meaningful"
        )

        ref_text = prompt_items[0].ref_text
        input_ids = fast.model._tokenize_texts([fast.model._build_assistant_text(text)])
        ref_ids = [base._tokenize_texts([base._build_ref_text(ref_text)])[0]]

        # Must NOT be wrapped in torch.inference_mode() — that is the scenario that used to crash.
        fast._build_talker_inputs_local(
            m=fast.model.model,
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=vcp,
            languages=["English"],
            speakers=None,
            non_streaming_mode=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — DYNAMIC CACHE PARITY TESTS (CustomVoice / VoiceDesign)
#
# Goal: verify that fast._build_talker_inputs_local correctly replicates what
# the upstream build internally, by comparing the two when both run the SAME
# underlying code (talker.generate with DynamicCache, parity_mode=True).
#
# dtype does not matter here: both sides execute the identical bfloat16 compute
# graph in the same GPU session, so any hardware-specific non-determinism affects
# them equally and the outputs are bit-for-bit identical.
#
# max_new_tokens: previously 32–64, now _MAX_NEW_TOKENS.  The test phrase has a
# natural EOS at ~84 tokens; the old budgets were too small, causing HF generate
# to pad with pad_token_id=eos_token_id and creating synthetic EOS that the two
# post-processing paths handled differently.
# ─────────────────────────────────────────────────────────────────────────────

class TestCustomVoice:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
    def test_custom_voice_bf16_generates_valid_tokens(self, custom_voice_fixture):
        """CustomVoice fast path (CUDA graph) produces valid codec tokens in bfloat16."""
        from faster_qwen3_tts.generate import fast_generate

        _seed_all(0)
        fast = custom_voice_fixture["fast"]
        config = fast.model.model.config.talker_config

        fast_codes, _ = fast_generate(
            talker=fast.model.model.talker,
            talker_input_embeds=custom_voice_fixture["tie"],
            attention_mask=custom_voice_fixture["tam"],
            trailing_text_hiddens=custom_voice_fixture["tth"],
            tts_pad_embed=custom_voice_fixture["tpe"],
            config=config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
        )

        _assert_codec_output_valid(fast_codes, config, _MAX_NEW_TOKENS, label="custom_voice/bf16")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for custom voice parity test.")
    def test_custom_voice_full_parity_dynamic_cache(self, custom_voice_fixture):
        """fast._build_talker_inputs_local matches upstream for CustomVoice (DynamicCache)."""
        from faster_qwen3_tts.generate import fast_generate

        base = custom_voice_fixture["base"]
        fast = custom_voice_fixture["fast"]
        speaker = custom_voice_fixture["speaker"]
        language = custom_voice_fixture["language"]

        assert torch.equal(custom_voice_fixture["input_ids_base"][0].cpu(), custom_voice_fixture["input_ids_fast"][0].cpu())

        fast_codes, _ = fast_generate(
            talker=fast.model.model.talker,
            talker_input_embeds=custom_voice_fixture["tie"],
            attention_mask=custom_voice_fixture["tam"],
            trailing_text_hiddens=custom_voice_fixture["tth"],
            tts_pad_embed=custom_voice_fixture["tpe"],
            config=fast.model.model.config.talker_config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            subtalker_dosample=False,
            subtalker_top_k=0,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
            parity_mode=True,
        )

        talker_codes_list, _ = base.model.generate(
            input_ids=custom_voice_fixture["input_ids_base"],
            instruct_ids=[None],
            speakers=[speaker],
            languages=[language],
            non_streaming_mode=False,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            subtalker_dosample=False,
            subtalker_top_k=0,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
        )

        upstream_codes = talker_codes_list[0].detach().cpu()
        fast_codes_cpu = fast_codes.detach().cpu()
        _assert_codes_match(upstream_codes, fast_codes_cpu, label="custom_voice/dynamic_cache")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
def test_instruct_prepends_tokens_to_voice_clone(parity_fixture):
    """instruct= should prepend exactly instruct_len tokens to the talker input embeds,
    leaving the suffix (text + codec part) byte-for-byte identical to the no-instruct case.
    """
    fast = parity_fixture["fast"]

    ref_audio = "ref_audio.wav"
    text = "Short parity test."
    language = "English"
    instruct_str = "Please speak in a slow, calm tone."

    with torch.inference_mode():
        _, _, _, tie_base, tam_base, _, _, _ = fast._prepare_generation(
            text, ref_audio, "", language=language, non_streaming_mode=False
        )
        _, _, _, tie_inst, tam_inst, _, _, _ = fast._prepare_generation(
            text, ref_audio, "", language=language, non_streaming_mode=False,
            instruct=instruct_str,
        )

    instruct_ids = fast.model._tokenize_texts([fast.model._build_instruct_text(instruct_str)])[0]
    instruct_len = instruct_ids.shape[1]

    # instruct tokens are prepended — sequence grows by exactly instruct_len
    assert tie_inst.shape[1] == tie_base.shape[1] + instruct_len
    assert tam_inst.shape[1] == tam_base.shape[1] + instruct_len

    # the suffix (everything after the instruct prefix) is unchanged
    assert torch.equal(tie_inst[:, instruct_len:], tie_base)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
def test_instruct_changes_generation_output(parity_fixture):
    """Passing instruct= must produce different codec tokens than not passing it,
    confirming the instruction propagates through the full decode loop.
    """
    _seed_all(0)

    fast = parity_fixture["fast"]

    ref_audio = "ref_audio.wav"
    text = "Short parity test."
    language = "English"

    with torch.inference_mode():
        wav_base, _ = fast.generate_voice_clone(
            text=text, language=language, ref_audio=ref_audio, ref_text="",
            max_new_tokens=32, do_sample=False, top_k=0, top_p=1.0, temperature=1.0,
            repetition_penalty=1.0,
        )
        wav_inst, _ = fast.generate_voice_clone(
            text=text, language=language, ref_audio=ref_audio, ref_text="",
            max_new_tokens=32, do_sample=False, top_k=0, top_p=1.0, temperature=1.0,
            repetition_penalty=1.0,
            instruct="Please speak very slowly and with a deep voice.",
        )

    assert not torch.equal(
        torch.tensor(wav_base[0]), torch.tensor(wav_inst[0])
    ), "instruct had no effect on the generated audio"


class TestVoiceDesign:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")
    def test_voice_design_bf16_generates_valid_tokens(self, voice_design_fixture):
        """VoiceDesign fast path (CUDA graph) produces valid codec tokens in bfloat16."""
        from faster_qwen3_tts.generate import fast_generate

        _seed_all(0)
        fast = voice_design_fixture["fast"]
        config = fast.model.model.config.talker_config

        fast_codes, _ = fast_generate(
            talker=fast.model.model.talker,
            talker_input_embeds=voice_design_fixture["tie"],
            attention_mask=voice_design_fixture["tam"],
            trailing_text_hiddens=voice_design_fixture["tth"],
            tts_pad_embed=voice_design_fixture["tpe"],
            config=config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
        )

        _assert_codec_output_valid(fast_codes, config, _MAX_NEW_TOKENS, label="voice_design/bf16")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for voice design parity test.")
    def test_voice_design_full_parity_dynamic_cache(self, voice_design_fixture):
        """fast._build_talker_inputs_local matches upstream for VoiceDesign (DynamicCache)."""
        from faster_qwen3_tts.generate import fast_generate

        base = voice_design_fixture["base"]
        fast = voice_design_fixture["fast"]
        language = voice_design_fixture["language"]
        instruct = voice_design_fixture["instruct"]

        assert torch.equal(voice_design_fixture["input_ids_base"][0].cpu(), voice_design_fixture["input_ids_fast"][0].cpu())

        fast_codes, _ = fast_generate(
            talker=fast.model.model.talker,
            talker_input_embeds=voice_design_fixture["tie"],
            attention_mask=voice_design_fixture["tam"],
            trailing_text_hiddens=voice_design_fixture["tth"],
            tts_pad_embed=voice_design_fixture["tpe"],
            config=fast.model.model.config.talker_config,
            predictor_graph=fast.predictor_graph,
            talker_graph=fast.talker_graph,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            subtalker_dosample=False,
            subtalker_top_k=0,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
            parity_mode=True,
        )

        instruct_ids = base._tokenize_texts([base._build_instruct_text(instruct)])[0]
        talker_codes_list, _ = base.model.generate(
            input_ids=voice_design_fixture["input_ids_base"],
            instruct_ids=[instruct_ids],
            languages=[language],
            non_streaming_mode=False,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            max_new_tokens=_MAX_NEW_TOKENS,
            min_new_tokens=0,
            subtalker_dosample=False,
            subtalker_top_k=0,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
        )

        upstream_codes = talker_codes_list[0].detach().cpu()
        fast_codes_cpu = fast_codes.detach().cpu()
        _assert_codes_match(upstream_codes, fast_codes_cpu, label="voice_design/dynamic_cache")
