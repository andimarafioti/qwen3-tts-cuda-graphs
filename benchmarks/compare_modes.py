#!/usr/bin/env python3
"""Compare CustomVoice vs VoiceClone (xvec vs full ICL) for 0.6B."""
import torch
import time
import os
import numpy as np
from faster_qwen3_tts import FasterQwen3TTS

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = os.environ.get(
    'TEXT',
    "Ladies and gentlemen, I have just been informed that this speech is being generated faster than I can speak it."
)
LANG = os.environ.get('LANGUAGE', 'English')
REF_AUDIO = os.path.join(PROJECT_DIR, 'ref_audio.wav')
REF_TEXT = os.environ.get(
    'REF_TEXT',
    "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs."
)
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '8'))
TTFA_RUNS = int(os.environ.get('TTFA_RUNS', '10'))
RTF_RUNS = int(os.environ.get('RTF_RUNS', '5'))


def bench_stream(fn, label):
    # Warmup (includes graph capture)
    _ = fn(max_new_tokens=20, streaming=False)

    # TTFA (streaming)
    ttfas = []
    for _ in range(TTFA_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = fn(max_new_tokens=512, chunk_size=CHUNK_SIZE, streaming=True)
        try:
            _chunk, _sr, _timing = next(gen)
        finally:
            gen.close()
        torch.cuda.synchronize()
        ttfas.append((time.perf_counter() - t0) * 1000)
    ttfa_mean = float(np.mean(ttfas))
    ttfa_std = float(np.std(ttfas))

    # Throughput (N runs)
    rtfs = []
    ms_steps = []
    for _ in range(RTF_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        audio_list, sr = fn(max_new_tokens=512, streaming=False)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        audio = audio_list[0]
        dur = len(audio) / sr
        rtf = dur / total
        rtfs.append(rtf)
        steps = int(round(dur * 12))
        ms_steps.append((total / max(steps, 1)) * 1000)

    print(
        f"{label:>18} | TTFA={ttfa_mean:6.0f}ms ± {ttfa_std:4.0f} | "
        f"RTF={np.mean(rtfs):.3f} ± {np.std(rtfs):.3f} | "
        f"ms/step={np.mean(ms_steps):.1f} ± {np.std(ms_steps):.1f}"
    )


def main():
    # Base model for voice clone
    base_id = 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
    print(f"Loading Base model: {base_id}")
    base = FasterQwen3TTS.from_pretrained(base_id, device='cuda', dtype=torch.bfloat16, attn_implementation='eager')

    def vc_fn(xvec_only, max_new_tokens=512, chunk_size=CHUNK_SIZE, streaming=True):
        if streaming:
            return base.generate_voice_clone_streaming(
                text=TEXT,
                language=LANG,
                ref_audio=REF_AUDIO,
                ref_text=REF_TEXT,
                xvec_only=xvec_only,
                max_new_tokens=max_new_tokens,
                chunk_size=chunk_size,
            )
        return base.generate_voice_clone(
            text=TEXT,
            language=LANG,
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            xvec_only=xvec_only,
            max_new_tokens=max_new_tokens,
        )

    # CustomVoice model
    custom_id = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'
    print(f"Loading CustomVoice model: {custom_id}")
    custom = FasterQwen3TTS.from_pretrained(custom_id, device='cuda', dtype=torch.bfloat16, attn_implementation='eager')
    speakers = custom.model.get_supported_speakers() or []
    if not speakers:
        raise RuntimeError("No speakers reported by custom voice model")
    speaker = os.environ.get('SPEAKER', speakers[0])

    def cv_fn(max_new_tokens=512, chunk_size=CHUNK_SIZE, streaming=True):
        if streaming:
            return custom.generate_custom_voice_streaming(
                text=TEXT,
                speaker=speaker,
                language=LANG,
                chunk_size=chunk_size,
                max_new_tokens=max_new_tokens,
            )
        return custom.generate_custom_voice(
            text=TEXT,
            speaker=speaker,
            language=LANG,
            max_new_tokens=max_new_tokens,
        )

    print("\n=== Compare modes (0.6B) ===")
    bench_stream(lambda **kw: vc_fn(True, **kw), "VoiceClone xvec")
    bench_stream(lambda **kw: vc_fn(False, **kw), "VoiceClone full")
    bench_stream(lambda **kw: cv_fn(**kw), "CustomVoice")


if __name__ == "__main__":
    main()
