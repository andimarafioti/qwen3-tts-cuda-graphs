#!/usr/bin/env python3
"""Generate non_streaming_mode (True/False) comparison samples for ICL voice cloning."""
import os
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from faster_qwen3_tts import FasterQwen3TTS

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("NSM_SAMPLES_DIR", PROJECT_DIR / "samples" / "non_streaming_mode"))
MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

PROMPTS = [
    "I left the window open last night, and the rain made a soft rhythm on the roof while I tried to sleep.",
    "Tomorrow is the first day of spring, so I'm taking a long walk and bringing a notebook to write down ideas.",
]

TRANSCRIPTS_PATH = PROJECT_DIR / "samples" / "parity" / "icl_transcripts.txt"


def load_transcripts():
    transcripts = {}
    if not TRANSCRIPTS_PATH.exists():
        return transcripts
    for line in TRANSCRIPTS_PATH.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key_part, text = line.split(":", 1)
        key = key_part.split("(")[0].strip()
        transcripts[key] = text.strip()
    return transcripts


REFS = [
    ("ref_audio", PROJECT_DIR / "ref_audio.wav"),
    ("ref_audio_2", PROJECT_DIR / "ref_audio_2.wav"),
    ("ref_audio_3", PROJECT_DIR / "ref_audio_3.wav"),
]

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "168"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.9"))
TOP_K = int(os.environ.get("TOP_K", "50"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
DO_SAMPLE = os.environ.get("DO_SAMPLE", "1") == "1"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = FasterQwen3TTS.from_pretrained(
        MODEL_ID,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        max_seq_len=2048,
    )

    transcripts = load_transcripts()

    for ref_idx, (ref_key, ref_path) in enumerate(REFS):
        ref_text = transcripts.get(ref_key, "")
        for prompt_idx, prompt in enumerate(PROMPTS):
            seed = 1337 + ref_idx * 10 + prompt_idx
            for mode in (False, True):
                set_seed(seed)
                audio_list, sr = model.generate_voice_clone(
                    text=prompt,
                    language="English",
                    ref_audio=str(ref_path),
                    ref_text=ref_text,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                    do_sample=DO_SAMPLE,
                    xvec_only=False,
                    non_streaming_mode=mode,
                )
                audio = audio_list[0]
                if hasattr(audio, "cpu"):
                    audio = audio.flatten().cpu().numpy()
                else:
                    audio = audio.flatten() if hasattr(audio, "flatten") else audio
                suffix = "nsm_true" if mode else "nsm_false"
                out_path = OUT_DIR / f"icl_{ref_key}_gen{prompt_idx + 1}_{suffix}.wav"
                sf.write(out_path, audio, sr)
                print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
