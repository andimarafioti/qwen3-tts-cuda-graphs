#!/usr/bin/env python3
"""
Extract a speaker embedding (x-vector) from a reference audio file.

This saves the speaker identity as a 4KB tensor that can be loaded instantly
at inference time, skipping the speaker encoder and audio tokenizer entirely.

Usage:
    python extract_speaker.py --ref_audio voice.wav --output speaker.pt
    python extract_speaker.py --ref_audio voice.wav --output speaker.pt --model_path ./models/Qwen3-TTS-12Hz-1.7B-Base
"""
import argparse
import torch
import sys
sys.path.insert(0, '.')

def main():
    parser = argparse.ArgumentParser(description="Extract speaker embedding from reference audio")
    parser.add_argument("--ref_audio", required=True, help="Path to reference audio file (wav)")
    parser.add_argument("--output", required=True, help="Output path for speaker embedding (.pt)")
    parser.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Model path or HF model id")
    parser.add_argument("--device", default="cuda:0", help="Device")
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel

    print(f"Loading model from {args.model_path}...")
    model = Qwen3TTSModel.from_pretrained(args.model_path, device_map=args.device, dtype=torch.bfloat16)

    print(f"Extracting speaker embedding from {args.ref_audio}...")
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text="",
        x_vector_only_mode=True,
    )

    spk_emb = prompt_items[0].ref_spk_embedding.cpu()
    torch.save(spk_emb, args.output)
    print(f"Saved speaker embedding to {args.output}")
    print(f"  Shape: {spk_emb.shape}, dtype: {spk_emb.dtype}, size: {spk_emb.nelement() * 2} bytes")


if __name__ == "__main__":
    main()
