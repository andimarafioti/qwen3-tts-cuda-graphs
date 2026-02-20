#!/usr/bin/env python3
"""
Real-time TTS with CUDA graphs using a precomputed speaker embedding (x-vector mode).

This mode uses only the speaker embedding for voice identity, without the full
acoustic prompt from reference audio. Benefits:
- No accent bleed from reference audio into other languages
- Shorter prefill (10 tokens vs ~80+ in ICL mode) = lower TTFT
- Speaker embedding can be precomputed and cached (4KB file)

Usage:
    # First extract the speaker embedding (one-time):
    python examples/extract_speaker.py --ref_audio voice.wav --output speaker.pt

    # Then generate with CUDA graphs:
    python examples/generate_with_embedding.py --speaker speaker.pt --text "Hello world" --language English --output out.wav
    python examples/generate_with_embedding.py --speaker speaker.pt --text "Bonjour le monde" --language French --output out.wav
"""
import argparse
import torch
import sys

sys.path.insert(0, '.')


def load_xvector_prompt(path: str, device: str = "cuda:0") -> dict:
    """Load a saved x-vector and return a voice_clone_prompt dict."""
    spk_emb = torch.load(path, weights_only=True).to(device)
    return dict(
        ref_code=[None],
        ref_spk_embedding=[spk_emb],
        x_vector_only_mode=[True],
        icl_mode=[False],
    )


def main():
    parser = argparse.ArgumentParser(description="CUDA-graphed TTS with precomputed speaker embedding")
    parser.add_argument("--speaker", required=True, help="Path to speaker embedding (.pt from extract_speaker.py)")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", default="Auto", help="Language (English, French, German, Spanish, ...)")
    parser.add_argument("--output", default="output.wav", help="Output wav path")
    parser.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Model path")
    parser.add_argument("--device", default="cuda:0", help="Device")
    args = parser.parse_args()

    import soundfile as sf
    from faster_qwen3_tts import FasterQwen3TTS
    from faster_qwen3_tts.generate import fast_generate

    print(f"Loading model from {args.model_path}...")
    model = FasterQwen3TTS.from_pretrained(
        args.model_path,
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Load precomputed speaker embedding
    print(f"Loading speaker embedding from {args.speaker}...")
    vcp = load_xvector_prompt(args.speaker, device=args.device)

    # Build inputs
    input_texts = [model.model._build_assistant_text(args.text)]
    input_ids = model.model._tokenize_texts(input_texts)

    tie, tam, tth, tpe = model._build_talker_inputs_local(
        m=model.model.model,
        input_ids=input_ids,
        ref_ids=[None],
        voice_clone_prompt=vcp,
        languages=[args.language],
        speakers=None,
        non_streaming_mode=False,
    )
    prefill_len = tie.shape[1]
    print(f"Prefill length: {prefill_len} tokens")

    # Warmup + capture CUDA graphs (one-time)
    model._warmup(prefill_len)

    # Warmup
    talker = model.model.model.talker
    config = model.model.model.config.talker_config
    talker.rope_deltas = None
    fast_generate(
        talker, tie, tam, tth, tpe, config, model.predictor_graph, model.talker_graph,
        temperature=0.9, top_k=50, do_sample=True, max_new_tokens=20,
    )

    # Generate
    print("Generating...")
    talker.rope_deltas = None
    codec_ids, timing = fast_generate(
        talker, tie, tam, tth, tpe, config, model.predictor_graph, model.talker_graph,
        temperature=0.9, top_k=50, do_sample=True, max_new_tokens=2048,
    )
    if codec_ids is not None and codec_ids.numel() > 0:
        wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": codec_ids.to(model.device)}])
        audio = wavs[0]
        sf.write(args.output, audio, sr)
        n_steps = timing['steps']
        audio_dur = n_steps / 12.0
        gen_time = timing['prefill_ms'] / 1000 + timing['decode_s']
        rtf = audio_dur / gen_time
        print(f"Saved {args.output} ({audio_dur:.1f}s audio, {gen_time:.2f}s gen, RTF {rtf:.2f})")
        print(f"  Prefill: {timing['prefill_ms']:.0f}ms | Decode: {n_steps} steps @ {timing['ms_per_step']:.1f}ms/step")
    else:
        print("ERROR: generation returned no tokens")
        sys.exit(1)


if __name__ == "__main__":
    main()
