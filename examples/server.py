#!/usr/bin/env python3
"""
FastAPI server for Qwen3-TTS voice cloning with CUDA graphs.

Loads the model once at startup, caches the speaker embedding from
a reference audio file, and serves a simple JSON-in / WAV-out API.

Usage:
    # Basic (uses defaults):
    python examples/server.py --ref-audio voice.wav --ref-text "transcript of voice.wav"

    # All options:
    python examples/server.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --ref-audio voice.wav \
        --ref-text "transcript of voice.wav" \
        --host 0.0.0.0 \
        --port 8100

    # Then:
    curl -X POST http://localhost:8100/tts \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello world!", "language": "English"}' \
        --output speech.wav

Environment variables (override defaults, CLI args take precedence):
    QWEN_TTS_MODEL       Model name or path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)
    QWEN_TTS_REF_AUDIO   Path to reference audio file
    QWEN_TTS_REF_TEXT    Transcript of reference audio
    QWEN_TTS_HOST        Server host (default: 0.0.0.0)
    QWEN_TTS_PORT        Server port (default: 8100)
"""
import argparse
import io
import os
import time

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from faster_qwen3_tts import FasterQwen3TTS

app = FastAPI(title="Qwen3-TTS Voice Clone API")

# Populated at startup
model = None
_config = {}


class TTSRequest(BaseModel):
    text: str
    language: str = "English"


@app.on_event("startup")
async def startup():
    global model

    model_name = _config["model"]
    ref_audio = _config["ref_audio"]
    ref_text = _config["ref_text"]

    print(f"Loading model: {model_name}")
    start = time.time()
    model = FasterQwen3TTS.from_pretrained(
        model_name,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    print(f"Model loaded in {time.time() - start:.1f}s")

    print("Warming up (CUDA graph capture + voice prompt caching)...")
    start = time.time()
    model.generate_voice_clone(
        text="Warmup.",
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    print(f"Warmup complete in {time.time() - start:.1f}s")
    print("Ready!")


@app.post("/tts")
async def generate_tts(request: TTSRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    wavs, sr = model.generate_voice_clone(
        text=request.text,
        language=request.language,
        ref_audio=_config["ref_audio"],
        ref_text=_config["ref_text"],
    )
    gen_time = time.time() - start

    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sr, format="WAV")
    buffer.seek(0)

    audio_duration = len(wavs[0]) / sr
    rtf = audio_duration / gen_time if gen_time > 0 else 0
    print(f"Generated {audio_duration:.1f}s audio in {gen_time:.2f}s (RTF: {rtf:.2f})")

    return Response(
        content=buffer.read(),
        media_type="audio/wav",
        headers={
            "X-Generation-Time": f"{gen_time:.2f}s",
            "X-Audio-Duration": f"{audio_duration:.2f}s",
            "X-RTF": f"{rtf:.2f}",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS voice clone API server")
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="Model name or path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    parser.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        help="Path to reference audio file for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of the reference audio",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("QWEN_TTS_HOST", "0.0.0.0"),
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("QWEN_TTS_PORT", "8100")),
        help="Server port (default: 8100)",
    )
    args = parser.parse_args()

    if not args.ref_audio:
        parser.error("--ref-audio is required (or set QWEN_TTS_REF_AUDIO)")

    _config["model"] = args.model
    _config["ref_audio"] = args.ref_audio
    _config["ref_text"] = args.ref_text
    _config["host"] = args.host
    _config["port"] = args.port

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
