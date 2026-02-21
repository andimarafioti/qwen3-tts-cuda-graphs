#!/usr/bin/env python3
"""
Faster Qwen3-TTS Demo Server

Usage:
    python demo/server.py
    python demo/server.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 7860
    python demo/server.py --no-preload  # skip startup model load
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import tempfile
import threading
import time
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from faster_qwen3_tts import FasterQwen3TTS
except ImportError:
    print("Error: faster_qwen3_tts not found.")
    print("Install with:  pip install -e .  (from the repo root)")
    sys.exit(1)


AVAILABLE_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]

app = FastAPI(title="Faster Qwen3-TTS Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: FasterQwen3TTS | None = None
_model_name: str | None = None
_model_lock = threading.Lock()
_loading = False
_ref_cache: dict[str, str] = {}
_ref_cache_lock = threading.Lock()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_wav_b64(audio: np.ndarray, sr: int) -> str:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    t0 = time.perf_counter()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    wav_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    b64 = base64.b64encode(buf.getvalue()).decode()
    b64_ms = (time.perf_counter() - t1) * 1000
    return b64, wav_ms, b64_ms


def _to_pcm_i16(audio: np.ndarray) -> tuple[bytes, int]:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    i16 = (audio * 32767.0).astype(np.int16)
    return i16.tobytes(), int(i16.shape[0])


def _concat_audio(audio_list) -> np.ndarray:
    if isinstance(audio_list, np.ndarray):
        return audio_list.astype(np.float32).squeeze()
    parts = [np.array(a, dtype=np.float32).squeeze() for a in audio_list if len(a) > 0]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/status")
async def get_status():
    return {
        "loaded": _model is not None,
        "model": _model_name,
        "loading": _loading,
        "available_models": AVAILABLE_MODELS,
    }


@app.post("/load")
async def load_model(model_id: str = Form(...)):
    global _model, _model_name, _loading

    if _model_name == model_id and _model is not None:
        return {"status": "already_loaded", "model": model_id}

    _loading = True

    def _do_load():
        global _model, _model_name, _loading
        try:
            with _model_lock:
                new_model = FasterQwen3TTS.from_pretrained(
                    model_id,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                print("Capturing CUDA graphs…")
                new_model._warmup(prefill_len=100)
                _model = new_model
                _model_name = model_id
                print("CUDA graphs captured — model ready.")
        finally:
            _loading = False

    await asyncio.to_thread(_do_load)
    return {"status": "loaded", "model": model_id}


@app.post("/generate/stream")
async def generate_stream(
    text: str = Form(...),
    language: str = Form("English"),
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    instruct: str = Form(""),
    chunk_size: int = Form(8),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.05),
    ref_audio: UploadFile = File(None),
):
    if _model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")

    model = _model
    tmp_path = None

    if ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(content)
            tmp_path = f.name

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def run_generation():
        try:
            t0 = time.perf_counter()
            first_chunk_t = None
            total_audio_s = 0.0
            voice_clone_ms = 0.0

            if mode == "voice_clone":
                gen = model.generate_voice_clone_streaming(
                    text=text,
                    language=language,
                    ref_audio=tmp_path,
                    ref_text=ref_text,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
            else:
                gen = model.generate_voice_design_streaming(
                    text=text,
                    instruct=instruct,
                    language=language,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )

            # Use timing data from the generator itself (measured after voice-clone
            # encoding, so TTFA and RTF reflect pure LLM generation latency).
            ttfa_ms = None
            total_gen_ms = 0.0

            # Prime generator to capture wall-clock time to first chunk
            first_audio = next(gen, None)
            if first_audio is not None:
                audio_chunk, sr, timing = first_audio
                wall_first_ms = (time.perf_counter() - t0) * 1000
                model_ms = timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
                voice_clone_ms = max(0.0, wall_first_ms - model_ms)
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms

                audio_chunk = _concat_audio(audio_chunk)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

                audio_b64, wav_ms, b64_ms = _to_wav_b64(audio_chunk, sr)
                payload = {
                    "type": "chunk",
                    "audio_b64": audio_b64,
                    "sample_rate": sr,
                    "ttfa_ms": round(ttfa_ms),
                    "voice_clone_ms": round(voice_clone_ms),
                    "wav_ms": round(wav_ms),
                    "b64_ms": round(b64_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
                }
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))

            for audio_chunk, sr, timing in gen:
                # prefill_ms is non-zero only on the first chunk
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms  # already in ms

                audio_chunk = _concat_audio(audio_chunk)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

                audio_b64, wav_ms, b64_ms = _to_wav_b64(audio_chunk, sr)
                payload = {
                    "type": "chunk",
                    "audio_b64": audio_b64,
                    "sample_rate": sr,
                    "ttfa_ms": round(ttfa_ms),
                    "voice_clone_ms": round(voice_clone_ms),
                    "wav_ms": round(wav_ms),
                    "b64_ms": round(b64_ms),
                    "rtf": round(rtf, 3),
                    "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
                }
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))

            rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0
            done_payload = {
                "type": "done",
                "ttfa_ms": round(ttfa_ms) if ttfa_ms else 0,
                "voice_clone_ms": round(voice_clone_ms),
                "rtf": round(rtf, 3),
                "total_audio_s": round(total_audio_s, 3),
                "total_ms": round((time.perf_counter() - t0) * 1000),
            }
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(done_payload))

        except Exception as e:
            import traceback
            err = {"type": "error", "message": str(e), "detail": traceback.format_exc()}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(err))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()

    async def sse():
        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    if _model is None:
        await ws.send_text(json.dumps({"type": "error", "message": "Model not loaded. Click 'Load' first."}))
        await ws.close()
        return

    try:
        cfg = await ws.receive_json()
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "message": f"Invalid init payload: {e}"}))
        await ws.close()
        return

    text = cfg.get("text", "")
    language = cfg.get("language", "English")
    mode = cfg.get("mode", "voice_clone")
    ref_text = cfg.get("ref_text", "")
    instruct = cfg.get("instruct", "")
    chunk_size = int(cfg.get("chunk_size", 8))
    temperature = float(cfg.get("temperature", 0.9))
    top_k = int(cfg.get("top_k", 50))
    repetition_penalty = float(cfg.get("repetition_penalty", 1.05))
    ref_audio_b64 = cfg.get("ref_audio_b64")
    ref_id = cfg.get("ref_id")

    tmp_path = None
    if ref_id:
        with _ref_cache_lock:
            tmp_path = _ref_cache.get(ref_id)
    if not tmp_path and ref_audio_b64:
        try:
            content = base64.b64decode(ref_audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(content)
                tmp_path = f.name
            # cache for reuse
            if ref_id:
                with _ref_cache_lock:
                    _ref_cache[ref_id] = tmp_path
        except Exception as e:
            await ws.send_text(json.dumps({"type": "error", "message": f"Invalid ref_audio_b64: {e}"}))
            await ws.close()
            return

    model = _model
    t0 = time.perf_counter()
    total_audio_s = 0.0
    ttfa_ms = None
    total_gen_ms = 0.0
    voice_clone_ms = 0.0

    try:
        if mode == "voice_clone":
            gen = model.generate_voice_clone_streaming(
                text=text,
                language=language,
                ref_audio=tmp_path,
                ref_text=ref_text,
                chunk_size=chunk_size,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        else:
            gen = model.generate_voice_design_streaming(
                text=text,
                instruct=instruct,
                language=language,
                chunk_size=chunk_size,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

        first_audio = next(gen, None)
        if first_audio is not None:
            audio_chunk, sr, timing = first_audio
            wall_first_ms = (time.perf_counter() - t0) * 1000
            model_ms = timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
            voice_clone_ms = max(0.0, wall_first_ms - model_ms)
            total_gen_ms += timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
            if ttfa_ms is None:
                ttfa_ms = total_gen_ms

            audio_chunk = _concat_audio(audio_chunk)
            dur = len(audio_chunk) / sr
            total_audio_s += dur
            rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

            pcm_bytes, n_samples = _to_pcm_i16(audio_chunk)
            meta = {
                "type": "chunk_meta",
                "sample_rate": int(sr),
                "n_samples": int(n_samples),
                "ttfa_ms": round(ttfa_ms),
                "voice_clone_ms": round(voice_clone_ms),
                "rtf": round(rtf, 3),
                "total_audio_s": round(total_audio_s, 3),
                "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
            }
            await ws.send_text(json.dumps(meta))
            await ws.send_bytes(pcm_bytes)

        for audio_chunk, sr, timing in gen:
            total_gen_ms += timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
            if ttfa_ms is None:
                ttfa_ms = total_gen_ms

            audio_chunk = _concat_audio(audio_chunk)
            dur = len(audio_chunk) / sr
            total_audio_s += dur
            rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0

            pcm_bytes, n_samples = _to_pcm_i16(audio_chunk)
            meta = {
                "type": "chunk_meta",
                "sample_rate": int(sr),
                "n_samples": int(n_samples),
                "ttfa_ms": round(ttfa_ms),
                "voice_clone_ms": round(voice_clone_ms),
                "rtf": round(rtf, 3),
                "total_audio_s": round(total_audio_s, 3),
                "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
            }
            await ws.send_text(json.dumps(meta))
            await ws.send_bytes(pcm_bytes)

        rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0
        done_payload = {
            "type": "done",
            "ttfa_ms": round(ttfa_ms) if ttfa_ms else 0,
            "voice_clone_ms": round(voice_clone_ms),
            "rtf": round(rtf, 3),
            "total_audio_s": round(total_audio_s, 3),
            "total_ms": round((time.perf_counter() - t0) * 1000),
        }
        await ws.send_text(json.dumps(done_payload))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/generate")
async def generate_non_streaming(
    text: str = Form(...),
    language: str = Form("English"),
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    instruct: str = Form(""),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.05),
    ref_audio: UploadFile = File(None),
):
    if _model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Click 'Load' first.")

    model = _model
    tmp_path = None

    if ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(content)
            tmp_path = f.name

    def run():
        t0 = time.perf_counter()
        if mode == "voice_clone":
            audio_list, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=tmp_path,
                ref_text=ref_text,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        else:
            audio_list, sr = model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        elapsed = time.perf_counter() - t0
        audio = _concat_audio(audio_list)
        dur = len(audio) / sr
        return audio, sr, elapsed, dur

    try:
        audio, sr, elapsed, dur = await asyncio.to_thread(run)
        rtf = dur / elapsed if elapsed > 0 else 0.0
        return JSONResponse({
            "audio_b64": _to_wav_b64(audio, sr),
            "sample_rate": sr,
            "metrics": {
                "total_ms": round(elapsed * 1000),
                "audio_duration_s": round(dur, 3),
                "rtf": round(rtf, 3),
            },
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Faster Qwen3-TTS Demo Server")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Model to preload at startup (default: 0.6B-Base)",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip model loading at startup (load via UI instead)",
    )
    args = parser.parse_args()

    if not args.no_preload:
        global _model, _model_name
        print(f"Loading model: {args.model}")
        _model = FasterQwen3TTS.from_pretrained(
            args.model,
            device="cuda",
            dtype=torch.bfloat16,
        )
        _model_name = args.model
        print("Capturing CUDA graphs…")
        _model._warmup(prefill_len=100)
        print(f"Model ready. Open http://localhost:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
