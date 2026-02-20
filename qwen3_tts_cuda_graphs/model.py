"""
Qwen3TTSCudaGraphs: Real-time TTS using CUDA graph capture.

Wrapper class that provides a qwen-tts compatible API while using
CUDA graphs for 6-10x speedup.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class Qwen3TTSCudaGraphs:
    """
    Qwen3-TTS model with CUDA graphs for real-time inference.
    
    Compatible API with qwen-tts Qwen3TTSModel, but uses CUDA graph
    capture for 6-10x speedup on NVIDIA GPUs.
    """
    
    def __init__(
        self,
        base_model,
        predictor_graph,
        talker_graph,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 2048,
    ):
        self.model = base_model  # The qwen-tts Qwen3TTSModel instance
        self.predictor_graph = predictor_graph
        self.talker_graph = talker_graph
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.sample_rate = 12000  # Qwen3-TTS uses 12kHz
        self._warmed_up = False
        self._voice_prompt_cache = {}  # Cache (ref_audio, ref_text) -> (vcp, ref_ids)
        
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = torch.bfloat16,
        attn_implementation: str = "eager",
        max_seq_len: int = 2048,
    ):
        """
        Load Qwen3-TTS model and prepare CUDA graphs.
        
        Args:
            model_name: Model path or HuggingFace Hub ID
            device: Device to use ("cuda" or "cpu")
            dtype: Data type for inference
            attn_implementation: Attention implementation (use "eager" on Jetson)
            max_seq_len: Maximum sequence length for static cache
            
        Returns:
            Qwen3TTSCudaGraphs instance
        """
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            
        if device != "cuda" or not torch.cuda.is_available():
            raise ValueError("CUDA graphs require CUDA device")
        
        logger.info(f"Loading Qwen3-TTS model: {model_name}")
        
        # Import here to avoid dependency issues
        from qwen_tts import Qwen3TTSModel
        from .predictor_graph import PredictorGraph
        from .talker_graph import TalkerGraph
        # Load base model using qwen-tts library
        base_model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cuda:0",
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        
        talker = base_model.model.talker
        talker_config = base_model.model.config.talker_config

        # Extract predictor config from loaded model
        predictor = talker.code_predictor
        pred_config = predictor.model.config
        talker_hidden = talker_config.hidden_size

        # Build CUDA graphs
        logger.info("Building CUDA graphs...")
        predictor_graph = PredictorGraph(
            predictor,
            pred_config,
            talker_hidden,
            device=device,
            dtype=dtype,
        )
        
        talker_graph = TalkerGraph(
            talker.model,
            talker_config,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )
        
        logger.info("CUDA graphs initialized (will capture on first run)")
        
        return cls(
            base_model=base_model,
            predictor_graph=predictor_graph,
            talker_graph=talker_graph,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )
    
    def _warmup(self, prefill_len: int):
        """Warm up and capture CUDA graphs with given prefill length."""
        if self._warmed_up:
            return
            
        logger.info("Warming up CUDA graphs...")
        self.predictor_graph.capture(num_warmup=3)
        self.talker_graph.capture(prefill_len=prefill_len, num_warmup=3)
        self._warmed_up = True
        logger.info("CUDA graphs captured and ready")
    
    def generate(
        self,
        text: str,
        language: str = "English",
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        """
        Generate speech from text using default voice.
        
        Not yet implemented - use generate_voice_clone() instead.
        """
        raise NotImplementedError(
            "Default voice generation not yet implemented. "
            "Use generate_voice_clone() with reference audio."
        )
    
    def _prepare_generation(self, text: str, ref_audio: Union[str, Path], ref_text: str):
        """Prepare inputs for generation (shared by streaming and non-streaming)."""
        input_texts = [f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"]
        input_ids = []
        for t in input_texts:
            inp = self.model.processor(text=t, return_tensors="pt", padding=True)
            iid = inp["input_ids"].to(self.model.device)
            input_ids.append(iid.unsqueeze(0) if iid.dim() == 1 else iid)

        cache_key = (str(ref_audio), ref_text)
        if cache_key in self._voice_prompt_cache:
            vcp, ref_ids = self._voice_prompt_cache[cache_key]
        else:
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text=ref_text
            )
            vcp = self.model._prompt_items_to_voice_clone_prompt(prompt_items)

            ref_ids = []
            rt = prompt_items[0].ref_text
            if rt:
                ref_ids.append(
                    self.model._tokenize_texts([f"<|im_start|>assistant\n{rt}<|im_end|>\n"])[0]
                )

            self._voice_prompt_cache[cache_key] = (vcp, ref_ids)

        m = self.model.model
        tie, tam, tth, tpe = m._build_talker_inputs(
            input_ids=input_ids,
            instruct_ids=None,
            ref_ids=ref_ids,
            voice_clone_prompt=vcp,
            languages=["Auto"],
            speakers=None,
            non_streaming_mode=False,
        )

        if not self._warmed_up:
            self._warmup(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None

        return m, talker, config, tie, tam, tth, tpe

    @torch.inference_mode()
    def generate_voice_clone(
        self,
        text: str,
        language: str,
        ref_audio: Union[str, Path],
        ref_text: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
    ) -> Tuple[list, int]:
        """
        Generate speech with voice cloning using reference audio.

        Args:
            text: Text to synthesize
            language: Target language
            ref_audio: Path to reference audio file
            ref_text: Transcription of reference audio
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            do_sample: Whether to sample
            repetition_penalty: Repetition penalty

        Returns:
            Tuple of ([audio_waveform], sample_rate)
        """
        from .generate import fast_generate

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation(
            text, ref_audio, ref_text
        )

        codec_ids, timing = fast_generate(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=self.predictor_graph,
            talker_graph=self.talker_graph,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        
        if codec_ids is None:
            logger.warning("Generation returned no tokens")
            return [np.zeros(1, dtype=np.float32)], self.sample_rate
        
        # Decode codec IDs to audio
        speech_tokenizer = m.speech_tokenizer
        audio_list, sr = speech_tokenizer.decode({"audio_codes": codec_ids.unsqueeze(0)})
        
        # Convert to numpy arrays (handle both torch tensors and numpy arrays)
        audio_arrays = []
        for a in audio_list:
            if hasattr(a, 'cpu'):  # torch tensor
                audio_arrays.append(a.flatten().cpu().numpy())
            else:  # already numpy
                audio_arrays.append(a.flatten() if hasattr(a, 'flatten') else a)
        
        n_steps = timing['steps']
        audio_duration = n_steps / 12.0  # 12 Hz codec
        total_time = timing['prefill_ms']/1000 + timing['decode_s']
        rtf = audio_duration / total_time if total_time > 0 else 0
        
        logger.info(
            f"Generated {audio_duration:.2f}s audio in {total_time:.2f}s "
            f"({timing['ms_per_step']:.1f}ms/step, RTF: {rtf:.2f})"
        )
        
        return audio_arrays, sr

    @torch.inference_mode()
    def generate_voice_clone_streaming(
        self,
        text: str,
        language: str,
        ref_audio: Union[str, Path],
        ref_text: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
    ) -> Generator[Tuple[np.ndarray, int, dict], None, None]:
        """
        Stream voice-cloned speech generation, yielding audio chunks.

        Same as generate_voice_clone() but yields (audio_chunk, sample_rate, timing)
        tuples every chunk_size codec steps (~chunk_size/12 seconds of audio).

        Args:
            text: Text to synthesize
            language: Target language
            ref_audio: Path to reference audio file
            ref_text: Transcription of reference audio
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            do_sample: Whether to sample
            repetition_penalty: Repetition penalty
            chunk_size: Codec steps per chunk (12 = ~1 second)

        Yields:
            Tuple of (audio_chunk_numpy, sample_rate, timing_dict)
        """
        from .streaming import fast_generate_streaming

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation(
            text, ref_audio, ref_text
        )

        speech_tokenizer = m.speech_tokenizer

        # Hybrid decode strategy:
        # 1. Accumulated decode for early chunks (correct, calibrates samples_per_frame)
        # 2. Sliding window with 25-frame left context once calibrated (constant cost)
        # This avoids boundary artifacts (pops) while keeping decode cost bounded.
        context_frames = 25
        min_calibration_frames = max(context_frames, chunk_size)
        all_codes = []
        prev_audio_len = 0
        samples_per_frame = None

        for codec_chunk, timing in fast_generate_streaming(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=self.predictor_graph,
            talker_graph=self.talker_graph,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
        ):
            all_codes.append(codec_chunk)
            n_new = codec_chunk.shape[0]
            all_flat = torch.cat(all_codes, dim=0)
            n_total = all_flat.shape[0]

            if samples_per_frame is None:
                # Phase 1: accumulated decode until we can calibrate
                audio_list, sr = speech_tokenizer.decode(
                    {"audio_codes": all_flat.unsqueeze(0)}
                )
                audio = audio_list[0]
                if hasattr(audio, 'cpu'):
                    audio = audio.flatten().cpu().numpy()
                else:
                    audio = audio.flatten() if hasattr(audio, 'flatten') else audio

                new_audio = audio[prev_audio_len:]
                prev_audio_len = len(audio)

                if n_total >= min_calibration_frames:
                    samples_per_frame = len(audio) / n_total
            else:
                # Phase 2: sliding window with left context
                ctx_start = max(0, n_total - n_new - context_frames)
                window = all_flat[ctx_start:]
                n_ctx = window.shape[0] - n_new

                audio_list, sr = speech_tokenizer.decode(
                    {"audio_codes": window.unsqueeze(0)}
                )
                audio = audio_list[0]
                if hasattr(audio, 'cpu'):
                    audio = audio.flatten().cpu().numpy()
                else:
                    audio = audio.flatten() if hasattr(audio, 'flatten') else audio

                if n_ctx > 0:
                    ctx_samples = int(round(n_ctx * samples_per_frame))
                    new_audio = audio[ctx_samples:]
                else:
                    new_audio = audio

            yield new_audio, sr, timing
