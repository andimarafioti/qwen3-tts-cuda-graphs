"""
FasterQwen3TTS: Real-time TTS using CUDA graph capture.

Wrapper class that provides a Qwen3-TTS API while using
CUDA graphs for 6-10x speedup.
"""
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Generator, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class FasterQwen3TTS:
    """
    Qwen3-TTS model with CUDA graphs for real-time inference.
    
    Compatible API with Qwen3TTSModel, but uses CUDA graph
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
            FasterQwen3TTS instance
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
    
    def _load_ref_audio_with_silence(self, ref_audio: Union[str, Path], silence_secs: float = 0.5) -> Tuple[np.ndarray, int]:
        """Load reference audio and append trailing silence.

        The ICL voice-cloning prompt ends with the last codec token of the reference
        audio, so the model's first generated token is conditioned on whatever phoneme
        the reference ends with.  Appending a short silence makes the last tokens
        encode silence instead, preventing that phoneme from bleeding into the start
        of the generated speech.
        """
        audio, sr = sf.read(str(ref_audio), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # convert to mono
        silence = np.zeros(int(silence_secs * sr), dtype=np.float32)
        return np.concatenate([audio, silence]), sr

    def _prepare_generation(
        self,
        text: str,
        ref_audio: Union[str, Path],
        ref_text: str,
        language: str,
        xvec_only: bool = True,
    ):
        """Prepare inputs for generation (shared by streaming and non-streaming).

        Args:
            xvec_only: When True (default), use only the speaker embedding (x-vector) for voice
                cloning instead of the full ICL acoustic prompt. This prevents the model from
                continuing the reference audio's last phoneme and allows natural language switching.
                When False, the full reference audio codec tokens are included in context (ICL mode).
        """
        input_texts = [self.model._build_assistant_text(text)]
        input_ids = self.model._tokenize_texts(input_texts)

        cache_key = (str(ref_audio), ref_text, xvec_only)
        if cache_key in self._voice_prompt_cache:
            vcp, ref_ids = self._voice_prompt_cache[cache_key]
        elif xvec_only:
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text="",
                x_vector_only_mode=True,
            )
            spk_emb = prompt_items[0].ref_spk_embedding
            vcp = dict(
                ref_code=[None],
                ref_spk_embedding=[spk_emb],
                x_vector_only_mode=[True],
                icl_mode=[False],
            )
            ref_ids = [None] * len(input_ids)
            self._voice_prompt_cache[cache_key] = (vcp, ref_ids)
        else:
            ref_audio_input = self._load_ref_audio_with_silence(ref_audio)
            prompt_items = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio_input,
                ref_text=ref_text
            )
            vcp = self.model._prompt_items_to_voice_clone_prompt(prompt_items)

            ref_ids = []
            rt = prompt_items[0].ref_text
            if rt:
                ref_texts = [self.model._build_ref_text(rt)]
                ref_ids.append(self.model._tokenize_texts(ref_texts)[0])
            else:
                ref_ids.append(None)

            self._voice_prompt_cache[cache_key] = (vcp, ref_ids)

        m = self.model.model

        tie, tam, tth, tpe = self._build_talker_inputs_local(
            m=m,
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=vcp,
            languages=[language] if language is not None else ["Auto"],
            speakers=None,
            non_streaming_mode=False,
        )

        if not self._warmed_up:
            self._warmup(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None

        return m, talker, config, tie, tam, tth, tpe

    def _build_talker_inputs_local(
        self,
        m,
        input_ids,
        ref_ids,
        voice_clone_prompt,
        languages,
        speakers,
        non_streaming_mode: bool,
    ):
        """Local copy of upstream talker input building for qwen-tts main repo."""
        talker_input_embeds = [[] for _ in range(len(input_ids))]

        voice_clone_spk_embeds = None
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = m.generate_speaker_prompt(voice_clone_prompt)

        if speakers is None:
            speakers = [None] * len(input_ids)

        trailing_text_hiddens = []
        tts_pad_embed = None

        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker is None:
                    speaker_embed = None
                else:
                    if speaker.lower() not in m.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    spk_id = m.config.talker_config.spk_id[speaker.lower()]
                    speaker_embed = m.talker.get_input_embeddings()(
                        torch.tensor(spk_id, device=m.talker.device, dtype=input_id.dtype)
                    )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            assert language is not None
            if language.lower() == "auto":
                language_id = None
            else:
                if language.lower() not in m.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                language_id = m.config.talker_config.codec_language_id[language.lower()]

            if (
                language.lower() in ["chinese", "auto"]
                and speaker not in ("", None)
                and m.config.talker_config.spk_is_dialect[speaker.lower()]
            ):
                dialect = m.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = m.config.talker_config.codec_language_id[dialect]

            tts_bos_embed, tts_eos_embed, tts_pad_embed = m.talker.text_projection(
                m.talker.get_text_embeddings()(
                    torch.tensor(
                        [[m.config.tts_bos_token_id, m.config.tts_eos_token_id, m.config.tts_pad_token_id]],
                        device=m.talker.device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)

            if language_id is None:
                codec_prefill_list = [[
                    m.config.talker_config.codec_nothink_id,
                    m.config.talker_config.codec_think_bos_id,
                    m.config.talker_config.codec_think_eos_id,
                ]]
            else:
                codec_prefill_list = [[
                    m.config.talker_config.codec_think_id,
                    m.config.talker_config.codec_think_bos_id,
                    language_id,
                    m.config.talker_config.codec_think_eos_id,
                ]]

            codec_input_emebdding_0 = m.talker.get_input_embeddings()(
                torch.tensor(codec_prefill_list, device=m.talker.device, dtype=input_id.dtype)
            )
            codec_input_emebdding_1 = m.talker.get_input_embeddings()(
                torch.tensor(
                    [[m.config.talker_config.codec_pad_id, m.config.talker_config.codec_bos_id]],
                    device=m.talker.device,
                    dtype=input_id.dtype,
                )
            )
            if speaker_embed is None:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, codec_input_emebdding_1], dim=1)
            else:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, speaker_embed.view(1, 1, -1), codec_input_emebdding_1], dim=1)

            _talker_input_embed_role = m.talker.text_projection(
                m.talker.get_text_embeddings()(input_id[:, :3])
            )
            _talker_input_embed = torch.cat(
                (
                    tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
                    tts_bos_embed,
                ),
                dim=1,
            ) + codec_input_emebdding[:, :-1]

            talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

            if (
                voice_clone_prompt is not None
                and voice_clone_prompt.get("ref_code", None) is not None
                and voice_clone_prompt["icl_mode"][index]
            ):
                icl_input_embed, trailing_text_hidden = m.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(m.talker.device),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed,
                        m.talker.text_projection(
                            m.talker.get_text_embeddings()(input_id[:, 3:4])
                        )
                        + codec_input_emebdding[:, -1:],
                    ],
                    dim=1,
                )
                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1]
                    talker_input_embed = torch.cat(
                        [
                            talker_input_embed,
                            torch.cat(
                                (
                                    m.talker.text_projection(
                                        m.talker.get_text_embeddings()(input_id[:, 3:-5])
                                    ),
                                    tts_eos_embed,
                                ),
                                dim=1,
                            )
                            + m.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[m.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                                    device=m.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                            tts_pad_embed
                            + m.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[m.config.talker_config.codec_bos_id]],
                                    device=m.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    trailing_text_hidden = torch.cat(
                        (
                            m.talker.text_projection(
                                m.talker.get_text_embeddings()(input_id[:, 4:-5])
                            ),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )

            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)

        for index, talker_input_embed in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed,
            batch_first=True,
            padding_value=0.0,
        )
        talker_input_embeds = padded_reversed.flip(dims=[1])

        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)

        pad_embedding_vector = tts_pad_embed.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(
            sequences_to_pad,
            batch_first=True,
            padding_value=0.0,
        )
        arange_tensor = torch.arange(max(trailing_text_original_lengths), device=padded_hiddens.device).expand(
            len(trailing_text_original_lengths), -1
        )
        lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens = padded_hiddens

        return talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed

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
        xvec_only: bool = True,
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
            xvec_only: When True (default), use only the speaker embedding for voice cloning.
                This prevents phoneme bleed-through from the reference and allows clean
                language switching. Set to False for full ICL mode (reference audio in context).

        Returns:
            Tuple of ([audio_waveform], sample_rate)
        """
        from .generate import fast_generate

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation(
            text, ref_audio, ref_text, language=language, xvec_only=xvec_only
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
        xvec_only: bool = True,
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
            xvec_only: When True (default), use only the speaker embedding for voice cloning.
                This prevents phoneme bleed-through from the reference and allows clean
                language switching. Set to False for full ICL mode (reference audio in context).

        Yields:
            Tuple of (audio_chunk_numpy, sample_rate, timing_dict)
        """
        from .streaming import fast_generate_streaming

        m, talker, config, tie, tam, tth, tpe = self._prepare_generation(
            text, ref_audio, ref_text, language=language, xvec_only=xvec_only
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
