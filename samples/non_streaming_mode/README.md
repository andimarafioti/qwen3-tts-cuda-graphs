# non_streaming_mode Comparison Samples

This folder compares **non_streaming_mode=True** vs **False** for ICL voice cloning
(reference audio in context, `xvec_only=False`).

**Model**
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

**Settings**
- `max_new_tokens=168`
- `temperature=0.9`, `top_k=50`, `top_p=1.0`
- `do_sample=True`
- `language="English"`
- Seeds: `1337 + ref_index*10 + prompt_index` (same seed used for both modes)

**Reference audio**
- `ref_audio.wav`
- `ref_audio_2.wav`
- `ref_audio_3.wav`

**Reference transcripts (used for ICL)**
- ref_audio: "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed."
- ref_audio_2: "when to leave and where to go. It's not Shakespeare. It does not speak in memorable lines. My inner voice always gives it to me straight,"
- ref_audio_3: "Don't be deceived by the name. There is nothing cuddly about this particular teddy bear. In fact, it's the most dangerous plant in the desert."

**Prompts**
1. "I left the window open last night, and the rain made a soft rhythm on the roof while I tried to sleep."
2. "Tomorrow is the first day of spring, so I'm taking a long walk and bringing a notebook to write down ideas."

**Filename pattern**
`icl_<ref_key>_gen<1|2>_nsm_<true|false>.wav`
