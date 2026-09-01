# Reference: rUv “Liar Ai” gist

Source: [gist.github.com/ruvnet/481d0f8c2190decead7b14164ae3323c](https://gist.github.com/ruvnet/481d0f8c2190decead7b14164ae3323c)  
Title: *Multi-Modal Lie Detection System using an Agentic ReAct Approach*

We use this as an **architecture checklist**, not as a claim that untrained toy CNNs or BERT-base-uncased detect lies.

## What we take

| Gist component | Truthsayer mapping |
|---|---|
| Vision pipeline | Face Mesh + FER + POS rPPG + VRMS + Eulerian magnification (`intercept.py`, `analyzers/`) |
| Audio pipeline | Pitch F0, RMS, spectral peak (`analyzers/voice.py`) — not Wav2Vec, not a deception classifier |
| Text / NLP pipeline | Optional lexical markers (`analyzers/linguistics.py`). No BERT weights; gist models are untrained stubs |
| Physiological | Remote pulse (POS), not contact GSR |
| Late fusion | Weighted cue load (`analyzers/fusion.py`) |
| ReAct agent | `agents/react_agent.py` — reasoning traces + actions (`continue` / `flag_review` / `need_more_data`) |
| Neuro-symbolic rules | Explicit if/then rules grounded in [SCIENCE.md](../SCIENCE.md), not “increase P(lie)” |
| Human-in-the-loop | `needs_review` when modalities disagree or the case is borderline |
| Audit log | JSONL session log (`analyzers/session_log.py`) |

## What we reject from the gist

- Branding the output **Deceptive / Truthful**. Cue load and a reasoning trace only.
- Averaging untrained `sigmoid` heads as “confidence %”.
- PyTorch + HuggingFace in this Windows TF 2.10 GPU venv (they fight the installed stack).
- The marketing line “world’s most powerful lie detector.” Independent literature (DDPM 2021, DePaulo 2003, Harnsberger 2009) does not support that.

## Gist file layout vs this repo

```
lie_detector/models/vision_model.py   →  analyzers/rppg.py, vrms.py, motion.py + FER
lie_detector/models/audio_model.py    →  analyzers/voice.py
lie_detector/models/text_model.py     →  analyzers/linguistics.py
lie_detector/models/fusion_model.py   →  analyzers/fusion.py
lie_detector/agents/lie_detect_agent.py → agents/react_agent.py
lie_detector/main.py                  →  intercept.py
```
