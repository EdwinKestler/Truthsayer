"""Combine multimodal cues into a research 'cue load' score.

Weights follow the literature, not a trained classifier:
- Heart-rate change and gaze/saccade-like shifts have the most support
  (Speth et al. 2021 DDPM; Vrij et al. 2015).
- Micro-expression *presence* and commercial voice-stress scores do not
  (DDPM micro-expression t-tests n.s.; Harnsberger 2009 LVA ~chance).
The score is an arousal/concordance index. It is not P(lie).
"""
from __future__ import annotations


class CueFusion:
    WEIGHTS = {
        "bpm_change": 0.22,
        "gaze": 0.16,
        "blink": 0.10,
        "lips": 0.08,
        "hand": 0.08,
        "vrms_burst": 0.12,
        "pitch_rise": 0.12,
        "voice_energy": 0.07,
        "motion_energy": 0.05,
    }

    def __init__(self):
        self.score = 0.0
        self.active = {}

    def update(self, tells, rppg, vrms, voice, motion_energy=0.0, bpm_change_text=""):
        flags = {
            "bpm_change": bool(bpm_change_text),
            "gaze": "gaze" in tells,
            "blink": "blinking" in tells,
            "lips": "lips" in tells,
            "hand": "hand" in tells,
            "vrms_burst": bool(vrms.get("burst")),
            "pitch_rise": bool(voice.get("voiced") and voice.get("pitch_delta", 0) > 20),
            "voice_energy": bool(voice.get("voiced") and abs(voice.get("rms_delta", 0)) > 6),
            "motion_energy": motion_energy > 2.5,
        }
        raw = sum(self.WEIGHTS[k] for k, on in flags.items() if on)
        self.score = min(1.0, raw / 0.7)
        self.active = {k: v for k, v in flags.items() if v}
        return self.snapshot()

    def snapshot(self):
        return {
            "score": self.score,
            "active": self.active,
            "label": self.label(),
        }

    def label(self):
        if self.score < 0.25:
            return "low cue load"
        if self.score < 0.55:
            return "moderate cue load"
        return "high cue load"
