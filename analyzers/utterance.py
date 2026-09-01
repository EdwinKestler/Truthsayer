"""Aggregate cues over a voiced statement window (GLD 2104.12345).

Rodriguez Diaz et al. label *statements* (~37 frames) rather than single
faces, and collapse a live statement with a frame-fraction rule. We keep
the windowing, not their VGG lie head: on speech end, emit a summary of
POS BPM, gaze, VRMS, pitch, and cue-load. No lie / truth label.
"""
from __future__ import annotations


def is_speaking(voice, vrms, have_mic):
    """True while the subject is likely in an utterance.

    Prefer the pitch tracker; fall back to RMS energy; without a mic use
    a short facial-motion gate (talking moves the mouth/jaw).
    """
    voice = voice or {}
    if have_mic:
        if voice.get("voiced"):
            return True
        if float(voice.get("rms") or -80) > -42:
            return True
        return False
    return float((vrms or {}).get("ratio") or 1.0) > 1.35


class UtteranceTracker:
    def __init__(self, min_sec=0.6, silence_sec=0.8):
        self.min_sec = float(min_sec)
        self.silence_sec = float(silence_sec)
        self.open = False
        self.silence_frames = 0
        self.frames = []
        self.fps = 30.0
        self.last = None
        self.live = None

    def update(self, speaking, sample, fps=None):
        if fps:
            self.fps = float(fps) or self.fps
        ended = None
        if speaking:
            if not self.open:
                self.open = True
                self.frames = []
            self.silence_frames = 0
            self.frames.append(dict(sample))
            self.live = self._summarize(self.frames, closed=False)
        elif self.open:
            self.silence_frames += 1
            need = max(3, int(self.silence_sec * self.fps))
            min_n = max(4, int(self.min_sec * self.fps))
            if self.silence_frames >= need:
                if len(self.frames) >= min_n:
                    ended = self._summarize(self.frames, closed=True)
                    self.last = ended
                self.open = False
                self.frames = []
                self.live = None
        return ended

    def current(self):
        return self.live or self.last

    def _summarize(self, frames, closed):
        n = len(frames)
        dur = n / max(self.fps, 1.0)
        bpms = [f["bpm"] for f in frames if (f.get("bpm") or 0) >= 40]
        vrms = [f["vrms_ratio"] for f in frames if f.get("vrms_ratio") is not None]
        loads = [f["cue_load"] for f in frames if f.get("cue_load") is not None]
        pitches = [f["pitch_delta"] for f in frames if f.get("pitch_delta") is not None]
        gaze_n = sum(1 for f in frames if f.get("gaze"))
        blink_n = sum(1 for f in frames if f.get("blink"))
        bpm_mean = sum(bpms) / len(bpms) if bpms else 0.0
        bpm_delta = (bpms[-1] - bpms[0]) if len(bpms) > 1 else 0.0
        vrms_mean = sum(vrms) / len(vrms) if vrms else 0.0
        cue_peak = max(loads) if loads else 0.0
        pitch_d = sum(pitches) / len(pitches) if pitches else 0.0
        hud = "utterance {:.1f}s | ΔBPM {:+.0f} | gaze {} | VRMS x{:.1f} | cue {:.2f}".format(
            dur, bpm_delta, gaze_n, vrms_mean, cue_peak
        )
        return {
            "closed": closed,
            "duration_s": dur,
            "n_frames": n,
            "bpm_mean": bpm_mean,
            "bpm_delta": bpm_delta,
            "vrms_mean": vrms_mean,
            "gaze_count": gaze_n,
            "blink_count": blink_n,
            "blink_rate": blink_n / max(dur, 1e-6),
            "pitch_delta_mean": pitch_d,
            "cue_peak": cue_peak,
            "hud": hud,
        }
