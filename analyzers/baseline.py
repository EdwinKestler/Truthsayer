"""Dual talking calibration: truth-talk vs directed-fabrication, then freeze.

Replaces silent 120-frame warmup and whole-session z-scores. After a timed
block of factual answers and a block of instructed false answers, μ/σ are
locked for the sitting. Interview utterances are scored against those
templates. This is a within-subject baseline, not P(lie).
See docs/SCIENCE.md (CTB / directed-lie comparison; GLD cross-task limits).
"""
from __future__ import annotations

import time

FEATURE_KEYS = (
    "bpm_mean",
    "bpm_delta",
    "vrms_mean",
    "blink_rate",
    "gaze_count",
    "pitch_delta_mean",
    "cue_peak",
)

TRUE_PROMPTS = (
    "Answer truthfully: What is the capital of France?",
    "Answer truthfully: How many days are in a week?",
    "Answer truthfully: What color is a stop sign?",
    "Answer truthfully: What is 2 + 2?",
    "Answer truthfully: What planet do we live on?",
    "Answer truthfully: What is today's weekday?",
)

LIE_PROMPTS = (
    "Invent a FALSE answer: Name any capital except the real one for France.",
    "Lie on purpose: How many days are in a week? Give the wrong number.",
    "Invent a FALSE answer: What color is a stop sign?",
    "Lie on purpose: What is 2 + 2? Say the wrong number.",
    "Invent a FALSE hometown, then a false number of siblings.",
    "Lie on purpose: What planet do we live on? Name a different one.",
)


def utterance_vec(utt):
    if not utt:
        return None
    return [float(utt.get(k) or 0.0) for k in FEATURE_KEYS]


def _mean_std(rows):
    if not rows:
        return None, None
    n = len(rows)
    dim = len(rows[0])
    mu = [sum(r[i] for r in rows) / n for i in range(dim)]
    sd = []
    for i in range(dim):
        var = sum((r[i] - mu[i]) ** 2 for r in rows) / max(n - 1, 1)
        sd.append(max(var ** 0.5, 1e-3))
    return mu, sd


def _dist(x, mu, sd):
    if x is None or mu is None or sd is None:
        return None
    return sum(((x[i] - mu[i]) / sd[i]) ** 2 for i in range(len(x))) ** 0.5


def _mean_abs_z(x, mu, sd):
    if x is None or mu is None or sd is None:
        return None
    return sum(abs(x[i] - mu[i]) / sd[i] for i in range(len(x))) / len(x)


class DualBaseline:
    def __init__(self, true_sec=30.0, lie_sec=30.0, warmup_frames=120):
        self.true_sec = max(0.0, float(true_sec))
        self.lie_sec = max(0.0, float(lie_sec))
        self.warmup_frames = int(warmup_frames)
        self.phase = "warmup"
        self.t0 = time.time()
        self.warmup_n = 0
        self.true_rows = []
        self.fab_rows = []
        self.mu_true = self.sd_true = None
        self.mu_fab = self.sd_fab = None
        self.last_score = {}

    def start(self):
        if self.true_sec > 0:
            self.phase = "cal_true"
        elif self.lie_sec > 0:
            self.phase = "cal_lie"
        else:
            self.phase = "warmup"
        self.t0 = time.time()
        return self.phase

    @property
    def ready(self):
        if self.phase == "frozen":
            return True
        if self.phase == "warmup":
            return self.warmup_n >= self.warmup_frames
        return False

    def remaining(self):
        elapsed = time.time() - self.t0
        if self.phase == "cal_true":
            return max(0.0, self.true_sec - elapsed)
        if self.phase == "cal_lie":
            return max(0.0, self.lie_sec - elapsed)
        return 0.0

    def prompt(self):
        elapsed = time.time() - self.t0
        idx = int(elapsed / 6.0)
        if self.phase == "cal_true":
            return TRUE_PROMPTS[idx % len(TRUE_PROMPTS)]
        if self.phase == "cal_lie":
            return LIE_PROMPTS[idx % len(LIE_PROMPTS)]
        return ""

    def note_warmup_face(self, had_face):
        if self.phase == "warmup" and had_face:
            self.warmup_n += 1

    def ingest(self, utt):
        if not utt or not utt.get("closed"):
            return
        vec = utterance_vec(utt)
        if vec is None:
            return
        if self.phase == "cal_true":
            self.true_rows.append(vec)
        elif self.phase == "cal_lie":
            self.fab_rows.append(vec)

    def handle_key(self, key):
        if key in (ord("n"), ord("N"), 13, 32):
            self.advance()
            return True
        return False

    def tick(self):
        if self.phase in ("cal_true", "cal_lie") and self.remaining() <= 0:
            self.advance()

    def advance(self):
        if self.phase == "cal_true":
            if self.lie_sec > 0:
                self.phase = "cal_lie"
                self.t0 = time.time()
            else:
                self.freeze()
        elif self.phase == "cal_lie":
            self.freeze()

    def freeze(self):
        self.mu_true, self.sd_true = _mean_std(self.true_rows)
        self.mu_fab, self.sd_fab = _mean_std(self.fab_rows)
        self.phase = "frozen"
        print("Calibration frozen: {} truth-talk samples, {} directed-fab samples".format(
            len(self.true_rows), len(self.fab_rows)))

    def score(self, utt):
        self.last_score = {}
        if self.phase != "frozen" or not utt:
            return self.last_score
        x = utterance_vec(utt)
        z_true = _mean_abs_z(x, self.mu_true, self.sd_true)
        d_true = _dist(x, self.mu_true, self.sd_true)
        d_fab = _dist(x, self.mu_fab, self.sd_fab)
        closer = None
        if d_true is not None and d_fab is not None:
            closer = "truth-talk" if d_true <= d_fab else "directed-fab"
        elif d_true is not None:
            closer = "truth-talk"
        contrast = None
        if d_true is not None and d_fab is not None:
            contrast = d_true - d_fab
        parts = []
        if z_true is not None:
            parts.append("vs truth-talk z={:.1f}".format(z_true))
        if closer:
            parts.append("closer: {}".format(closer))
        elif self.mu_true is None:
            parts.append("truth-talk template empty")
        hud = " | ".join(parts)
        self.last_score = {
            "z_true": z_true,
            "d_true": d_true,
            "d_fab": d_fab,
            "contrast": contrast,
            "closer": closer,
            "hud": hud,
        }
        return self.last_score

    def snapshot(self):
        return {
            "phase": self.phase,
            "ready": self.ready,
            "remaining": self.remaining(),
            "prompt": self.prompt(),
            "n_true": len(self.true_rows),
            "n_fab": len(self.fab_rows),
            "warmup_n": self.warmup_n,
            "warmup_need": self.warmup_frames,
            "score": dict(self.last_score),
        }
