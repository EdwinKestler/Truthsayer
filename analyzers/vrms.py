"""Visual RMS of facial-landmark velocity (VRMS).

RMS of inter-frame landmark displacement is a compact motion-energy feature
used in facial biomechanics and micro-expression spotting. Spikes can mark
brief muscle bursts; a high baseline can mark restlessness or speech.
Not specific to lying (Speth et al. 2021 found micro-expression *rates*
were not diagnostic of deceit on DDPM).
"""
from __future__ import annotations

import numpy as np

# Brows, eyes, nose, mouth — MediaPipe Face Mesh indices
KEY_LANDMARKS = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
                 33, 133, 362, 263, 1, 4, 61, 291, 13, 14, 17, 0)


class FacialVRMS:
    def __init__(self, window=90, recent=15):
        self.window = int(window)
        self.recent = int(recent)
        dim = len(KEY_LANDMARKS) * 2
        self.pos = np.zeros((self.window, dim), dtype=np.float64)
        self.count = 0
        self.value = 0.0
        self.baseline = 0.0
        self.ratio = 1.0
        self.burst = False

    def update(self, landmarks):
        if landmarks is None:
            return self.snapshot()
        vec = []
        for idx in KEY_LANDMARKS:
            pt = landmarks[idx]
            vec.extend((float(pt.x), float(pt.y)))
        sample = np.asarray(vec, dtype=np.float64)
        self.pos = np.roll(self.pos, -1, axis=0)
        self.pos[-1] = sample
        self.count = min(self.count + 1, self.window)
        if self.count < 4:
            return self.snapshot()

        used = self.pos[-self.count:]
        vel = np.diff(used, axis=0)
        mag = np.sqrt(np.mean(np.square(vel), axis=1))
        self.value = float(np.sqrt(np.mean(np.square(mag[-self.recent:]))))
        self.baseline = float(np.sqrt(np.mean(np.square(mag)))) or 1e-9
        self.ratio = self.value / self.baseline
        self.burst = self.ratio > 2.2 and self.count > self.recent * 2
        return self.snapshot()

    def snapshot(self):
        return {
            "vrms": self.value,
            "baseline": self.baseline,
            "ratio": self.ratio,
            "burst": self.burst,
        }
