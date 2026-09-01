"""Real-time Eulerian motion magnification on a face crop.

Wu et al., "Eulerian Video Magnification for Revealing Subtle Changes in the
World", SIGGRAPH 2012. Used here as a microscope for low-amplitude facial
motion (micro-movements, pulse-related skin motion). Magnification makes
motion easier to inspect; it does not classify deception.
"""
from __future__ import annotations

import cv2
import numpy as np


class MotionMagnifier:
    def __init__(self, size=96, buffer=48, alpha=18.0, lo_hz=0.8, hi_hz=5.0, fps=30.0):
        self.size = int(size)
        self.buffer_len = int(buffer)
        self.alpha = float(alpha)
        self.lo_hz = float(lo_hz)
        self.hi_hz = float(hi_hz)
        self.fps = float(fps)
        self.frames = np.zeros((self.buffer_len, self.size, self.size, 3), dtype=np.float32)
        self.count = 0
        self.energy = 0.0
        self.panel = None

    def update(self, bgr_face, fps=None):
        if bgr_face is None or getattr(bgr_face, "size", 0) == 0:
            return self.panel, self.energy
        if fps:
            self.fps = float(fps)

        small = cv2.resize(bgr_face, (self.size, self.size), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = small
        self.count = min(self.count + 1, self.buffer_len)
        if self.count < 16:
            self.panel = small.astype(np.uint8)
            return self.panel, self.energy

        clip = self.frames[-self.count:]
        residual = self._bandpass(clip)
        mag = np.clip(clip[-1] + self.alpha * residual[-1], 0, 255)
        self.energy = float(np.mean(np.abs(residual[-1])))
        self.panel = mag.astype(np.uint8)
        return self.panel, self.energy

    def _bandpass(self, clip):
        """FFT bandpass along time at every pixel (coarse Eulerian filter)."""
        n = clip.shape[0]
        rate = max(8.0, self.fps)
        freqs = np.fft.rfftfreq(n, d=1.0 / rate)
        spec = np.fft.rfft(clip, axis=0)
        mask = (freqs >= self.lo_hz) & (freqs <= self.hi_hz)
        spec[~mask] = 0
        return np.fft.irfft(spec, n=n, axis=0)
