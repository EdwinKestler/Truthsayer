"""Plane-orthogonal-to-skin (POS) remote photoplethysmography.

Wang, den Brinker, Stuijk, de Haan, "Algorithmic Principles of Remote PPG",
IEEE TBME 2017. On the DDPM interview corpus POS reached MAE ~3.16 bpm
(Speth et al., 2021). This is a heart-rate estimator, not a lie detector.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


class FacialRPPG:
    def __init__(self, window=150, fps_fallback=30.0):
        self.window = int(window)
        self.fps_fallback = float(fps_fallback)
        self.rgb = np.zeros((self.window, 3), dtype=np.float64)
        self.count = 0
        self.signal = np.zeros(self.window, dtype=np.float64)
        self.bpm = 0.0
        self.snr = 0.0
        self.last_sample = 0.0

    def update(self, cheek_bgr_a, cheek_bgr_b, fps=None):
        """Append cheek RGB means and recompute POS pulse + BPM."""
        roi_parts = [c for c in (cheek_bgr_a, cheek_bgr_b) if c is not None and getattr(c, "size", 0) > 0]
        if not roi_parts:
            return self.snapshot()

        stacked = np.concatenate([p.reshape(-1, 3) for p in roi_parts], axis=0)
        mean_bgr = stacked.mean(axis=0)
        mean_rgb = mean_bgr[::-1]
        if not np.all(np.isfinite(mean_rgb)):
            return self.snapshot()

        self.rgb = np.roll(self.rgb, -1, axis=0)
        self.rgb[-1] = mean_rgb
        self.count = min(self.count + 1, self.window)
        if self.count < 24:
            return self.snapshot()

        C = self.rgb[-self.count:]
        mu = C.mean(axis=0)
        mu[mu == 0] = 1.0
        Cn = C / mu
        # POS projection matrix (Wang 2017)
        S = np.stack((Cn[:, 1] - Cn[:, 2], -2.0 * Cn[:, 0] + Cn[:, 1] + Cn[:, 2]), axis=1)
        std1 = float(S[:, 0].std()) or 1.0
        std2 = float(S[:, 1].std()) or 1.0
        h = S[:, 0] + (std1 / std2) * S[:, 1]
        h = h - h.mean()

        rate = float(fps) if fps else self.fps_fallback
        rate = max(8.0, min(rate, 90.0))
        lo, hi = 0.7 / (rate * 0.5), 3.0 / (rate * 0.5)
        lo = min(max(lo, 1e-3), 0.45)
        hi = min(max(hi, lo + 0.05), 0.49)
        try:
            sos = butter(2, [lo, hi], btype="band", output="sos")
            h = sosfiltfilt(sos, h)
        except ValueError:
            pass

        pad = self.window - h.size
        self.signal = np.concatenate([np.zeros(pad), h]) if pad > 0 else h[-self.window:]
        self.last_sample = float(h[-1])

        freqs = np.fft.rfftfreq(h.size, d=1.0 / rate)
        spec = np.abs(np.fft.rfft(h * np.hanning(h.size)))
        band = (freqs >= 0.7) & (freqs <= 3.0)
        if not np.any(band) or spec[band].sum() <= 0:
            return self.snapshot()
        peak_i = np.argmax(spec[band])
        self.bpm = float(60.0 * freqs[band][peak_i])
        total = float(spec[band].sum()) + 1e-9
        self.snr = float(spec[band][peak_i] / total)
        return self.snapshot()

    def snapshot(self):
        return {
            "bpm": self.bpm,
            "snr": self.snr,
            "sample": self.last_sample,
            "waveform": self.signal.copy(),
        }

    def pulse_peaks(self, fps=None):
        rate = float(fps) if fps else self.fps_fallback
        distance = max(3, int(rate * 0.35))
        peaks, _ = find_peaks(self.signal, distance=distance, prominence=0.05)
        return peaks
