"""Voice RMS, pitch (F0), and dominant-frequency analysis.

Pitch often rises with anger/fear in ~70% of people (Ekman). RMS tracks
loudness. Neither is a validated lie detector: commercial voice-stress
systems have performed at chance (Harnsberger et al., 2009, JFS).
"""
from __future__ import annotations

import threading

import numpy as np

try:
    import sounddevice as sd
    HAS_SD = True
except Exception:
    sd = None
    HAS_SD = False


def estimate_pitch(x, sr, fmin=80.0, fmax=400.0):
    """Autocorrelation F0. Returns 0 if unvoiced or silent."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 64:
        return 0.0
    x = x - x.mean()
    peak = float(np.max(np.abs(x)))
    if peak < 5e-4:
        return 0.0
    x = x / peak
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2:]
    lag_min = max(1, int(sr / fmax))
    lag_max = min(int(sr / fmin), corr.size - 1)
    if lag_max <= lag_min:
        return 0.0
    lag = lag_min + int(np.argmax(corr[lag_min:lag_max]))
    if corr[0] <= 0 or corr[lag] < 0.35 * corr[0]:
        return 0.0
    return float(sr / lag)


def rms_db(x):
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def dominant_hz(x, sr):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 64:
        return 0.0
    win = x * np.hanning(x.size)
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    band = (freqs >= 80.0) & (freqs <= 800.0)
    if not np.any(band):
        return 0.0
    return float(freqs[band][np.argmax(spec[band])])


class VoiceAnalyzer:
    def __init__(self, sr=16000, block=1024):
        self.sr = int(sr)
        self.block = int(block)
        self._buf = np.zeros(self.sr, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = None
        self._running = False
        self.pitch = 0.0
        self.rms = -80.0
        self.peak_hz = 0.0
        self.voiced = False
        self.pitch_baseline = 0.0
        self.rms_baseline = -80.0
        self._pitch_hist = []
        self._rms_hist = []
        self.pitch_delta = 0.0
        self.rms_delta = 0.0
        self.available = HAS_SD
        self.status = "mic ready" if HAS_SD else "sounddevice not installed"

    def start_mic(self):
        if not HAS_SD:
            self.status = "sounddevice not installed"
            return False

        def callback(indata, frames, time_info, status):
            mono = indata[:, 0] if indata.ndim > 1 else indata
            with self._lock:
                n = min(mono.size, self._buf.size)
                self._buf = np.roll(self._buf, -n)
                self._buf[-n:] = mono[:n]

        try:
            self._stream = sd.InputStream(
                samplerate=self.sr, channels=1, dtype="float32",
                blocksize=self.block, callback=callback
            )
            self._stream.start()
            self._running = True
            self.status = "mic live"
            return True
        except Exception as exc:
            self.status = "mic error: {}".format(exc)
            self.available = False
            return False

    def ingest(self, samples, sr=None):
        """Push PCM from a file decoder (ffpyplayer) instead of the mic."""
        if sr and int(sr) != self.sr:
            # crude resample
            x = np.asarray(samples, dtype=np.float32).ravel()
            n_out = int(round(x.size * float(self.sr) / float(sr)))
            if n_out < 8:
                return
            idx = np.linspace(0, x.size - 1, n_out)
            samples = np.interp(idx, np.arange(x.size), x)
        x = np.asarray(samples, dtype=np.float32).ravel()
        with self._lock:
            n = min(x.size, self._buf.size)
            self._buf = np.roll(self._buf, -n)
            self._buf[-n:] = x[-n:]

    def poll(self):
        with self._lock:
            chunk = self._buf.copy()
        self.rms = rms_db(chunk[-self.block * 4:])
        voiced_chunk = chunk[-int(self.sr * 0.08):]
        self.pitch = estimate_pitch(voiced_chunk, self.sr)
        self.peak_hz = dominant_hz(voiced_chunk, self.sr)
        self.voiced = self.pitch > 0 and self.rms > -45.0

        if self.voiced:
            self._pitch_hist.append(self.pitch)
            self._pitch_hist = self._pitch_hist[-200:]
            self._rms_hist.append(self.rms)
            self._rms_hist = self._rms_hist[-200:]
            if len(self._pitch_hist) >= 20:
                self.pitch_baseline = float(np.median(self._pitch_hist[: max(20, len(self._pitch_hist) // 2)]))
                self.rms_baseline = float(np.median(self._rms_hist[: max(20, len(self._rms_hist) // 2)]))
                self.pitch_delta = self.pitch - self.pitch_baseline
                self.rms_delta = self.rms - self.rms_baseline
        return self.snapshot()

    def snapshot(self):
        return {
            "pitch": self.pitch,
            "rms": self.rms,
            "peak_hz": self.peak_hz,
            "voiced": self.voiced,
            "pitch_delta": self.pitch_delta,
            "rms_delta": self.rms_delta,
            "status": self.status,
        }

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
