# Architecture

```
camera / file / screen ──► intercept.py (main loop)
                              │
                              ├─ MediaPipe Face Mesh + Hands
                              ├─ FER mood (background thread)
                              ├─ analyzers.rppg.FacialRPPG      POS pulse + waveform
                              ├─ analyzers.vrms.FacialVRMS      landmark velocity RMS
                              ├─ analyzers.motion.MotionMagnifier  Eulerian face crop
                              ├─ analyzers.voice.VoiceAnalyzer   F0 / RMS / peak Hz
                              ├─ analyzers.linguistics           optional transcript markers
                              ├─ analyzers.fusion.CueFusion      cue-load score
                              ├─ analyzers.utterance             voiced statement windows
                              ├─ analyzers.baseline.DualBaseline  30s+30s talking cal, then freeze
                              ├─ agents.react_agent              ReAct trace + review flag
                              └─ analyzers.session_log           JSONL audit log
                                         │
                                         ▼
                              OpenCV HUD + optional matplotlib BPM chart
```

## Entry points

| File | Role |
|---|---|
| `intercept.py` | **Product.** Live analysis, CLI, recording, second-camera mirroring. |
| `modular_face_analysis.py` | Class-based refactor of the original loop. Prefer `intercept.py`. |
| `analyzers/` | Isolated signal math so it can be unit-tested without a camera. |
| `run_smoke_tests.py` | Headless checks (helpers, demo.png, optional camera). |
| `scripts/setup_gpu_venv.py` | Create `.venv`, vendor CUDA/cuDNN/ptxas, install `requirements-gpu.txt`. |
| `agents/react_agent.py` | ReAct traces; face-only never `flag_review`. |

## Analyzer contracts

Every analyzer exposes `update(...)` / `poll()` and `snapshot()` returning a dict. None of them raise on missing face or silence; they return zeros and let the HUD show “calibrating” or “silent.”

### POS rPPG (`analyzers/rppg.py`)

1. Spatial-mean RGB of both cheek ROIs.
2. Rolling window (default 150 samples).
3. POS projection `S = [G−B, −2R+G+B]`, then `h = S1 + (σ1/σ2) S2`.
4. Butterworth 0.7–3 Hz (42–180 BPM).
5. BPM = 60 × peak FFT bin in that band.

Cheeks come from MediaPipe points 449/350/429/280 and 121/229/50/209. Empty crops are skipped so NaNs never enter `find_peaks`.

### Motion magnification (`analyzers/motion.py`)

Face box from landmarks 10, 152, 234, 454 → 96×96 Gaussian-blurred crop → 48-frame buffer → FFT bandpass 0.8–5 Hz along time → amplify ×18. The residual energy is a cheap micro-motion scalar.

### VRMS (`analyzers/vrms.py`)

22 landmarks (brows, eyes, nose, mouth). Per-frame velocity, then RMS over 15 frames vs a 90-frame baseline. `ratio > 2.2` → “Facial motion burst.”

### Voice (`analyzers/voice.py`)

Optional `sounddevice` InputStream at 16 kHz. Autocorrelation F0 (80–400 Hz), RMS dB, rFFT peak 80–800 Hz. Deltas are vs the median of the first half of the voiced history (subject baseline).

### Fusion (`analyzers/fusion.py`)

Weighted sum of binary flags. Highest weights: BPM change (0.22) and gaze (0.16), matching DDPM. Voice and magnified-motion energy are down-weighted. Output is clamped to `[0, 1]` and labeled *low / moderate / high cue load*.

### Dual talking baseline (`analyzers/baseline.py`)

Timed calibration, then freeze: `cal_true` (~30 s factual talk) → `cal_lie` (~30 s directed false answers) → `frozen`. Closed P0 utterances in each phase fill μ/σ. Interview utterances get `z_true` (mean |z| vs truth-talk) and `closer` (truth-talk vs directed-fab). Keys: `N` skip, Enter/Space next. `--cal-true 0 --cal-lie 0` falls back to 120 face-frame warmup.

### Utterance tracker (`analyzers/utterance.py`)

Voiced frames (mic F0/RMS, or VRMS>1.35 without a mic) accumulate POS BPM, VRMS ratio, gaze/blink flags, pitch Δ, and cue-load. After 0.8 s of silence and ≥0.6 s of speech, a **closed statement summary** is passed to ReAct and the HUD. Face/motion-only clusters never `flag_review` (GLD 2104.12345).

## Threading

- FER mood: one daemon worker, frame copied, `mood_lock` around the flag and label.
- Microphone: PortAudio callback fills a 1 s ring buffer; `poll()` is lock-protected and runs on the UI thread.

## Color space

Displayed and recorded frames stay BGR. MediaPipe gets an RGB copy inside `find_face_and_hands`. Screen capture (`mss`) drops alpha and does **not** convert to RGB in `main()` — a previous double `BGR2RGB` broke both display and rPPG.
