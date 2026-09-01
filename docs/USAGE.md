# Usage

Python 3.9. Prefer the GPU `.venv` (TensorFlow 2.10 + CUDA 11.2). **Not** Python 2.

```powershell
cd E:\Truthsayer
.\.venv\Scripts\Activate.ps1
python intercept.py -h
```

If `.venv` does not exist yet, run `python scripts\setup_gpu_venv.py` using the 3.9 interpreter at `.conda\python.exe`. See [GPU.md](GPU.md).

Quit the preview with `Q`, or stop the process with Ctrl+C.

## Inputs

| `--input` / `-i` | Meaning |
|---|---|
| `0` (default) | Camera index 0 |
| `2` | Camera index 2 |
| `path\to\file.mp4` | Video file (FPS taken from the container) |
| `x y width height` | Screen crop via `mss` (four integers) |

`--second` / `-s` opens a **second** camera or file for mirroring prompts (“Blink less”, “Too close”). Device `0` is valid; do not omit it just because zero is falsy.

## Analysis flags

| Flag | Default | Effect |
|---|---|---|
| `--landmarks` / `-l` | off | Draw Face Mesh + hands |
| `--bpm` / `-b` | off | Extra matplotlib chart of raw cheek color |
| `--flip` / `-f` | off | Selfie view (in-place, overlays stay readable) |
| `--magnify` | **on** | Eulerian motion-magnified face panel |
| `--no-magnify` | | Disable that panel |
| `--voice` / `-v` | off | Microphone pitch, RMS, peak frequency |
| `--text` | off | Transcript for lexical markers (optional text channel) |
| `--log` | off | JSONL audit log of cues + ReAct thoughts under `logs/` |
| `--cal-true` | 30 | Seconds of **truthful** general-knowledge talk (session baseline). `0` skips. |
| `--cal-lie` | 30 | Seconds of **instructed false** answers. `0` skips. Both `0` restores the old 120-frame silent warmup. |
| `--record` / `-r` | off | Timestamped MJPG AVI in the current directory |
| `--ttl` / `-t` | 30 | Frames each on-screen tell stays visible |

POS pulse, VRMS, cue-load bar, and the rPPG waveform are always computed when a face is present.

## Examples

Webcam with landmarks, selfie view, voice, calibration log:

```powershell
python intercept.py --input 0 --landmarks 1 --flip 1 --voice --log
```

Skip dual calibration (old 120-frame silent warmup):

```powershell
python intercept.py --input 0 --cal-true 0 --cal-lie 0
```

Video file + interviewer camera for mirroring:

```powershell
python intercept.py -i "C:\clips\interview.mp4" --second 0
```

Screen region (e.g. a video-call window):

```powershell
python intercept.py --input 100 80 1280 720 --no-magnify
```

## Reading the HUD

- **Cue load** (top bar + meter marker): how many *weighted* cues are currently active. Not P(lie).
- **POS BPM**: remote heart rate from cheek color. Ignore values until calibration finishes and SNR is not tiny.
- **rPPG pulse** (bottom-left waveform): the POS signal. Should look quasi-periodic at rest.
- **VRMS**: landmark motion energy vs the subject’s own baseline. `x2.2+` triggers “Facial motion burst.”
- **Pitch / Voice RMS**: only with `--voice`, and only while the mic hears a voiced frame.
- **motion x18** (bottom-right): magnified face crop. Use it to *inspect* micro-movements, not as a verdict.
- **CAL TRUE / CAL LIE** (first ~60 s): interviewer prompts. Subject answers factually, then invents false answers. `N` skips a phase, Enter/Space advances, `Q` quits. Templates then **freeze** for the rest of the sitting.
- **vs truth-talk z=… | closer: …** (after freeze): this utterance vs the calibration templates. `directed-fab` means it resembles the instructed-false block, not that the person is lying.
- **utterance …** (bottom-left, after calibration): last voiced statement — duration, ΔBPM, gaze count, VRMS, peak cue-load. A leading `...` means the utterance is still open. Not a lie label.
- Left-side tells (after calibration is **frozen**): BPM change, blink change, gaze change, lips, hand, pitch rise, etc. `Human review flagged` only when ReAct sees discordant *supported* channels — never for face/motion-only.

Bright light, a still head, and visible cheeks improve rPPG. Talking, walking, or rolling the chair will junk both pulse and VRMS.

## Tests

```powershell
.\.venv\Scripts\python.exe run_smoke_tests.py
```

## Environment notes

- GPU `.venv`: TensorFlow 2.10 sees the RTX 3090 Ti (~21.6 GB). See [GPU.md](GPU.md).
- Do not upgrade NumPy to 2.x in that venv; TF 2.10 will crash.
- `sounddevice` is required only for `--voice`.
