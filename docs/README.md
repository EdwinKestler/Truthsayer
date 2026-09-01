# Documentation

Truthsayer is a **cue dashboard**, not a lie detector. Start here:

| Doc | What it covers |
|---|---|
| [USAGE.md](USAGE.md) | Install, CLI, HUD, calibration keys |
| [GPU.md](GPU.md) | Windows `.venv` + CUDA 11.2 / cuDNN 8.1 / ptxas |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Modules, data flow, threading, color space |
| [SCIENCE.md](SCIENCE.md) | What each cue is, what the literature allows, design rules |
| [references/LIAR_AI.md](references/LIAR_AI.md) | rUv gist: ReAct / fusion / HITL mapping |
| [references/GLD_2104.12345.md](references/GLD_2104.12345.md) | Why we keep statement windows and skip their face CNN |

Entry point: `intercept.py` in the GPU venv (`requirements-gpu.txt`). Rebuild: `scripts/setup_gpu_venv.py`.
