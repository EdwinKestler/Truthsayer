# GPU environment (Windows `.venv`)

TensorFlow **2.10** is the last native Windows GPU build. It needs **CUDA 11.2** and **cuDNN 8.1**. The RTX 3090 Ti driver (591.x, CUDA capability 13.1) can run that older toolkit.

This repo vendors those libraries **inside `.venv`** so you do not need a system CUDA 11.2 install.

## Create / repair the venv

Needs the Python 3.9 interpreter at `.conda\python.exe` (only used as the venv base).

```powershell
cd E:\Truthsayer
& .\.conda\python.exe scripts\setup_gpu_venv.py
```

That will:

1. Create `.venv` from Python 3.9
2. Download conda-forge `cudatoolkit 11.2.2`, `cudnn 8.1.0.77`, and NVIDIA `cuda-nvcc` 11.8 (`ptxas.exe`) into `.cache/cuda`
3. Extract runtimes to `.venv\Library\bin` (and `nvvm/`)
4. Install `requirements-gpu.txt` (`tensorflow==2.10.0`, MediaPipe 0.10.9, …)
5. Check that `tf.config.list_physical_devices("GPU")` is non-empty

## Activate and run

```powershell
cd E:\Truthsayer
.\.venv\Scripts\Activate.ps1
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python intercept.py --input 0 --landmarks 1 --flip 1
```

Or without activating:

```powershell
& .\.venv\Scripts\python.exe intercept.py --input 0 --landmarks 1 --flip 1
```

`Lib\site-packages\sitecustomize.py` calls `os.add_dll_directory` so CUDA is found even when you invoke `python.exe` directly.

## Verified on this machine

- GPU: NVIDIA GeForce RTX 3090 Ti (~21.6 GB visible to TF)
- `tensorflow 2.10.0`, `numpy 1.23.5`, `protobuf 3.20.3`
- Device string: `/job:localhost/replica:0/task:0/device:GPU:0`

`protobuf 3.20.3` is slightly newer than TF 2.10’s pin (`<3.20`). That is required for MediaPipe 0.10.9. TensorFlow still loads and uses the GPU.

`ptxas.exe` (CUDA assembler, from CUDA 11.8 nvcc) is vendored next to the runtime so TensorFlow can compile GPU kernels. Without it you get `Couldn't invoke ptxas.exe --version` and a slower driver-JIT fallback.

Do **not** let pip upgrade NumPy to 2.x; TF 2.10 will crash.

## Why not CUDA 13 from winget?

`winget` only offers CUDA Toolkit 13.x. TensorFlow 2.10 looks for `cudart64_110.dll` and friends. A 13.x toolkit will not satisfy those names.
