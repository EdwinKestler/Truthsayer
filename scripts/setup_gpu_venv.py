"""Create .venv with Python 3.9 and vendor CUDA 11.2 + cuDNN 8.1 for TensorFlow 2.10 GPU on Windows.

TensorFlow 2.10 is the last native-Windows GPU build. It needs:
  CUDA 11.2 runtime (cudart64_110.dll, cublas64_11.dll, ...)
  cuDNN 8.1 (cudnn64_8.dll)
Driver 591.x already supports this (nvidia-smi reports CUDA 13.1 capability).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
CACHE = ROOT / ".cache" / "cuda"
LIBBIN = VENV / "Library" / "bin"
SITE = VENV / "Lib" / "site-packages"

CUDA_URL = (
    "https://conda.anaconda.org/conda-forge/win-64/"
    "cudatoolkit-11.2.2-h933977f_10.tar.bz2"
)
CUDNN_URL = (
    "https://conda.anaconda.org/conda-forge/win-64/"
    "cudnn-8.1.0.77-h3e0f4f4_0.tar.bz2"
)
NVCC_URL = (
    "https://conda.anaconda.org/nvidia/win-64/"
    "cuda-nvcc-11.8.89-0.tar.bz2"
)

REQUIRED_DLLS = (
    "cudart64_110.dll",
    "cublas64_11.dll",
    "cublasLt64_11.dll",
    "cufft64_10.dll",
    "curand64_10.dll",
    "cusolver64_11.dll",
    "cusparse64_11.dll",
    "cudnn64_8.dll",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log("cached {}".format(dest.name))
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    log("downloading {} -> {}".format(url, dest.name))

    def hook(block, blocksize, total):
        if total <= 0:
            return
        got = block * blocksize
        pct = min(100.0, 100.0 * got / total)
        if block % 50 == 0 or got >= total:
            log("  {:5.1f}%  {:.0f}/{:.0f} MB".format(pct, got / 1e6, total / 1e6))

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.replace(dest)
    return dest


def extract_library(archive: Path, dest_root: Path) -> None:
    log("extracting {}".format(archive.name))
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/").lstrip("./")
            if name.startswith("Library/"):
                rel = name[len("Library/"):]
            elif name.startswith("bin/") or name.startswith("nvvm/"):
                rel = name
            else:
                continue
            if "libnvvm-samples" in rel:
                continue
            target = dest_root / "Library" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def write_sitecustomize() -> None:
    text = r'''"""Load CUDA 11.2 / cuDNN 8.1 DLLs vendored in this venv."""
import os
from pathlib import Path

_venv = Path(__file__).resolve().parents[2]
_dll = _venv / "Library" / "bin"
if _dll.is_dir():
    os.environ["PATH"] = str(_dll) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("CUDA_PATH", str(_venv / "Library"))
    os.environ.setdefault("CUDA_PATH_V11_2", str(_venv / "Library"))
    add = getattr(os, "add_dll_directory", None)
    if add is not None:
        add(str(_dll))
'''
    path = SITE / "sitecustomize.py"
    path.write_text(text, encoding="utf-8")
    log("wrote {}".format(path))


def patch_activators() -> None:
    extra_ps = (
        "\n# CUDA 11.2 + cuDNN 8.1 vendored in this venv\n"
        '$cudaBin = Join-Path $env:VIRTUAL_ENV "Library\\bin"\n'
        '$env:PATH = "$cudaBin;" + $env:PATH\n'
        '$env:CUDA_PATH = Join-Path $env:VIRTUAL_ENV "Library"\n'
        '$env:CUDA_PATH_V11_2 = $env:CUDA_PATH\n'
    )
    extra_bat = (
        "\r\nREM CUDA 11.2 + cuDNN 8.1 vendored in this venv\r\n"
        "set PATH=%VIRTUAL_ENV%\\Library\\bin;%PATH%\r\n"
        "set CUDA_PATH=%VIRTUAL_ENV%\\Library\r\n"
        "set CUDA_PATH_V11_2=%VIRTUAL_ENV%\\Library\r\n"
    )
    ps1 = VENV / "Scripts" / "Activate.ps1"
    bat = VENV / "Scripts" / "activate.bat"
    if ps1.exists() and "CUDA_PATH_V11_2" not in ps1.read_text(encoding="utf-8", errors="ignore"):
        with ps1.open("a", encoding="utf-8") as fh:
            fh.write(extra_ps)
    if bat.exists() and "CUDA_PATH_V11_2" not in bat.read_text(encoding="utf-8", errors="ignore"):
        with bat.open("a", encoding="utf-8") as fh:
            fh.write(extra_bat)
    log("patched venv activators")


def create_venv(py39: Path) -> Path:
    python = VENV / "Scripts" / "python.exe"
    if python.exists():
        log("venv already exists: {}".format(VENV))
        return python
    log("creating venv with {}".format(py39))
    subprocess.check_call([str(py39), "-m", "venv", str(VENV)])
    return python


def pip_install(python: Path) -> None:
    log("upgrading pip")
    subprocess.check_call([str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    subprocess.check_call([str(python), "-m", "pip", "install", "setuptools==69.5.1"])
    log("installing tensorflow==2.10.0 + numpy 1.23.5 (Windows GPU wheel)")
    subprocess.check_call([str(python), "-m", "pip", "install", "tensorflow==2.10.0", "numpy==1.23.5"])
    req = ROOT / "requirements-gpu.txt"
    log("installing {}".format(req))
    subprocess.check_call([str(python), "-m", "pip", "install", "-r", str(req)])
    subprocess.call([str(python), "-m", "pip", "uninstall", "-y", "opencv-python"])
    subprocess.check_call([str(python), "-m", "pip", "install", "numpy==1.23.5"])


def verify(python: Path) -> None:
    missing = [name for name in REQUIRED_DLLS if not (LIBBIN / name).exists()]
    if missing:
        raise SystemExit("missing CUDA DLLs: {}".format(", ".join(missing)))
    log("all required DLLs present in {}".format(LIBBIN))
    code = (
        "import tensorflow as tf;"
        "gpus=tf.config.list_physical_devices('GPU');"
        "print('tensorflow', tf.__version__);"
        "print('devices', tf.config.list_physical_devices());"
        "print('GPUS', gpus);"
        "print('built_with_cuda', tf.test.is_built_with_cuda());"
        "raise SystemExit(0 if gpus else 2)"
    )
    env = os.environ.copy()
    env["PATH"] = str(LIBBIN) + os.pathsep + env.get("PATH", "")
    env["CUDA_PATH"] = str(VENV / "Library")
    rc = subprocess.call([str(python), "-c", code], env=env)
    if rc != 0:
        raise SystemExit("TensorFlow did not see a GPU (exit {})".format(rc))
    log("GPU visible to TensorFlow")


def find_python39() -> Path:
    candidates = [
        ROOT / ".conda" / "python.exe",
        Path(r"C:\Users\kestl\.conda\python.exe"),
    ]
    for c in candidates:
        if c.exists():
            out = subprocess.check_output([str(c), "-c", "import sys; print(sys.version_info[:2])"], text=True)
            if "(3, 9)" in out:
                return c
    raise SystemExit(
        "Need Python 3.9 to create this venv (TensorFlow 2.10 GPU). "
        "Expected at E:\\Truthsayer\\.conda\\python.exe"
    )


def main() -> int:
    py39 = find_python39()
    python = create_venv(py39)
    cuda_pkg = download(CUDA_URL, CACHE / Path(CUDA_URL).name)
    cudnn_pkg = download(CUDNN_URL, CACHE / Path(CUDNN_URL).name)
    nvcc_pkg = download(NVCC_URL, CACHE / Path(NVCC_URL).name)
    extract_library(cuda_pkg, VENV)
    extract_library(cudnn_pkg, VENV)
    extract_library(nvcc_pkg, VENV)
    write_sitecustomize()
    patch_activators()
    pip_install(python)
    verify(python)
    log("done. Activate with:  .\\.venv\\Scripts\\Activate.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
