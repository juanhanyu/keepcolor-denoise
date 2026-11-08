"""First-time setup for Denoise (Denoise.exe runs it through launcher/setup.ps1 and uv; standard library only).

Picks the PyTorch build for this computer's graphics card, installs everything into runtime/venv, downloads the default
models into models/, then starts Denoise. Safe to run again: finished steps are skipped.

    python setup_env.py                 # detect the graphics card
    python setup_env.py --kind cpu      # or force one: auto / cuda / rocm / cpu
    python setup_env.py --no-start      # do not open Denoise at the end

- NVIDIA: the CUDA build of PyTorch, the newest one the installed driver can run (only the driver is needed, the wheels
  bring their own CUDA runtime).
- AMD (Windows): AMD's ROCm 7.2.1 for Windows PyTorch, for the cards it supports (RX 7000 / 9000, Radeon PRO W7000 /
  W9000 / R9700, Ryzen AI Max). Other AMD cards use the CPU build.
- Anything else: the CPU build.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RT = ROOT / "runtime"
VENV = RT / "venv"
WIN = os.name == "nt"
PY = VENV / ("Scripts/python.exe" if WIN else "bin/python")
PYW = VENV / ("Scripts/pythonw.exe" if WIN else "bin/python")
READY = RT / "ready.txt"
UV = os.environ.get("DENOISE_UV") or shutil.which("uv")
NO_WINDOW = subprocess.CREATE_NO_WINDOW if WIN else 0

TORCH = "https://download.pytorch.org/whl"
CUDA_BUILDS = [130, 129, 128, 126, 124, 121, 118]  # cu130 ... cu118, newest first
CN_PYPI = "https://pypi.tuna.tsinghua.edu.cn/simple"  # mirror tried when PyPI itself is unreachable
CN_TORCH = "https://mirrors.aliyun.com/pytorch-wheels"
ROCM = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
ROCM_SDK = ["rocm_sdk_core-7.2.1-py3-none-win_amd64.whl", "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
            "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl", "rocm-7.2.1.tar.gz"]
ROCM_TORCH = ["torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl", "torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"]
ROCM_CARDS = re.compile(r"RX\s*(7[6-9]|9[0-9])\d{2}|Radeon\s*(AI\s*)?PRO\s*(W7\d{3}|W9\d{3}|R9\d{3})|Radeon\s*80[56]0S", re.I)
NAMES = {"cuda": "NVIDIA CUDA", "rocm": "AMD ROCm 7.2.1", "cpu": "CPU"}
SIZES = {"cuda": "约 2.5–3 GB", "rocm": "约 2.3 GB", "cpu": "约 300 MB"}


def say(text="", color=""):
    codes = {"ok": "32", "warn": "33", "err": "31", "head": "36"}
    print(f"\x1b[{codes[color]}m{text}\x1b[0m" if color else text, flush=True)


def capture(cmd, timeout=60):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace",
                              creationflags=NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def run(cmd):
    say("  > " + " ".join(str(c) for c in cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    return subprocess.run([str(c) for c in cmd]).returncode == 0


def uv_install(*args, mirror=True):
    """uv pip install into the venv; on failure once more with the PyPI mirror (dependencies come from PyPI)."""
    base = [UV, "pip", "install", "--python", PY]
    if run(base + list(args)):
        return True
    if mirror and "--index-url" not in args:
        say("  官方源失败，换国内镜像再试 ...", "warn")
        return run(base + list(args) + ["--index-url", CN_PYPI])
    return False


def graphics():
    if WIN:
        names = capture(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"])
    elif shutil.which("lspci"):
        names = "\n".join(x.split(": ", 1)[-1] for x in capture(["lspci"]).splitlines() if "VGA" in x or "3D" in x)
    else:
        names = ""
    return [n.strip() for n in names.splitlines() if n.strip()]


def detect():
    names = graphics()
    smi = capture(["nvidia-smi"]) if shutil.which("nvidia-smi") else ""
    nvidia = next((n for n in names if "nvidia" in n.lower()), None)
    if nvidia or smi:
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi)
        return {"kind": "cuda", "gpu": nvidia or "NVIDIA", "names": names, "driver_cuda": (int(m[1]), int(m[2])) if m else None}
    amd = next((n for n in names if ROCM_CARDS.search(n)), None)
    if amd and WIN:
        return {"kind": "rocm", "gpu": amd, "names": names}
    return {"kind": "cpu", "gpu": None, "names": names}


def install_torch(kind, info):
    if kind == "rocm":
        say("安装 AMD ROCm 运行库和 ROCm 版 PyTorch（约 2.2 GB，来自 repo.radeon.com）...", "head")
        return uv_install(*[f"{ROCM}/{f}" for f in ROCM_SDK]) and uv_install(*[f"{ROCM}/{f}" for f in ROCM_TORCH])
    if kind == "cuda":
        builds = CUDA_BUILDS
        if info.get("driver_cuda"):
            major, minor = info["driver_cuda"]
            fits = [b for b in builds if b <= major * 10 + minor]
            builds = fits or builds[::-1]  # very old driver: try the oldest builds first
        else:
            say("没找到 nvidia-smi：显卡驱动可能没装。照样装 CUDA 版，装好驱动后就能用显卡。", "warn")
        for b in builds:
            say(f"安装 PyTorch（CUDA {b // 10}.{b % 10}）...", "head")
            if run([UV, "pip", "install", "--python", PY, "torch", "torchvision", "--index-url", f"{TORCH}/cu{b}"]):
                return True
            say("  换阿里云镜像再试 ...", "warn")
            if run([UV, "pip", "install", "--python", PY, "torch", "torchvision", "--find-links", f"{CN_TORCH}/cu{b}/",
                    "--index-url", CN_PYPI]):
                return True
        return False
    say("安装 CPU 版 PyTorch ...", "head")
    # PyPI's own Windows wheels are the CPU build, so the mirror is a full fallback here
    return run([UV, "pip", "install", "--python", PY, "torch", "torchvision", "--index-url", f"{TORCH}/cpu"]) or \
        run([UV, "pip", "install", "--python", PY, "torch", "torchvision", "--index-url", CN_PYPI])


CHECK = ("import json, torch, spandrel, numpy, PIL\n"
         "try:\n import tkinter; tk = True\nexcept Exception:\n tk = False\n"
         "ok = torch.cuda.is_available()\n"
         "print('CHECK' + json.dumps({'torch': torch.__version__, 'hip': getattr(torch.version, 'hip', None), "
         "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if ok else None, 'tk': tk}))")


def check():
    line = next((x for x in capture([str(PY), "-c", CHECK], timeout=300).splitlines() if x.startswith("CHECK")), None)
    return json.loads(line[5:]) if line else None


def start():
    if WIN:
        subprocess.Popen([str(PYW), str(ROOT / "server.py")], cwd=ROOT, close_fds=True,
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        subprocess.Popen([str(PY), str(ROOT / "server.py")], cwd=ROOT, start_new_session=True)


def main():
    ap = argparse.ArgumentParser(description="Install Denoise's dependencies into runtime/venv.")
    ap.add_argument("--kind", choices=["auto", "cuda", "rocm", "cpu"], default=os.environ.get("DENOISE_KIND", "auto"))
    ap.add_argument("--no-start", action="store_true", default=os.environ.get("DENOISE_NO_START") == "1")
    a = ap.parse_args()
    if WIN:
        os.system("")  # turns on colour codes in the old Windows console
    if not UV:
        raise SystemExit("需要 uv：请用 Denoise.exe 启动（它会自动下载），或先 pip install uv。")

    info = detect()
    kind = info["kind"] if a.kind == "auto" else a.kind
    say("Denoise 首次安装", "head")
    say("显卡：" + ("、".join(info["names"]) or "没有识别到"))
    say(f"方案：{NAMES[kind]}" + ("（手动指定）" if a.kind != "auto" else "") + f"，下载{SIZES[kind]}")
    say(f"位置：{RT}（只装在这里，不改系统；删掉文件夹就是卸载）")
    say()

    t0 = time.time()
    if READY.exists() and PY.exists() and check():
        say("运行环境已经装好，跳过安装。", "ok")
    else:
        READY.unlink(missing_ok=True)
        if not PY.exists():
            say("创建 Python 环境 ...", "head")
            shutil.rmtree(VENV, ignore_errors=True)
            if not run([UV, "venv", VENV, "--python", sys.executable]):
                raise SystemExit("创建 Python 环境失败，见上面的信息。")
        if not install_torch(kind, info):
            raise SystemExit("PyTorch 安装失败，见上面的信息。网络不稳时重新打开 Denoise.exe 会接着装。")
        say("安装其余依赖（spandrel、numpy、pillow）...", "head")
        if not uv_install("spandrel>=0.4", "numpy>=1.24", "pillow>=10.0"):
            raise SystemExit("依赖安装失败，见上面的信息。")
        r = check()
        if not r:
            raise SystemExit("装完了，但依赖导入失败，见上面的信息。")
        backend = f"ROCm {r['hip']}" if r["hip"] else f"CUDA {r['cuda']}" if r["cuda"] else "CPU"
        say(f"PyTorch {r['torch']} · {backend} · " + (f"显卡 {r['gpu']}" if r["gpu"] else "使用 CPU"), "ok")
        if kind in ("cuda", "rocm") and not r["gpu"]:
            say("环境能用，但现在只能跑 CPU：请更新显卡驱动后重启电脑。", "warn")
        if not r["tk"]:
            say("这个 Python 没有 tkinter：「更改输出文件夹」用不了，可以直接改 config.json。", "warn")
        READY.write_text(json.dumps({"kind": kind, **r, "installed": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False), "utf-8")
        shutil.rmtree(RT / "uv-cache", ignore_errors=True)  # the installed packages do not need the download cache

    say()
    say("下载默认模型（DRUNet 131 MB、APISR 2× 18 MB）...", "head")
    if subprocess.run([str(PY), str(ROOT / "denoise.py"), "--prefetch"]).returncode != 0:
        say("模型这次没下完：第一次处理图片时会再下载。", "warn")
    say(f"\n完成，用时 {(time.time() - t0) / 60:.1f} 分钟。", "ok")
    if not a.no_start:
        start()
        say("Denoise 正在打开 ...", "ok")


if __name__ == "__main__":
    main()
