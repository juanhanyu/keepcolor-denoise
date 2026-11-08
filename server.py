"""Denoise local web app. Denoise.exe starts it; by hand:

    runtime\\venv\\Scripts\\python server.py          # opens http://127.0.0.1:7870/ in the browser

Everything stays on this computer: the page talks only to this server, which listens on 127.0.0.1 only.
Starting it again while it runs just reopens the page; an older copy left running after an update is replaced.
"""
import atexit
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PORT = int(os.environ.get("DENOISE_PORT", "7870"))  # the env var is for testing a second copy
API = 1  # bumped whenever the page needs something new from this server; the page checks it
ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
RUNTIME = ROOT / "runtime"
STATE = RUNTIME / "server.json"  # pid and quit key of the running copy, so a newer copy can shut it down
CONFIG = ROOT / "config.json"
LOG = RUNTIME / "server.log"
VERSION = hashlib.sha1(b"".join((ROOT / n).read_bytes() for n in ("server.py", "denoise.py"))).hexdigest()[:12]
LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never send 127.0.0.1 through a proxy


def log(text):
    try:
        RUNTIME.mkdir(exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text.rstrip() + "\n")
    except OSError:
        pass


def running():
    try:
        return json.loads(LOCAL.open(f"http://127.0.0.1:{PORT}/api/info", timeout=1.5).read())
    except Exception:
        return None


def replace_old_copy(info):
    """Shut down an older copy (from before an update). A picture it is working on gets up to 20 s to finish."""
    for _ in range(40):
        if not (info and info.get("busy")):
            break
        time.sleep(0.5)
        info = running()
    if not info:
        return
    try:
        key = json.loads(STATE.read_text("utf-8")).get("key", "")
        LOCAL.open(urllib.request.Request(f"http://127.0.0.1:{PORT}/api/quit", data=b"", method="POST",
                                          headers={"X-Token": key}), timeout=3)
    except Exception:
        try:
            os.kill(int(info["pid"]), signal.SIGTERM)
        except Exception:
            pass
    for _ in range(20):
        if not running():
            return
        time.sleep(0.25)


if __name__ == "__main__":  # before the slow torch import
    info = running()
    if info and info.get("app") == "denoise":
        if info.get("version") == VERSION:
            if "--no-browser" not in sys.argv:
                webbrowser.open(f"http://127.0.0.1:{PORT}/")
            sys.exit(0)
        replace_old_copy(info)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import denoise as kc  # noqa: E402

DEVICE = kc.default_device()
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
TOKEN = secrets.token_urlsafe(24)
MAX_BYTES, MAX_PIXELS = 100 * 1024 * 1024, 4096 * 4096
Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 2

# Pictures (sources and results) live in a temp folder and the page loads them by URL, so a long session with many
# images never holds them all in the browser's memory. One folder per running copy, removed on exit; a folder left by
# a copy that was killed is cleared by a later start once it is a day old.
TMP = Path(tempfile.gettempdir()) / f"denoise-{os.getpid()}"
for old in Path(tempfile.gettempdir()).glob("denoise-*"):
    try:
        if old != TMP and time.time() - old.stat().st_mtime > 86400:
            shutil.rmtree(old, ignore_errors=True)
    except OSError:
        pass
TMP.mkdir(parents=True, exist_ok=True)
atexit.register(shutil.rmtree, TMP, True)

lock = threading.Lock()  # one picture at a time on the GPU
cache = kc.StageCache(16)
images = {}  # id -> {"name", "w", "h", "alpha" (array or None), "icc"}
decoded = {}  # id -> rgb array, a few recent ones
progress = {"busy": False, "stage": "", "job": "", "started": 0.0}
cancelled = set()


def stage(text):
    progress["stage"] = text


def config():
    try:
        return json.loads(CONFIG.read_text("utf-8"))
    except Exception:
        return {}


def output_dir():
    custom = config().get("output")
    return Path(custom) if custom else kc.OUTPUT_DIR


def set_output(path):
    c = config()
    if path:
        c["output"] = str(path)
    else:
        c.pop("output", None)
    CONFIG.write_text(json.dumps(c, ensure_ascii=False, indent=1), "utf-8")


def source_rgb(iid):
    if iid not in decoded:
        decoded[iid] = np.asarray(Image.open(TMP / f"{iid}.png").convert("RGB")).copy()
        while len(decoded) > 6:
            decoded.pop(next(iter(decoded)))
    return decoded[iid]


def add_image(data, name):
    try:
        with Image.open(io.BytesIO(data)) as im:
            if getattr(im, "is_animated", False):
                raise ValueError("请用静态图片；动图请先导出要处理的那一帧")
            if im.width * im.height > MAX_PIXELS:
                raise ValueError(f"图片太大（{im.width}×{im.height}），最多 4096×4096")
            if min(im.size) < 16:
                raise ValueError("图片至少要 16×16")
        rgb, alpha, icc = kc.read_image(io.BytesIO(data))
    except (Image.UnidentifiedImageError, OSError):
        raise ValueError("读不出这张图（支持 PNG、JPEG、WebP、BMP、GIF、TIFF、AVIF）") from None
    except Image.DecompressionBombError:
        raise ValueError("图片太大，最多 4096×4096") from None
    iid = uuid.uuid4().hex
    (TMP / f"{iid}.png").write_bytes(kc.png_bytes(rgb, alpha, icc))  # what the page shows as the original
    images[iid] = {"name": name, "w": rgb.shape[1], "h": rgb.shape[0], "alpha": alpha, "icc": icc}
    decoded[iid] = rgb
    return iid


def settings_from(q):
    g = lambda k, d: q.get(k, [d])[0]  # noqa: E731
    return kc.Settings(apisr=g("apisr", "0") == "1", apisr_model=g("apisr_model", "2x_rrdb"),
                       apisr_amount=float(g("apisr_amount", "100")) / 100, drunet=g("drunet", "1") == "1",
                       sigma=float(g("sigma", "10")), l0=g("l0", "0") == "1", lam=float(g("lam", "0.001"))).clean()


def run_job(iid, st, job):
    meta = images[iid]
    stop = (lambda: job in cancelled) if job else None
    with lock:
        if stop and stop():
            raise kc.Cancelled()
        progress.update(busy=True, stage="准备", job=job, started=time.time())
        try:
            t0 = time.time()
            res = kc.process(source_rgb(iid), st, DEVICE, cache, stop, stage)
            rid = uuid.uuid4().hex
            (TMP / f"{rid}.png").write_bytes(kc.png_bytes(res, meta["alpha"], meta["icc"]))
            return rid, time.time() - t0
        finally:
            progress.update(busy=False, stage="", job="")
            cancelled.discard(job)


def safe_stem(name):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(name).stem).strip(" .")[:100] or "image"


def save_result(iid, rid, mask_png, name):
    """Write the result to the output folder; a protection mask (grey = share of the original) is blended in first."""
    meta = images[iid]
    res = np.asarray(Image.open(TMP / f"{rid}.png").convert("RGB"))
    if mask_png:
        m = np.asarray(Image.open(io.BytesIO(mask_png)).convert("L"), np.float32)[..., None] / 255
        if m.shape[:2] != res.shape[:2]:
            raise ValueError("保护区大小和图片不一致")
        res = np.rint(res * (1 - m) + source_rgb(iid) * m).astype(np.uint8)
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{safe_stem(name or meta['name'])}_clean.png"
    part = target.with_name(target.name + ".part")
    part.write_bytes(kc.png_bytes(res, meta["alpha"], meta["icc"], level=4))
    part.replace(target)
    return target


def reveal(path):
    if os.name == "nt":
        subprocess.Popen(["explorer", f"/select,{path}"] if path.is_file() else ["explorer", str(path)])
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path if path.is_dir() else path.parent)])


def choose_folder():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askdirectory(parent=root, title="选择输出文件夹", initialdir=str(output_dir()), mustexist=False)
    finally:
        root.destroy()


def model_info():
    return {k: {"label": v[1], "mb": round((kc.MODEL_DIR / kc.WEIGHTS[v[0]][0]).stat().st_size / 1e6)
                if (kc.MODEL_DIR / kc.WEIGHTS[v[0]][0]).exists() else None} for k, v in kc.APISR.items()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # no console under pythonw
        pass

    def send(self, code, body, ctype, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", **(headers or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def json(self, obj, code=200):
        self.send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            page = (WEB / "index.html").read_bytes().replace(b"__TOKEN__", TOKEN.encode())
            return self.send(200, page, "text/html; charset=utf-8", {
                "Content-Security-Policy": "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
                                           "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'"})
        if path in ("/icon.svg", "/favicon.ico"):
            f = WEB / ("icon.svg" if path == "/icon.svg" else "icon.ico")
            return self.send(200, f.read_bytes(), "image/svg+xml" if f.suffix == ".svg" else "image/x-icon")
        if path == "/api/info":
            return self.json({"app": "denoise", "api": API, "version": VERSION, "pid": os.getpid(), "device": DEVICE_NAME,
                              "busy": progress["busy"], "output": str(output_dir()), "default_output": str(kc.OUTPUT_DIR),
                              "custom_output": bool(config().get("output")), "apisr": model_info()})
        if path == "/api/progress":
            p = dict(progress)
            p["elapsed"] = round(time.time() - p["started"], 1) if p["busy"] else 0
            return self.json(p)
        if path == "/api/out":  # a file saved to the output folder, for the page's download button
            f = output_dir() / Path(parse_qs(urlparse(self.path).query).get("file", [""])[0]).name
            if f.suffix.lower() == ".png" and f.is_file():
                return self.send(200, f.read_bytes(), "image/png")
            return self.send(404, b"not found", "text/plain")
        m = re.fullmatch(r"/api/file/([0-9a-f]{32})\.png", path)
        if m and (TMP / f"{m[1]}.png").is_file():  # never changes once written: the browser may cache it freely
            return self.send(200, (TMP / f"{m[1]}.png").read_bytes(), "image/png", {"Cache-Control": "private, max-age=86400, immutable"})
        self.send(404, b"not found", "text/plain")

    def allowed(self):
        origin = self.headers.get("Origin")
        ok_origin = origin is None or re.fullmatch(r"http://(127\.0\.0\.1|localhost):%d" % self.server.server_port, origin)
        return bool(ok_origin) and secrets.compare_digest(self.headers.get("X-Token", ""), TOKEN)

    def body(self, required=True):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BYTES or (required and n <= 0):
            raise ValueError("文件为空或超过 100 MB")
        return self.rfile.read(n) if n else b""

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self.allowed():
            return self.json({"error": "页面已过期，请刷新"}, 403)
        arg = lambda k, d="": q.get(k, [d])[0]  # noqa: E731
        try:
            if u.path == "/api/add":
                iid = add_image(self.body(), arg("name", "image.png"))
                meta = images[iid]
                return self.json({"id": iid, "w": meta["w"], "h": meta["h"], "alpha": meta["alpha"] is not None})
            if u.path == "/api/process":
                iid = arg("id")
                if iid not in images:
                    raise ValueError("这张图已经不在了，请重新添加")
                try:
                    rid, sec = run_job(iid, settings_from(q), arg("job")[:64])
                except kc.Cancelled:
                    return self.json({"cancelled": True}, 409)
                return self.json({"rid": rid, "elapsed": round(sec, 1)})
            if u.path == "/api/cancel":
                if arg("job"):
                    cancelled.add(arg("job")[:64])
                return self.json({"ok": True})
            if u.path == "/api/save":
                if arg("id") not in images or not (TMP / f"{arg('rid')}.png").is_file():
                    raise ValueError("结果已经不在了")
                target = save_result(arg("id"), arg("rid"), self.body(required=False), arg("name"))
                return self.json({"file": target.name, "path": str(target), "dir": str(target.parent)})
            if u.path == "/api/reveal":
                p = Path(arg("path")) if arg("path") else output_dir()
                if not arg("path"):
                    p.mkdir(parents=True, exist_ok=True)
                reveal(p)
                return self.json({"ok": True})
            if u.path == "/api/output/choose":
                chosen = choose_folder()
                if chosen:
                    set_output(Path(chosen))
                return self.json({"output": str(output_dir()), "custom": bool(config().get("output")), "changed": bool(chosen)})
            if u.path == "/api/output/reset":
                set_output(None)
                return self.json({"output": str(output_dir()), "custom": False})
            if u.path == "/api/quit":
                self.json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            return self.json({"error": "没有这个操作"}, 404)
        except ValueError as e:
            return self.json({"error": str(e)}, 400)
        except Exception as e:
            log(traceback.format_exc())
            return self.json({"error": f"{type(e).__name__}: {e}"}, 500)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # on Windows reuse would let two copies share the port

    def server_bind(self):
        import socket
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main():
    try:
        srv = Server(("127.0.0.1", PORT), Handler)
    except OSError:  # started at the same moment as another copy, or the port is taken by something else
        if running():
            webbrowser.open(f"http://127.0.0.1:{PORT}/")
            return
        srv = Server(("127.0.0.1", 0), Handler)
    RUNTIME.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({"pid": os.getpid(), "port": srv.server_port, "version": VERSION, "key": TOKEN}), "utf-8")
    url = f"http://127.0.0.1:{srv.server_port}/"
    log(f"start {VERSION} on {url} ({DEVICE_NAME})")
    print(f"Denoise running at {url} ({DEVICE_NAME}); results go to {output_dir()}", flush=True)
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    finally:
        try:
            if json.loads(STATE.read_text("utf-8")).get("pid") == os.getpid():
                STATE.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
