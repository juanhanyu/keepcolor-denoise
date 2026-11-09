"""Color-preserving DRUNet denoising and optional L0 smoothing."""
import argparse
import hashlib
import io
import sys
import time
import urllib.request
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageCms, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".tga", ".avif"}

_REL = "https://github.com/{}/releases/download/{}/{}"
# key: (file, sha256, sources in the order tried: the original release first, then byte-identical mirrors)
WEIGHTS = {
    "drunet": ("drunet_color.pth", "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4", [
        _REL.format("cszn/KAIR", "v1.0", "drunet_color.pth"),
        "https://huggingface.co/deepinv/drunet/resolve/main/drunet_color.pth",
        "https://hf-mirror.com/deepinv/drunet/resolve/main/drunet_color.pth"]),
}


@dataclass(frozen=True)
class Settings:
    drunet: bool = True          # pixel-shuffle DRUNet denoising
    sigma: float = 10.0          # DRUNet noise level (8-bit units): higher removes more
    l0: bool = False             # L0 smoothing at the end
    lam: float = 0.001           # L0 strength

    def clean(self):
        """The same settings with every value clamped into its valid range."""
        return replace(self, sigma=min(max(float(self.sigma), 1.0), 50.0),
                       lam=min(max(float(self.lam), 1e-5), 0.05))


class Cancelled(Exception):
    """Raised inside a long step when the stop() callback says the result is no longer wanted."""


def _poll(stop):
    if stop is not None and stop():
        raise Cancelled()


# ------------------------------------------------------------------------------------------------ images

def read_image(src):
    """Decode an image the way a viewer shows it: (rgb uint8 HxWx3, alpha uint8 HxW or None, icc bytes or None).

    EXIF rotation is applied, 16/32-bit greyscale is scaled down to 8 bits, CMYK goes through its own profile when it
    has one, and the alpha channel is returned only when some pixel is actually transparent.
    """
    im = Image.open(src)
    im.load()
    im = ImageOps.exif_transpose(im)
    icc = im.info.get("icc_profile")
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        a = np.asarray(im, dtype=np.float64)
        im = Image.fromarray(np.clip(np.rint(a / (257.0 if a.max() > 255 else 1.0)), 0, 255).astype(np.uint8), "L")
    if im.mode == "CMYK" and icc:
        try:
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile("sRGB"),
                                           outputMode="RGB")
        except Exception:
            pass
    if im.mode not in ("RGB", "RGBA", "RGBX", "P"):
        icc = None  # a grey or CMYK profile does not describe the RGB pixels produced here
    alpha = None
    if im.mode in ("RGBA", "LA", "PA", "La", "RGBa") or "transparency" in im.info:
        a = np.asarray(im.convert("RGBA"))[..., 3]
        if a.min() < 255:
            alpha = a.copy()
    return np.asarray(im.convert("RGB")).copy(), alpha, icc


def png_bytes(rgb, alpha=None, icc=None, level=1):
    im = Image.fromarray(rgb)
    if alpha is not None:
        im.putalpha(Image.fromarray(alpha))
    buf = io.BytesIO()
    im.save(buf, "PNG", compress_level=level, **({"icc_profile": icc} if icc else {}))
    return buf.getvalue()


def to_tensor(rgb, device):
    return torch.from_numpy(rgb).to(device).permute(2, 0, 1)[None].float().div_(255)


def to_rgb(t):
    return t[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


# ------------------------------------------------------------------------------------------------ models

def fetch(key, report=None):
    """Path of a weights file, downloaded (and checked against its SHA-256) the first time it is needed."""
    name, sha, urls = WEIGHTS[key]
    path = MODEL_DIR / name
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            return path
        raise ValueError(f"{path}: cached weights checksum mismatch")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    problem = None
    for url in urls:
        try:
            data = _download(url, name, report)
        except OSError as e:  # GitHub release downloads time out now and then: try the next source
            problem = e
            continue
        if hashlib.sha256(data).hexdigest() != sha:
            problem = ValueError(f"{name}: checksum mismatch from {url}")
            continue
        part = path.with_suffix(".part")
        part.write_bytes(data)
        part.replace(path)
        return path
    raise RuntimeError(f"could not download {name} ({problem})")


def _download(url, name, report):
    req = urllib.request.Request(url, headers={"User-Agent": "keepcolor-denoise"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total, chunks, got, shown = int(r.headers.get("Content-Length") or 0), [], 0, 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
            if report and time.time() - shown > 0.5:
                shown = time.time()
                report(f"下载模型 {name} · {got / 1e6:.0f}" + (f" / {total / 1e6:.0f} MB" if total else " MB"))
    return b"".join(chunks)


_nets = {}


def network(key, device, report=None):
    """A loaded spandrel model descriptor (spandrel's loader refuses pickled code, so weights cannot run anything)."""
    if (key, str(device)) not in _nets:
        import spandrel
        path = fetch(key, report)
        if report:
            report(f"加载模型 {path.name}")
        _nets[key, str(device)] = spandrel.ModelLoader().load_from_file(str(path)).to(device).eval()
    return _nets[key, str(device)]


# ------------------------------------------------------------------------------------------------ steps

def _feather(n, overlap, lead, tail, device):
    w = torch.ones(n, device=device)
    k = min(overlap, n)
    if lead:
        w[:k] = torch.linspace(0, 1, k + 2, device=device)[1:-1]
    if tail:
        w[-k:] = torch.minimum(w[-k:], torch.linspace(1, 0, k + 2, device=device)[1:-1])
    return w


@torch.no_grad()
def tiled(fn, x, tile, overlap=32, stop=None):
    """fn over overlapping tiles of x (1xCxHxW); fn returns each tile at its own size. Overlaps are cross-faded, and
    the tile size is fixed (not chosen from free memory), so the same input always gives the same output."""
    _, c, h, w = x.shape
    if h <= tile and w <= tile:
        _poll(stop)
        return fn(x)
    step = tile - overlap
    ys = sorted({*range(0, max(h - tile, 0) + 1, step), max(h - tile, 0)})
    xs = sorted({*range(0, max(w - tile, 0) + 1, step), max(w - tile, 0)})
    out = torch.zeros_like(x)
    weight = torch.zeros(1, 1, h, w, device=x.device)
    for y in ys:
        for x0 in xs:
            _poll(stop)
            y1, x1 = min(y + tile, h), min(x0 + tile, w)
            m = _feather(y1 - y, overlap, y > 0, y1 < h, x.device)[:, None] * _feather(x1 - x0, overlap, x0 > 0, x1 < w, x.device)
            out[..., y:y1, x0:x1] += fn(x[..., y:y1, x0:x1]) * m
            weight[..., y:y1, x0:x1] += m
    return out / weight


def _drunet(net, x, sigma):
    h, w = x.shape[-2:]
    xp = F.pad(x, (0, -w % 8, 0, -h % 8), mode="replicate")
    level = torch.full_like(xp[:, :1], sigma / 255.0)
    return net(torch.cat([xp, level], 1))[..., :h, :w]


@torch.no_grad()
def drunet_pd(x, sigma, device, stop=None, report=None, refine=4.0):
    """Pixel-shuffle down-sampled DRUNet denoising, then a light full-size pass to even out the four sub-images."""
    net = network("drunet", device, report).model  # the raw network: the noise level is our own input
    if report:
        report("DRUNet 去噪")
    _, c, h, w = x.shape
    xp = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
    hh, ww = xp.shape[-2] // 2, xp.shape[-1] // 2
    subs = xp.view(1, c, hh, 2, ww, 2).permute(3, 5, 0, 1, 2, 4).reshape(4, c, hh, ww)  # sub (i, j) = pixels [i::2, j::2]
    den = torch.cat([tiled(lambda t: _drunet(net, t, sigma), subs[k:k + 1], 512, 32, stop) for k in range(4)])
    y = den.view(2, 2, 1, c, hh, ww).permute(2, 3, 4, 0, 5, 1).reshape(1, c, hh * 2, ww * 2)[..., :h, :w]
    return tiled(lambda t: _drunet(net, t, refine), y, 512, 32, stop).clamp_(0, 1)


@torch.no_grad()
def l0_smooth(x, lam, stop=None, kappa=2.0, beta_max=1e5):
    """L0 gradient minimisation (Xu, Lu, Xu & Jia, SIGGRAPH Asia 2011) of a 1x3xHxW image, in float64 on its device.

    Alternates between keeping only gradients strong enough to be worth their L0 cost and solving the quadratic
    problem for the image in the Fourier domain, where the difference operators are diagonal.
    """
    s = x[0].double()
    h, w = s.shape[-2:]
    fy = torch.fft.fftfreq(h, device=s.device, dtype=s.dtype)[:, None]
    fx = torch.fft.fftfreq(w, device=s.device, dtype=s.dtype)[None, :]
    lap = (2 - 2 * torch.cos(2 * torch.pi * fx)) + (2 - 2 * torch.cos(2 * torch.pi * fy))  # eigenvalues of DᵀD
    f_in = torch.fft.fft2(s)
    beta = 2 * lam
    while beta < beta_max:
        _poll(stop)
        gx = torch.roll(s, -1, -1) - s
        gy = torch.roll(s, -1, -2) - s
        flat = (gx * gx + gy * gy).sum(0, keepdim=True) <= lam / beta
        gx = gx.masked_fill(flat, 0)
        gy = gy.masked_fill(flat, 0)
        div = (torch.roll(gx, 1, -1) - gx) + (torch.roll(gy, 1, -2) - gy)  # Dᵀg
        s = torch.fft.ifft2((f_in + beta * torch.fft.fft2(div)) / (1 + beta * lap)).real
        beta *= kappa
    return s.clamp(0, 1).float()[None]


# ------------------------------------------------------------------------------------------------ pipeline

class StageCache:
    """The most recent intermediate results (kept on the CPU), so changing only a later step reruns only that step."""

    def __init__(self, size=16):
        self.size, self.items = size, OrderedDict()

    def get(self, key, make, device):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key].to(device)
        value = make()
        self.items[key] = value.detach().to("cpu")
        while len(self.items) > self.size:
            self.items.popitem(last=False)
        return value


def process(rgb, st, device, cache=None, stop=None, report=None):
    """Run the enabled steps on an RGB uint8 image and return the RGB uint8 result."""
    st = st.clean()
    cache = cache or StageCache(0)
    tag = hashlib.sha1(rgb.tobytes() + repr(rgb.shape).encode()).hexdigest()
    x = to_tensor(rgb, device)
    key = (tag,)
    if st.drunet:
        x = cache.get(key + ("drunet", round(st.sigma, 2)), lambda: drunet_pd(x, st.sigma, device, stop, report), device)
    if st.l0:
        if report:
            report("L0 平滑")
        x = l0_smooth(x, st.lam, stop)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return to_rgb(x)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="images or folders")
    ap.add_argument("--out", default=str(OUTPUT_DIR), help="output folder (default: ./output)")
    ap.add_argument("--no-drunet", action="store_true", help="skip DRUNet denoising")
    ap.add_argument("--sigma", type=float, default=10.0, help="DRUNet strength (default 10)")
    ap.add_argument("--l0", action="store_true", help="L0 smoothing at the end")
    ap.add_argument("--lam", type=float, default=0.001, help="L0 strength (default 0.001)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: the GPU when there is one)")
    ap.add_argument("--prefetch", nargs="*", metavar="KEY", help="only download weights (default: drunet)")
    a = ap.parse_args()

    if a.prefetch is not None:
        for key in a.prefetch or ["drunet"]:
            path = fetch(key, lambda text: print("\r  " + text + "    ", end="", flush=True))
            print(f"\r  {WEIGHTS[key][0]}：已就绪 ({path.stat().st_size / 1e6:.0f} MB)          ", flush=True)
        return
    if not a.inputs:
        ap.error("give at least one image or folder")
    st = Settings(drunet=not a.no_drunet, sigma=a.sigma, l0=a.l0, lam=a.lam)
    device = a.device or default_device()
    files = []
    for p in map(Path, a.inputs):
        files += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} · settings: {asdict(st)}", flush=True)
    for p in files:
        t0 = time.time()
        rgb, alpha, icc = read_image(p)
        res = process(rgb, st, device)
        target = out / f"{p.stem}_clean.png"
        target.write_bytes(png_bytes(res, alpha, icc, level=6))
        print(f"{p} -> {target} ({time.time() - t0:.1f} s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
