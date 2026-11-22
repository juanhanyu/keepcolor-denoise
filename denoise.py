"""Denoise: clean compression artifacts, ringing, moiré and fish-scale texture out of an image while keeping its
original colours.

Steps, each optional, in this order:
  0. Speck removal      small bright white specks (a few pixels across) are found by their shape and filled in from
                        the pixels around them; everything else is left untouched (off by default)
  1. APISR sharpening   an anime/illustration restoration network, run at full size and blended with the input
                        (changes colours a little, so it is off by default)
  2. DRUNet denoising   a Gaussian denoiser applied to the 2x2 pixel-shuffled sub-images of the picture: neighbouring
                        artifact pixels are correlated, pixels two apart hardly are, so in a sub-image the artifacts
                        look like plain noise and come out cleanly; nothing is downscaled or redrawn (on by default)
  3. L0 smoothing       flattens small ripples inside flat colour areas, keeps edges (off by default)
  4. Real-ESRGAN 4x     super-resolution with RealESRGAN_x4plus_anime_6B on the full-size picture (no downscaling
                        first); the result is kept at 4x, brought back to the original size, or resized to a given
                        size. It redraws the picture, so colours and fine detail change (off by default)

  python denoise.py photo.png                    # DRUNet only, result in ./output
  python denoise.py a.png b.jpg --apisr          # APISR 2x first, then DRUNet
  python denoise.py folder --l0 --out clean      # every image in a folder, DRUNet then L0
  python denoise.py photo.png --sr               # DRUNet, then 4x super-resolution (result is 4x the size)
  python denoise.py photo.png --sr orig          # the same, brought back to the original size
  python denoise.py photo.png --sr 4k            # fitted into 3840x2160 (or 1080p: 1920x1080), proportions kept
  python denoise.py photo.png --sr 3840x2160     # ... or to an exact size; 3840x / x2160 keep the proportions
"""
import argparse
import hashlib
import io
import math
import re
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
        "https://huggingface.co/deepinv/drunet/resolve/main/drunet_color.pth"]),
    "apisr_2x_rrdb": ("2x_APISR_RRDB_GAN_generator.pth", "3b0d2b3a3c0461ac17d00f4f32240666fb832b738ea5a48449b1acf07fbb07e5", [
        _REL.format("Kiteretsu77/APISR", "v0.1.0", "2x_APISR_RRDB_GAN_generator.pth"),
        "https://huggingface.co/camenduru/APISR/resolve/main/2x_APISR_RRDB_GAN_generator.pth"]),
    "apisr_4x_rrdb": ("4x_APISR_RRDB_GAN_generator.pth", "6bd14a66224c90d4754011f378ac828b18e221f2d031026ec99cb5facdf40c19", [
        _REL.format("Kiteretsu77/APISR", "v0.2.0", "4x_APISR_RRDB_GAN_generator.pth")]),
    "apisr_4x_grl": ("4x_APISR_GRL_GAN_generator.pth", "56fff250139563dea59c4ca81af19cc098d94dc3abaad23640f14cec488e5da1", [
        _REL.format("Kiteretsu77/APISR", "v0.1.0", "4x_APISR_GRL_GAN_generator.pth"),
        "https://huggingface.co/camenduru/APISR/resolve/main/4x_APISR_GRL_GAN_generator.pth"]),
    "apisr_4x_dat": ("4x_APISR_DAT_GAN_generator.pth", "cc625c4ec5242a57e46f0941a3c36e3a731dcd859750c5f72a5251045b1e6d72", [
        _REL.format("Kiteretsu77/APISR", "v0.3.0", "4x_APISR_DAT_GAN_generator.pth")]),
    "esrgan_4x_anime": ("RealESRGAN_x4plus_anime_6B.pth", "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da", [
        _REL.format("xinntao/Real-ESRGAN", "v0.2.2.4", "RealESRGAN_x4plus_anime_6B.pth")]),
}
# APISR level: (weights key, label, upscale factor, tile edge in input pixels)
APISR = {
    "2x_rrdb": ("apisr_2x_rrdb", "2× RRDB", 2, 384),
    "4x_rrdb": ("apisr_4x_rrdb", "4× RRDB", 4, 256),
    "4x_grl": ("apisr_4x_grl", "4× GRL", 4, 192),
    "4x_dat": ("apisr_4x_dat", "4× DAT", 4, 192),
}
# super-resolution: weights key, label, the network's factor, tile edge in input pixels
SR_KEY, SR_LABEL, SR_SCALE, SR_TILE = "esrgan_4x_anime", "Real-ESRGAN 4×", 4, 256
SR_MODES = ("4x", "orig", "4k", "1080p", "custom")
SR_BOXES = {"4k": (3840, 2160), "1080p": (1920, 1080)}  # the picture is fitted into the box (turned for portrait)
SR_MAX_PIXELS = 8192 * 8192  # largest result of the super-resolution step


@dataclass(frozen=True)
class Settings:
    speck: bool = False          # remove small bright white specks first
    speck_size: int = 2          # largest speck, in pixels across
    speck_thr: float = 0.10      # how much brighter than its surroundings a speck must be (0..1)
    speck_ratio: float = 1.0     # share of the specks that is removed, picked at random (1 = all of them)
    apisr: bool = False          # sharpen with APISR first (changes colours a little)
    apisr_model: str = "2x_rrdb"
    apisr_amount: float = 1.0    # 0..1: how much of the APISR result is blended over the input
    drunet: bool = True          # pixel-shuffle DRUNet denoising
    sigma: float = 10.0          # DRUNet noise level (8-bit units): higher removes more
    l0: bool = False             # L0 smoothing at the end
    lam: float = 0.001           # L0 strength
    sr: bool = False             # Real-ESRGAN 4x super-resolution as the last step (redraws the picture)
    sr_mode: str = "4x"          # size of the result: "4x", "orig" (the input size), "4k" / "1080p" (fitted into
                                 # 3840x2160 / 1920x1080 in the picture's own proportions) or "custom" (sr_w x sr_h)
    sr_w: int = 0                # custom size; 0 = follow the other side in proportion (both 0 = the input size)
    sr_h: int = 0

    def clean(self):
        """The same settings with every value clamped into its valid range."""
        return replace(self, apisr_model=self.apisr_model if self.apisr_model in APISR else "2x_rrdb",
                       apisr_amount=min(max(float(self.apisr_amount), 0.0), 1.0),
                       sigma=min(max(float(self.sigma), 1.0), 50.0), lam=min(max(float(self.lam), 1e-5), 0.05),
                       speck_size=min(max(int(self.speck_size), 1), 15),
                       speck_thr=min(max(float(self.speck_thr), 0.02), 0.5),
                       speck_ratio=min(max(float(self.speck_ratio), 0.0), 1.0),
                       sr_mode=self.sr_mode if self.sr_mode in SR_MODES else "4x",
                       sr_w=min(max(int(self.sr_w), 0), 65536), sr_h=min(max(int(self.sr_h), 0), 65536))


SR_RATIOS = ((1, 1), (5, 4), (4, 3), (3, 2), (16, 10), (16, 9), (2, 1), (21, 9))
SR_SNAP = 4  # pixels


def _snap(fixed, free):
    """The side worked out from the picture's proportions, moved onto a common aspect ratio when that is at most
    SR_SNAP pixels away (3838x2160 becomes 3840x2160); the other side, which was asked for, stays."""
    best = free
    for a, b in SR_RATIOS:
        for c in (int(fixed * a / b + 0.5), int(fixed * b / a + 0.5)):
            if abs(c - free) <= SR_SNAP and (best == free or abs(c - free) < abs(best - free)):
                best = c
    return best


def sr_size(w, h, st):
    """(width, height) of the result for a w x h input; ValueError when the super-resolution result would be too big."""
    if not st.sr or st.sr_mode == "orig":
        return w, h
    if st.sr_mode == "4x":
        tw, th = w * SR_SCALE, h * SR_SCALE
    elif st.sr_mode in SR_BOXES:
        long, short = SR_BOXES[st.sr_mode]
        f = min(long / max(w, h), short / min(w, h))
        tw, th = max(1, int(w * f + 0.5)), max(1, int(h * f + 0.5))
        wide_fixed = (long / max(w, h) <= short / min(w, h)) == (w >= h)  # is the width the side that hit the box?
        tw, th = (tw, _snap(tw, th)) if wide_fixed else (_snap(th, tw), th)
    else:
        tw, th = int(st.sr_w), int(st.sr_h)
        if tw <= 0 and th <= 0:
            return w, h
        if tw <= 0:
            tw = _snap(th, max(1, int(w * th / h + 0.5)))
        elif th <= 0:
            th = _snap(tw, max(1, int(h * tw / w + 0.5)))
    if tw * th > SR_MAX_PIXELS:
        raise ValueError(f"超分结果太大（{tw}×{th}），最多约 {SR_MAX_PIXELS // 1000000} 百万像素（如 8192×8192）；"
                         "请改成「原分辨率」或指定一个小一些的尺寸")
    return tw, th


def parse_sr(text):
    """--sr value -> (sr_mode, sr_w, sr_h): 4x, orig, or a size such as 3840x2160, 3840x, x2160."""
    t = text.strip().lower().replace("×", "x").replace("*", "x")
    if t in ("4x", "x4", "4"):
        return "4x", 0, 0
    if t in ("orig", "original", "1x", "keep"):
        return "orig", 0, 0
    if t in SR_BOXES:
        return t, 0, 0
    m = re.fullmatch(r"(\d*)x(\d*)", t)
    if not m or not (m[1] or m[2]):
        raise argparse.ArgumentTypeError(f"use 4x, orig, 4k, 1080p, or a size like 3840x2160 / 3840x / x2160, got {text!r}")
    return "custom", int(m[1] or 0), int(m[2] or 0)


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


def fit_alpha(alpha, w, h):
    """The alpha channel resized to a result of another size (None stays None)."""
    if alpha is None or alpha.shape[:2] == (h, w):
        return alpha
    return np.asarray(Image.fromarray(alpha).resize((w, h), Image.LANCZOS)).copy()


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
        return path
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
def tiled(fn, x, tile, overlap=32, stop=None, scale=1, store=None):
    """fn over overlapping tiles of x (1xCxHxW); fn returns each tile at `scale` times its own size. Overlaps are
    cross-faded, and the tile size is fixed (not chosen from free memory), so the same input always gives the same
    output. store: the device the result is put together on (default: where x is)."""
    _, c, h, w = x.shape
    store = store or x.device
    if h <= tile and w <= tile:
        _poll(stop)
        return fn(x).to(store)
    step, k = tile - overlap, scale
    ys = sorted({*range(0, max(h - tile, 0) + 1, step), max(h - tile, 0)})
    xs = sorted({*range(0, max(w - tile, 0) + 1, step), max(w - tile, 0)})
    out = torch.zeros(1, c, h * k, w * k, device=store)
    weight = torch.zeros(1, 1, h * k, w * k, device=store)
    for y in ys:
        for x0 in xs:
            _poll(stop)
            y1, x1 = min(y + tile, h), min(x0 + tile, w)
            m = (_feather((y1 - y) * k, overlap * k, y > 0, y1 < h, x.device)[:, None]
                 * _feather((x1 - x0) * k, overlap * k, x0 > 0, x1 < w, x.device))
            out[..., y * k:y1 * k, x0 * k:x1 * k] += (fn(x[..., y:y1, x0:x1]) * m).to(store)
            weight[..., y * k:y1 * k, x0 * k:x1 * k] += m.to(store)
    return out.div_(weight)


def _open_line(x, k, dy, dx):
    """Morphological opening of 1x1xHxW with a k-pixel line in direction (dy, dx): keeps what is k long that way.
    k may be even: the line then reaches one pixel further forward than back, and the dilation uses its mirror."""
    a, b = (k - 1) // 2, k // 2
    h, w = x.shape[-2:]

    def shifts(t, lo, hi):
        p = F.pad(t, (b, b, b, b), mode="replicate")
        return torch.stack([p[..., b + i * dy:b + i * dy + h, b + i * dx:b + i * dx + w] for i in range(lo, hi + 1)])
    return shifts(shifts(x, -a, b).min(0).values, -b, a).max(0).values


def _pick_specks(mask, share, steps):
    """A random `share` of the specks in a 0/1 mask (1x1xHxW), whole specks at a time.

    Every pixel gets a random number (fixed seed, so the same picture always loses the same specks), each speck takes
    the largest one inside it, and a hash of that decides: uniform whatever the speck's size, and a lower share always
    picks a subset of what a higher share picks.
    """
    g = torch.Generator().manual_seed(0)
    key = torch.randint(1, 1 << 24, mask.shape, generator=g, dtype=torch.int32).to(mask.device).float() * mask
    for _ in range(steps):
        key = F.max_pool2d(key, 3, 1, 1) * mask
    h = key.long() * 2654435761 % (1 << 32)  # an integer hash in three mixing rounds
    h = (h ^ (h >> 15)) * 2246822519 % (1 << 32)
    h = (h ^ (h >> 13)) * 3266489917 % (1 << 32)
    return mask * ((h ^ (h >> 16)) < int(share * (1 << 32)))


@torch.no_grad()
def despeckle(x, size=2, thr=0.10, ratio=1.0, stop=None, report=None):
    """Remove bright white specks up to `size` pixels across from a 1x3xHxW image.

    A speck is short in every direction, a thin highlight line or an edge is long in at least one. So the picture
    (its darkest channel: white specks are bright in all three) is opened with lines in four directions and the
    largest result is what the picture looks like without specks; pixels brighter than that by more than thr are the
    specks. They (and a one-pixel rim) are filled with a blur of the untouched pixels around them; no other pixel
    changes. ratio below 1 removes only that share of the specks, picked at random.
    """
    if report:
        report("去白噪点")
    k = size + 1  # a line one pixel longer than the largest speck
    low = x.min(1, keepdim=True).values
    opened = low
    for i, (dy, dx) in enumerate(((0, 1), (1, 0), (1, 1), (1, -1))):
        _poll(stop)
        o = _open_line(low, k, dy, dx)
        opened = o if i == 0 else torch.maximum(opened, o)
    mask = F.max_pool2d(((low - opened) > thr).float(), 3, 1, 1)
    if ratio < 1:
        mask = _pick_specks(mask, ratio, 2 * size + 4)
    if not mask.any():
        return x
    keep = 1 - mask
    fill, got, sigma = torch.zeros_like(x), torch.zeros_like(mask), 1.0
    for _ in range(4):  # normalised blur of the kept pixels; wider only where a speck had too few neighbours
        _poll(stop)
        r = int(3 * sigma) | 1
        g = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma * sigma))
        g = g / g.sum()

        def blur(t):
            c = t.shape[1]
            t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
            return F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
        den = blur(keep)
        ok = (den > 0.2).float() * (1 - got)
        fill += ok * blur(x * keep) / den.clamp_min(1e-6)
        got += ok
        sigma *= 2
    fill += (1 - got) * x
    return (x * keep + fill * mask).clamp_(0, 1)


def apisr(x, level, device, stop=None, report=None):
    """APISR at full size: each tile is upscaled by the network and brought straight back to its own size."""
    key, _, scale, tile = APISR[level]
    net = network(key, device, report)
    if report:
        report(f"APISR {APISR[level][1]} 锐化")

    def run(t):
        y = net(t)
        return F.interpolate(y, size=t.shape[-2:], mode="bicubic", antialias=True, align_corners=False).clamp_(0, 1)
    return tiled(run, x, tile, 24, stop)


@torch.no_grad()
def superres(x, size, device, stop=None, report=None):
    """Real-ESRGAN x4plus anime 6B on the full-size image (nothing is downscaled first), delivered at size = (w, h).

    Each tile is upscaled 4x and, when the result should be smaller than 4x, brought down at once to the smallest
    whole multiple of the input that still covers the wanted size (1x for the original size), so memory follows the
    size of the result and not the 4x picture. A size that is not a whole multiple gets one final resize.
    """
    net = network(SR_KEY, device, report)
    if report:
        report(f"{SR_LABEL} 超分")
    h, w = x.shape[-2:]
    tw, th = size
    k = min(SR_SCALE, max(1, math.ceil(max(tw / w, th / h) - 1e-9)))
    while k > 1 and k * k * w * h > SR_MAX_PIXELS:
        k -= 1

    def run(t):
        y = net(t)
        if k != SR_SCALE:
            y = F.interpolate(y, size=(t.shape[-2] * k, t.shape[-1] * k), mode="bicubic", antialias=True, align_corners=False)
        return y.clamp(0, 1)  # not in place: the network's own output is an inference tensor
    y = tiled(run, x, SR_TILE, 24, stop, scale=k, store=device if k == 1 else "cpu")  # big results are built in RAM
    if (y.shape[-1], y.shape[-2]) != (tw, th):
        _poll(stop)
        y = F.interpolate(y, size=(th, tw), mode="bicubic", antialias=True, align_corners=False).clamp_(0, 1)
    return y


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
    """Run the enabled steps on an RGB uint8 image and return the RGB uint8 result (same size as the input unless the
    super-resolution step asks for another size, see sr_size)."""
    st = st.clean()
    size = sr_size(rgb.shape[1], rgb.shape[0], st)  # before any work, so a size that is too big fails at once
    cache = cache or StageCache(0)
    tag = hashlib.sha1(rgb.tobytes() + repr(rgb.shape).encode()).hexdigest()
    x = to_tensor(rgb, device)
    key = (tag,)
    if st.speck:
        key += ("speck", st.speck_size, round(st.speck_thr, 3), round(st.speck_ratio, 3))
        x = cache.get(key, lambda: despeckle(x, st.speck_size, st.speck_thr, st.speck_ratio, stop, report), device)
    if st.apisr and st.apisr_amount > 0:
        sharp = cache.get(key + ("apisr", st.apisr_model), lambda: apisr(x, st.apisr_model, device, stop, report), device)
        x = x + st.apisr_amount * (sharp - x)
        key += ("apisr", st.apisr_model, round(st.apisr_amount, 3))
    if st.drunet:
        x = cache.get(key + ("drunet", round(st.sigma, 2)), lambda: drunet_pd(x, st.sigma, device, stop, report), device)
    if st.l0:
        if report:
            report("L0 平滑")
        x = l0_smooth(x, st.lam, stop)
    if st.sr:
        x = superres(x, size, device, stop, report)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return to_rgb(x)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="images or folders")
    ap.add_argument("--out", default=str(OUTPUT_DIR), help="output folder (default: ./output)")
    ap.add_argument("--despeckle", action="store_true", help="remove small bright white specks first")
    ap.add_argument("--speck-size", type=int, default=2, help="largest speck in pixels across (default 2)")
    ap.add_argument("--speck-ratio", type=float, default=1.0,
                    help="share of the specks to remove, picked at random, 0-1 (default 1 = all)")
    ap.add_argument("--speck-thr", type=float, default=0.10,
                    help="how much brighter than its surroundings a speck is, 0-1 (default 0.10; lower removes more)")
    ap.add_argument("--apisr", action="store_true", help="sharpen with APISR first (changes colours a little)")
    ap.add_argument("--apisr-model", default="2x_rrdb", choices=list(APISR))
    ap.add_argument("--apisr-amount", type=float, default=1.0, help="0-1, share of the APISR result (default 1)")
    ap.add_argument("--no-drunet", action="store_true", help="skip DRUNet denoising")
    ap.add_argument("--sigma", type=float, default=10.0, help="DRUNet strength (default 10)")
    ap.add_argument("--l0", action="store_true", help="L0 smoothing at the end")
    ap.add_argument("--lam", type=float, default=0.001, help="L0 strength (default 0.001)")
    ap.add_argument("--sr", nargs="?", const="4x", type=parse_sr, metavar="SIZE",
                    help="Real-ESRGAN 4x super-resolution as the last step (redraws the picture). SIZE: 4x (default), "
                         "orig (back to the input size), 4k / 1080p (fitted into 3840x2160 / 1920x1080, proportions kept), or WIDTHxHEIGHT; 3840x or x2160 keeps the proportions")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: the GPU when there is one)")
    ap.add_argument("--prefetch", nargs="*", metavar="KEY", help="only download weights (default: drunet apisr_2x_rrdb)")
    a = ap.parse_args()

    if a.prefetch is not None:
        for key in a.prefetch or ["drunet", "apisr_2x_rrdb"]:
            path = fetch(key, lambda text: print("\r  " + text + "    ", end="", flush=True))
            print(f"\r  {WEIGHTS[key][0]}：已就绪 ({path.stat().st_size / 1e6:.0f} MB)          ", flush=True)
        return
    if not a.inputs:
        ap.error("give at least one image or folder")
    st = Settings(speck=a.despeckle, speck_size=a.speck_size, speck_thr=a.speck_thr, speck_ratio=a.speck_ratio, apisr=a.apisr, apisr_model=a.apisr_model, apisr_amount=a.apisr_amount, drunet=not a.no_drunet,
                  sigma=a.sigma, l0=a.l0, lam=a.lam)
    if a.sr:
        st = replace(st, sr=True, sr_mode=a.sr[0], sr_w=a.sr[1], sr_h=a.sr[2])
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
        target.write_bytes(png_bytes(res, fit_alpha(alpha, res.shape[1], res.shape[0]), icc, level=6))
        print(f"{p} -> {target} ({time.time() - t0:.1f} s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
