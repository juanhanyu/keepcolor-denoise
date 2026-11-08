"""Draw the Denoise icon: a colour drop on a rounded square, speckled on the left half (noise) and smooth on the right
(cleaned, same colours). Writes web/icon.svg and the multi-size web/icon.ico + launcher/icon.ico from one geometry.

    python launcher/make_icon.py
"""
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
BG = ("#14b8a6", "#4f46e5")                 # rounded square, top-left -> bottom-right
DROP = ("#fde68a", "#fb7185", "#a78bfa")     # drop, top -> middle -> bottom
CX, CY, R, APEX = 32, 37, 15, 9              # drop: circle centre / radius, apex height (64-unit canvas)
random.seed(7)
DOTS = []
while len(DOTS) < 46:  # speckles inside the left half of the drop
    x, y = random.uniform(16, 31.2), random.uniform(13, 52)
    inside = (x - CX) ** 2 + (y - CY) ** 2 < (R - 0.8) ** 2 or (y < CY and abs(x - CX) < (y - APEX) * 0.62)
    if inside:
        DOTS.append((round(x, 2), round(y, 2), round(random.uniform(0.7, 1.35), 2), random.random() < 0.6))


def drop_points(steps=90):
    def quad(p0, p1, p2, t):
        return tuple((1 - t) ** 2 * a + 2 * (1 - t) * t * b + t * t * c for a, b, c in zip(p0, p1, p2))
    pts = [quad((CX, APEX), (CX + R, 24), (CX + R, CY), i / steps) for i in range(steps)]
    pts += [(CX + R * math.cos(math.pi * i / steps), CY + R * math.sin(math.pi * i / steps)) for i in range(steps + 1)]
    pts += [quad((CX - R, CY), (CX - R, 24), (CX, APEX), i / steps) for i in range(1, steps + 1)]
    return pts


def svg():
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{"#fff" if light else "#1e1b4b"}" opacity="{0.6 if light else 0.32}"/>'
                   for x, y, r, light in DOTS)
    d = f"M{CX} {APEX} Q{CX + R} 24 {CX + R} {CY} A{R} {R} 0 0 1 {CX - R} {CY} Q{CX - R} 24 {CX} {APEX}Z"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            f'<defs><linearGradient id="b" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{BG[0]}"/><stop offset="1" stop-color="{BG[1]}"/></linearGradient>'
            f'<linearGradient id="d" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{DROP[0]}"/><stop offset=".5" stop-color="{DROP[1]}"/><stop offset="1" stop-color="{DROP[2]}"/></linearGradient>'
            f'<clipPath id="c"><path d="{d}"/></clipPath></defs>'
            f'<rect x="2" y="2" width="60" height="60" rx="14" fill="url(#b)"/>'
            f'<path d="{d}" fill="url(#d)"/>'
            f'<g clip-path="url(#c)"><g>{dots}</g><rect x="{CX}" y="0" width="32" height="64" fill="url(#d)"/></g>'
            f'<path d="{d}" fill="none" stroke="#fff" stroke-opacity=".55" stroke-width="1.2"/>'
            f'<path d="M{CX} {APEX + 3} V{CY + R - 2}" stroke="#fff" stroke-width="1.6" stroke-linecap="round" opacity=".95"/></svg>')


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def lerp(c0, c1, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))


def png(size=1024):
    k = size / 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    grad = Image.new("RGBA", (size, size))
    g0, g1 = hexrgb(BG[0]), hexrgb(BG[1])
    px = grad.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = lerp(g0, g1, (x + y) / (2 * size)) + (255,)
    mask = Image.new("L", (size, size))
    ImageDraw.Draw(mask).rounded_rectangle([2 * k, 2 * k, 62 * k, 62 * k], 14 * k, fill=255)
    img.paste(grad, (0, 0), mask)
    # the drop, with its vertical gradient
    dm = Image.new("L", (size, size))
    pts = [(x * k, y * k) for x, y in drop_points()]
    ImageDraw.Draw(dm).polygon(pts, fill=255)
    fill = Image.new("RGBA", (size, size))
    d0, d1, d2 = map(hexrgb, DROP)
    top, bottom = APEX * k, (CY + R) * k
    fp = fill.load()
    for y in range(size):
        t = min(max((y - top) / (bottom - top), 0), 1)
        c = lerp(d0, d1, t * 2) if t < 0.5 else lerp(d1, d2, t * 2 - 1)
        for x in range(size):
            fp[x, y] = c + (255,)
    img.paste(fill, (0, 0), dm)
    # speckles, left half only
    dots = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dots)
    for x, y, r, light in DOTS:
        col = (255, 255, 255, round(255 * 0.6)) if light else hexrgb("#1e1b4b") + (round(255 * 0.32),)
        dd.ellipse([(x - r) * k, (y - r) * k, (x + r) * k, (y + r) * k], fill=col)
    left = Image.new("L", (size, size))
    ImageDraw.Draw(left).rectangle([0, 0, CX * k, size], fill=255)
    from PIL import ImageChops
    img.alpha_composite(Image.composite(dots, Image.new("RGBA", (size, size), (0, 0, 0, 0)), ImageChops.multiply(dm, left)))
    line = ImageDraw.Draw(img)
    line.line(pts + [pts[0]], fill=(255, 255, 255, 140), width=max(1, round(1.2 * k)))
    line.line([(CX * k, (APEX + 3) * k), (CX * k, (CY + R - 2) * k)], fill=(255, 255, 255, 242), width=round(1.6 * k))
    return img


if __name__ == "__main__":
    (ROOT / "web").mkdir(exist_ok=True)
    (ROOT / "web" / "icon.svg").write_text(svg(), "utf-8")
    big = png().resize((256, 256), Image.LANCZOS)
    sizes = [(s, s) for s in (16, 20, 24, 32, 40, 48, 64, 128, 256)]
    for target in (ROOT / "web" / "icon.ico", ROOT / "launcher" / "icon.ico"):
        big.save(target, sizes=sizes)
    big.save(ROOT / "docs" / "icon.png")
    print("wrote web/icon.svg, web/icon.ico, launcher/icon.ico, docs/icon.png")
