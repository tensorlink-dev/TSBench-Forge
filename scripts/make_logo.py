#!/usr/bin/env python3
"""Generate the TSBench-Forge logo (icon + wordmark) as PNG assets.

Recreates the brand lockup: a cream upward line-chart glyph feeding into a
glowing orange "forge" segment, next to a serif "TSBench-Forge" wordmark
(cream + orange) on the brand navy. Run:

    python scripts/make_logo.py

Writes assets/logo.png (full lockup), assets/logo-mark.png (icon only,
transparent), and a 2x favicon-ish square is not produced here.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---- brand palette -------------------------------------------------------
NAVY = (20, 33, 56)        # background
CREAM = (244, 236, 221)    # primary marks / "TSBench-"
ORANGE = (236, 100, 66)    # accent / "Forge"
ORANGE_CORE = (247, 140, 96)  # hot centre of the glow

SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

SS = 4  # supersample factor for crisp anti-aliasing


def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_mark(draw: ImageDraw.ImageDraw, ox: int, oy: int, s: int, glow_layer):
    """Draw the icon into a box at (ox, oy) of size s (square). y grows down.

    glow_layer is a separate RGBA draw target for the orange glow.
    """
    def P(nx, ny):
        return (ox + nx * s, oy + ny * s)

    lw = max(2, int(s * 0.055))          # stroke width
    r = max(3, int(s * 0.072))           # node radius

    # cream upward trend: A -> B -> C -> D (D highest, top-right)
    A = P(0.06, 0.50)
    B = P(0.30, 0.15)
    C = P(0.52, 0.44)
    D = P(0.78, 0.10)
    # orange forge segment at the base
    E = P(0.42, 0.86)
    F = P(0.78, 0.86)

    def line(p, q, fill, width):
        draw.line([p, q], fill=fill, width=width)

    def node(p, fill, rad):
        x, y = p
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=fill)

    # --- orange glow (drawn on the blurred layer, then composited) ---
    gd = ImageDraw.Draw(glow_layer)
    gwidth = lw + int(s * 0.05)
    grad = r + int(s * 0.05)
    for p, q in [(E, F)]:
        gd.line([p, q], fill=ORANGE + (255,), width=gwidth)
    for p in (E, F):
        x, y = p
        gd.ellipse([x - grad, y - grad, x + grad, y + grad], fill=ORANGE + (255,))
    # vertical drop D -> F participates in the glow too (lower half)
    gd.line([P(0.78, 0.48), F], fill=ORANGE + (200,), width=gwidth)

    # --- cream strokes ---
    line(A, B, CREAM, lw)
    line(B, C, CREAM, lw)
    line(C, D, CREAM, lw)
    # right vertical axis: top node D straight down into the forge
    line(D, F, CREAM, lw)
    # trend descends into the orange base
    line(C, E, ORANGE, lw)

    # --- orange base segment (sharp, on top of glow) ---
    line(E, F, ORANGE, lw)

    # --- nodes ---
    for p in (A, B, C, D):
        node(p, CREAM, r)
    node(E, ORANGE, r)
    node(F, ORANGE_CORE, r)


def build():
    root = Path(__file__).resolve().parent.parent
    assets = root / "assets"
    assets.mkdir(exist_ok=True)

    # ---- layout (in final px; everything is supersampled by SS) ----
    pad = 70
    icon = 230
    gap = 64
    font_px = 190
    part1, part2 = "TSBench-", "Forge"

    font = ImageFont.truetype(SERIF, font_px * SS)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    # tight ink bounds of the full wordmark, at supersampled scale
    full = part1 + part2
    l, t, r, b = probe.textbbox((0, 0), full, font=font)
    text_w = (r - l) / SS
    text_h = (b - t) / SS
    w1 = probe.textlength(part1, font=font) / SS

    content_h = max(icon, text_h)
    W = pad + icon + gap + int(round(text_w)) + pad
    H = pad + int(round(content_h)) + pad

    Wb, Hb = W * SS, H * SS
    cy = Hb // 2
    ix = pad * SS
    icon_box = icon * SS
    iy = cy - icon_box // 2

    # 1) base + glow layer
    base = Image.new("RGBA", (Wb, Hb), NAVY + (255,))
    glow = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    marks = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    draw_mark(ImageDraw.Draw(marks), ix, iy, icon_box, glow)
    glow_blur = glow.filter(ImageFilter.GaussianBlur(radius=16 * SS))

    composed = Image.alpha_composite(base, glow_blur)
    composed = Image.alpha_composite(composed, marks)

    # 2) wordmark, ink-box vertically centred on canvas
    d = ImageDraw.Draw(composed)
    tx = ix + icon_box + gap * SS
    # place so the wordmark's ink top sits at (cy - ink_height/2)
    top_y = cy - (b - t) // 2
    draw_y = top_y - t  # compensate for the font's internal top offset
    d.text((tx, draw_y), part1, font=font, fill=CREAM)
    d.text((tx + int(round(w1 * SS)), draw_y), part2, font=font, fill=ORANGE)

    out = composed.convert("RGB").resize((W, H), Image.LANCZOS)
    out.save(assets / "logo.png")
    print(f"wrote {assets/'logo.png'}  ({W}x{H})")

    # ---- icon-only mark, transparent ----
    m = 40
    mS = 400
    canvas = (mS + 2 * m) * SS
    mi = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    md = ImageDraw.Draw(mi)
    mglow = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw_mark(md, m * SS, m * SS, mS * SS, mglow)
    mglow_b = mglow.filter(ImageFilter.GaussianBlur(radius=18 * SS))
    mcomposed = Image.alpha_composite(mglow_b, mi)
    md2 = ImageDraw.Draw(mcomposed)
    draw_mark(md2, m * SS, m * SS, mS * SS, Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0)))
    mout = mcomposed.resize((mS + 2 * m, mS + 2 * m), Image.LANCZOS)
    mout.save(assets / "logo-mark.png")
    print(f"wrote {assets/'logo-mark.png'}")


if __name__ == "__main__":
    build()
