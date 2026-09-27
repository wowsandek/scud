#!/usr/bin/env python3
"""Beat-synced travel reel renderer (1080x1920, 30 fps) — motion-design edition.

Cuts 1080x1920 proxies of the source clips on a 120 BPM grid and adds:
  * transitions: zoom punch, whip pan, spin, light leak (+ chromatic split)
  * "text behind the scene": big serif titles composited under hills,
    cliffs and buildings using a SegFormer sky mask (see sky.py)
  * per-letter blur-in reveals, tracking animation, scramble-decoded
    monospace metadata, hairlines, chapter tags with local time
  * beat pulses, vignette, film grain
Audio: music segment + synthesized whooshes / booms on the cuts.

Usage: python3 render.py <work_dir>
  work_dir must contain proxy/c1..c5.mp4, music/823.wav, fonts/*.ttf,
  models/segformer_b2.onnx
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallax import DepthEstimator, warp as parallax_warp  # noqa: E402
from sky import SkySegmenter  # noqa: E402

WORK = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(WORK, "out")
os.makedirs(OUT, exist_ok=True)

W, H, FPS = 1080, 1920, 30
BEAT = 0.5
DUR = 29.5
NFRAMES = int(DUR * FPS)
MUSIC_START = 48.1336  # track time of reel t=0 (bar start, drop lands at t=6.0)
DROP = 6.0
SR = 44100

WHITE = (255, 255, 255)
SHADOW = (8, 12, 20)

GRADE = ("eq=contrast=1.07:saturation=1.18:gamma=0.97,"
         "colorbalance=rs=-0.02:bs=0.03:rh=0.03:bh=-0.02,"
         "unsharp=5:5:0.4")

F_SERIF = "PlayfairItalic.ttf"
F_SANS = "Inter.ttf"
F_MONO = "JetBrainsMono.ttf"

# ---------------------------------------------------------------- edit list
# clip, source in-point, reel start, reel duration, speed ramp [(dur, speed)]
SEGMENTS = [
    ("map", 0.0, 0.0, 6.0, [(6.0, 1.0)]),            # 3D globe + terrain flyover (map3d.py)
    ("c1", 7.45, 6.0, 1.5, [(1.5, 1.0)]),            # 01 Garni — title behind the cliff (drop)
    ("c1", 0.50, 7.5, 1.0, [(1.0, 1.5)]),
    ("c1", 4.60, 8.5, 0.5, [(0.5, 1.5)]),
    ("c1", 13.0, 9.0, 1.0, [(1.0, 1.2)]),
    ("c2", 1.60, 10.0, 1.5, [(1.5, 1.3)]),           # 02 Sevan — title in the sky
    ("c2", 17.0, 11.5, 1.5, [(1.5, 1.0)]),
    ("c3", 12.8, 13.0, 1.5, [(1.5, 1.5)]),           # 03 jet ski (+ chrome 3D word)
    ("c3", 26.6, 14.5, 1.0, [(1.0, 1.5)]),
    ("c3", 41.0, 15.5, 2.5, [(0.5, 2.0), (1.5, 0.5), (0.5, 2.0)]),
    # 04 Sevanavank — frozen frame, 3D orbit around the church, title behind it
    ("c4", 11.9, 18.0, 2.0, [], dict(zoom=(1.03, 1.11), kz=0.08, tx=(48, -48), ty=(-8, 8))),
    ("c2", 12.4, 20.0, 1.0, [(1.0, 1.2)]),
    ("c4", 4.50, 21.0, 1.0, [(1.0, 1.5)]),
    ("c5", 15.4, 22.0, 2.0, [(2.0, 1.0)]),           # 05 peninsula
    ("c5", 17.4, 24.0, 2.5, [(2.5, 0.78)]),
    # end card — frozen panorama, 3D push-in, title behind the ridge
    ("c5", 4.3, 26.5, 3.0, [], dict(zoom=(1.0, 1.08), kz=0.13, tx=(10, -18), ty=(0, -12))),
]

TRANSITIONS = {
    DROP: dict(kind="zoom", amp=0.55, n=(4, 5), flash=0.7, shake=1.0),
    7.5: dict(kind="whip", dir=(0, -1)),
    8.5: dict(kind="whip", dir=(-1, 0)),
    9.0: dict(kind="zoom", amp=0.4, n=(3, 4), shake=0.4),
    10.0: dict(kind="cube", axis="y", n=5),
    11.5: dict(kind="whip", dir=(-1, 0)),
    13.0: dict(kind="cube", axis="x", n=5),
    14.5: dict(kind="whip", dir=(1, 0)),
    15.5: dict(kind="zoom", amp=0.4, n=(3, 4), shake=0.5),
    18.0: dict(kind="zoom", amp=0.5, n=(4, 5), flash=0.35, shake=0.8),
    20.0: dict(kind="whip", dir=(0, -1)),
    21.0: dict(kind="whip", dir=(-1, 0)),
    22.0: dict(kind="flip", n=6),
    24.0: dict(kind="whip", dir=(-1, 0)),
    26.5: dict(kind="leak"),
}

# big serif titles living "in the scene" (under transitions); depth => behind non-sky
TITLES = [
    dict(text="Гарни", start=6.05, end=7.5, size=250, depth=True),
    dict(text="Севан", start=10.05, end=11.5, size=250, depth=True),
    dict(text="Севанаванк", start=18.05, end=20.0, size=190, depth=True, vis=0.95),
    dict(text="Армения", start=26.6, end=29.5, size=235, depth=True, punch=28.0, vis=0.88),
]

CHAPTERS = [
    dict(start=6.15, end=9.92, num="01", time="11:56", title="ГАРНИ · СИМФОНИЯ КАМНЕЙ",
         coords="40.11° N  44.73° E"),
    dict(start=10.15, end=12.92, num="02", time="14:50", title="ОЗЕРО СЕВАН · 1900 М",
         coords="40.55° N  45.00° E"),
    dict(start=13.15, end=17.92, num="03", time="15:26", title="ГИДРОЦИКЛ ПО СЕВАНУ", coords=None),
    dict(start=18.15, end=21.92, num="04", time="16:03", title="СЕВАНАВАНК · IX ВЕК",
         coords="40.56° N  45.01° E"),
    dict(start=22.15, end=26.42, num="05", time="16:22", title="ПОЛУОСТРОВ СЕВАН", coords=None),
]

CHROME_AT = 13.1      # chrome 3D word (map3d.py chrome), frames 0..72
CHROME_DY = -330      # shift up from frame centre

# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def expo_out(p):
    p = clamp(p)
    return 1.0 if p >= 1 else 1 - 2 ** (-10 * p)


def cubic_out(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3


def cubic_in(p):
    return clamp(p) ** 3


# ---------------------------------------------------------------- fonts / glyphs
_font_cache = {}


def font(name, size, axes):
    key = (name, size, tuple(axes))
    if key not in _font_cache:
        f = ImageFont.truetype(os.path.join(WORK, "fonts", name), size)
        f.set_variation_by_axes(list(axes))
        _font_cache[key] = f
    return _font_cache[key]


def serif(size):
    return font(F_SERIF, size, [500])


def sans(size, weight=600):
    return font(F_SANS, size, [min(32, max(14, size)), weight])


def mono(size, weight=500):
    return font(F_MONO, size, [weight])


def glyph_sprite(ch, fnt, fill=WHITE, shadow=0.0, shadow_blur=14, pad=None):
    """One glyph as straight-alpha BGRA float32, fixed box (asc+desc) so baselines align.
    Returns sprite and the x of the pen origin inside the sprite."""
    asc, desc = fnt.getmetrics()
    pad = pad if pad is not None else int(fnt.size * 0.45)
    adv = fnt.getlength(ch)
    w, h = int(adv + 2 * pad), asc + desc + 2 * pad
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((pad, pad), ch, font=fnt, fill=255)
    a = np.asarray(m, np.float32) / 255
    if shadow > 0:
        sh = np.asarray(m.filter(ImageFilter.GaussianBlur(shadow_blur)), np.float32) / 255 * shadow
        sh = np.roll(sh, int(shadow_blur * 0.3), axis=0)
    else:
        sh = np.zeros_like(a)
    alpha = a + sh * (1 - a)
    col = np.empty((h, w, 3), np.float32)
    fg = np.array(fill[::-1], np.float32)
    bg = np.array(SHADOW[::-1], np.float32)
    wf = (a / np.maximum(alpha, 1e-6))[..., None]
    col[:] = fg * wf + bg * (1 - wf)
    return np.dstack([col, alpha * 255]), pad


class Word:
    """A line of text laid out glyph by glyph, drawable with per-glyph transforms."""

    def __init__(self, text, fnt, fill=WHITE, shadow=0.0, tracking=0.0, shadow_blur=14):
        self.text = text
        self.fnt = fnt
        self.glyphs, self.adv = [], []
        for ch in text:
            spr, pad = glyph_sprite(ch, fnt, fill, shadow, shadow_blur)
            self.glyphs.append((spr, pad))
            self.adv.append(fnt.getlength(ch))
        self.tracking = tracking
        self.asc, self.desc = fnt.getmetrics()
        self._blur_cache = {}

    def width(self, tracking=None):
        tr = self.tracking if tracking is None else tracking
        return sum(self.adv) + tr * (len(self.text) - 1)

    def origins(self, x_left, tracking=None):
        tr = self.tracking if tracking is None else tracking
        xs, x = [], x_left
        for a in self.adv:
            xs.append(x)
            x += a + tr
        return xs

    def blurred(self, i, sigma):
        s = round(sigma * 2) / 2
        if s < 0.5:
            return self.glyphs[i][0]
        key = (i, s)
        if key not in self._blur_cache:
            self._blur_cache[key] = cv2.GaussianBlur(self.glyphs[i][0], (0, 0), s)
        return self._blur_cache[key]


# ---------------------------------------------------------------- layers
class Layer:
    """Premultiplied RGBA accumulation buffer with a dirty rectangle."""

    def __init__(self):
        self.C = np.zeros((H, W, 3), np.float32)
        self.A = np.zeros((H, W), np.float32)
        self.box = None

    def _grow(self, x0, y0, x1, y1):
        if self.box is None:
            self.box = [x0, y0, x1, y1]
        else:
            b = self.box
            self.box = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]

    def _blend(self, X0, Y0, X1, Y1, rgb, a):
        self.C[Y0:Y1, X0:X1] = self.C[Y0:Y1, X0:X1] * (1 - a[..., None]) + rgb * a[..., None]
        self.A[Y0:Y1, X0:X1] = self.A[Y0:Y1, X0:X1] * (1 - a) + a
        self._grow(X0, Y0, X1, Y1)

    def draw(self, spr, x0, y0, alpha=1.0, scale=1.0):
        """Draw a straight-alpha float sprite with its top-left at (x0, y0) (after scaling)."""
        if alpha <= 0.003:
            return
        if abs(scale - 1) > 1e-3:
            if scale < 0.03:
                return
            spr = cv2.resize(spr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
        h, w = spr.shape[:2]
        x0, y0 = int(round(x0)), int(round(y0))
        X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if X1 <= X0 or Y1 <= Y0:
            return
        s = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        self._blend(X0, Y0, X1, Y1, s[..., :3], s[..., 3] / 255 * alpha)

    def _shape(self, draw_fn, bx0, by0, bx1, by1, alpha):
        bx0, by0 = max(0, int(bx0)), max(0, int(by0))
        bx1, by1 = min(W, int(bx1)), min(H, int(by1))
        if bx1 <= bx0 or by1 <= by0:
            return
        m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        draw_fn(m, bx0, by0)
        self._blend(bx0, by0, bx1, by1, 255.0, m.astype(np.float32) / 255 * alpha)

    def line(self, x0, y0, x1, y1, alpha=1.0, thick=2):
        if alpha <= 0.003 or (abs(x1 - x0) < 1 and abs(y1 - y0) < 1):
            return
        s = 16  # sub-pixel precision for cv2 drawing

        def fn(m, ox, oy):
            cv2.line(m, (int((x0 - ox) * s), int((y0 - oy) * s)), (int((x1 - ox) * s), int((y1 - oy) * s)),
                     255, thick, cv2.LINE_AA, 4)
        self._shape(fn, min(x0, x1) - thick - 2, min(y0, y1) - thick - 2,
                    max(x0, x1) + thick + 3, max(y0, y1) + thick + 3, alpha)

    def dot(self, x, y, r, alpha=1.0):
        if alpha <= 0.003 or r <= 0.3:
            return

        def fn(m, ox, oy):
            cv2.circle(m, (int((x - ox) * 16), int((y - oy) * 16)), int(r * 16), 255, -1, cv2.LINE_AA, 4)
        self._shape(fn, x - r - 2, y - r - 2, x + r + 3, y + r + 3, alpha)

    def composite(self, frame, mask=None):
        if self.box is None:
            return frame
        x0, y0, x1, y1 = self.box
        A = self.A[y0:y1, x0:x1]
        C = self.C[y0:y1, x0:x1]
        if mask is not None:
            m = mask[y0:y1, x0:x1]
            A = A * m
            C = C * m[..., None]
        f = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (f * (1 - A[..., None]) + C).clip(0, 255).astype(np.uint8)
        self.C[y0:y1, x0:x1] = 0
        self.A[y0:y1, x0:x1] = 0
        self.box = None
        return frame


# ---------------------------------------------------------------- text animators
def draw_blur_in(layer, word, cx, cy, t, t0, stagger=0.06, dur=0.6, sigma0=18, rise=46,
                 spread=26, scale=1.0, alpha=1.0, exit_t=None, exit_dur=0.3):
    """Glyphs fade in from blur, rising and converging; optional blur-out exit.
    (cx, cy) is the centre of the x-height band."""
    n = len(word.text)
    width = word.width()
    xs = word.origins(cx - width / 2)
    base = cy + (word.asc * 0.62) / 2  # baseline so that the x-height is centred on cy
    for i, ch in enumerate(word.text):
        if ch == " ":
            continue
        p = expo_out((t - t0 - i * stagger) / dur)
        if p <= 0:
            continue
        sigma = sigma0 * (1 - p)
        a = min(1.0, p * 1.3) * alpha
        dy = rise * (1 - p)
        dx = (i - (n - 1) / 2) * spread * (1 - p)
        if exit_t is not None and t > exit_t:
            q = cubic_in((t - exit_t - i * 0.02) / exit_dur)
            sigma += 20 * q
            a *= 1 - q
            dy -= 30 * q
        spr = word.blurred(i, sigma)
        pad = word.glyphs[i][1]
        # glyph box top-left before scaling, then scale about (cx, cy)
        gx = xs[i] - pad + dx
        gy = base - word.asc - pad + dy
        layer.draw(spr, cx + (gx - cx) * scale, cy + (gy - cy) * scale, a, scale)


class Scramble:
    """Monospace text that decodes from random characters, left to right."""
    POOL = "0123456789#%&*+=/<>"

    def __init__(self, text, fnt, fill=WHITE, shadow=0.45):
        self.text = text
        self.fnt = fnt
        self.adv = fnt.getlength("0")
        self.cache = {}
        self.fill, self.shadow = fill, shadow
        self.asc, self.desc = fnt.getmetrics()

    def sprite(self, ch):
        if ch not in self.cache:
            self.cache[ch] = glyph_sprite(ch, self.fnt, self.fill, self.shadow, 6, pad=12)
        return self.cache[ch]

    def draw(self, layer, x, y, t, t0, fi, alpha=1.0, anchor="left", per=0.022):
        """y is the baseline."""
        if t < t0:
            return
        width = self.adv * len(self.text)
        x0 = x - (width / 2 if anchor == "center" else 0)
        rng = np.random.default_rng(fi * 131 + len(self.text))
        for i, ch in enumerate(self.text):
            if ch == " ":
                continue
            appear = t0 + i * per * 0.5
            reveal = t0 + 0.12 + i * per
            if t < appear:
                continue
            c = ch if t >= reveal else self.POOL[rng.integers(len(self.POOL))]
            spr, pad = self.sprite(c)
            layer.draw(spr, x0 + i * self.adv - pad, y - self.asc - pad,
                       alpha * (1.0 if t >= reveal else 0.55))


def draw_type_on(layer, word, x, y, t, t0, alpha=1.0, per=0.016, dur=0.35, anchor="left"):
    """Tracked caps line: glyphs fade/slide in one after another. y is the baseline."""
    width = word.width()
    x0 = x - (width / 2 if anchor == "center" else 0)
    xs = word.origins(x0)
    for i, ch in enumerate(word.text):
        if ch == " ":
            continue
        p = expo_out((t - t0 - i * per) / dur)
        if p <= 0:
            continue
        spr, pad = word.glyphs[i]
        layer.draw(spr, xs[i] - pad, y - word.asc - pad + 14 * (1 - p), alpha * p)


# ---------------------------------------------------------------- graphics blocks
class Title:
    def __init__(self, spec):
        self.spec = spec
        self.pos = None  # (cx, cy), decided by place_titles()
        self._build(spec["size"])

    def _build(self, size):
        self.word = Word(self.spec["text"], serif(size), WHITE, shadow=0.38, tracking=size * 0.01,
                         shadow_blur=int(size * 0.09))

    def resize(self, k):
        self._build(int(self.word.fnt.size * k))

    def alive(self, t):
        return self.spec["start"] <= t < self.spec["end"]

    def draw(self, layer, t):
        s = self.spec
        life = (t - s["start"]) / (s["end"] - s["start"])
        scale = 1.0 + 0.04 * life
        if s.get("punch") and t >= s["punch"]:
            scale *= 1 + 0.05 * math.exp(-(t - s["punch"]) / 0.15)
        draw_blur_in(layer, self.word, self.pos[0], self.pos[1], t, s["start"],
                     stagger=0.055, dur=0.65, scale=scale)

    def alpha_footprint(self):
        """Final-state alpha (without shadow halo) and its offset from the centre point."""
        lay = Layer()
        draw_blur_in(lay, self.word, W / 2, H / 2, 99, 0)
        x0, y0, x1, y1 = lay.box
        a = lay.A[y0:y1, x0:x1].copy()
        a = np.where(a > 0.6, a, 0)
        lay.composite(np.zeros((H, W, 3), np.uint8))
        ys, xs = np.nonzero(a)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return a, x0 + xs.min() - W / 2, y0 + ys.min() - H / 2


class Chapter:
    X = 72
    Y = 1262

    def __init__(self, spec):
        self.spec = spec
        self.head = Scramble(f"{spec['num']} — {spec['time']}", mono(30, 500), WHITE, 0.5)
        self.title = Word(spec["title"], sans(30, 600), WHITE, shadow=0.5, tracking=4.5, shadow_blur=6)
        self.coords = Scramble(spec["coords"], mono(26, 500), WHITE, 0.5) if spec["coords"] else None

    def visibility(self, t):
        s = self.spec
        if not (s["start"] - 0.1 <= t < s["end"]):
            return 0.0
        return clamp((t - s["start"] + 0.1) / 0.3) * (1 - clamp((t - (s["end"] - 0.22)) / 0.22))

    def draw(self, layer, t, fi):
        s = self.spec
        if not (s["start"] <= t < s["end"]):
            return
        out = clamp((t - (s["end"] - 0.22)) / 0.22)
        a = 1 - cubic_in(out)
        x = self.X - 24 * cubic_in(out)
        y = self.Y
        self.head.draw(layer, x, y, t, s["start"], fi, 0.92 * a)
        lp = expo_out((t - s["start"] - 0.1) / 0.6)
        layer.line(x, y + 26, x + 210 * lp, y + 26, 0.75 * a, 2)
        draw_type_on(layer, self.title, x, y + 78, t, s["start"] + 0.12, a)
        if self.coords:
            self.coords.draw(layer, x, y + 124, t, s["start"] + 0.3, fi, 0.82 * a, per=0.018)


class Intro:
    """Title over the 3D globe, gone before the dive into the terrain."""

    def __init__(self):
        self.kicker = Word("ОДИН ДЕНЬ", sans(34, 500), WHITE, shadow=0.5, tracking=14, shadow_blur=8)
        f = serif(186)
        while f.getlength("в Армении") > 900:
            f = serif(f.size - 6)
        self.title = Word("в Армении", f, WHITE, shadow=0.42, tracking=2, shadow_blur=16)
        self.date = Scramble("27.09.2026", mono(28, 500), WHITE, 0.5)
        self.credit = Word("SENTINEL-2 CLOUDLESS 2021 © EOX · NASA BLUE MARBLE", sans(17, 500), WHITE, 0.5,
                           tracking=2, shadow_blur=4)

    def draw(self, layer, t, fi):
        if 1.9 <= t < DROP:
            a = 0.55 * clamp((t - 1.9) / 0.3) * (1 - clamp((t - (DROP - 0.3)) / 0.3))
            draw_type_on(layer, self.credit, W / 2, 1488, t, 1.9, a, per=0.004, anchor="center")
        if t >= 2.0:
            return
        fade = 1 - clamp((t - 1.45) / 0.3)
        kp = expo_out((t - 0.15) / 0.9)
        tr = 40 - 26 * kp
        width = self.kicker.width(tr)
        xs = self.kicker.origins(W / 2 - width / 2, tr)
        ky = 330
        for i, (spr, pad) in enumerate(self.kicker.glyphs):
            if self.kicker.text[i] != " ":
                layer.draw(spr, xs[i] - pad, ky - self.kicker.asc - pad, clamp(kp * 1.4) * fade)
        lp = expo_out((t - 0.35) / 0.8)
        gap, ln = 26, 110 * lp
        ly = ky - 13
        layer.line(W / 2 - width / 2 - gap - ln, ly, W / 2 - width / 2 - gap, ly, 0.8 * fade, 2)
        layer.line(W / 2 + width / 2 + gap, ly, W / 2 + width / 2 + gap + ln, ly, 0.8 * fade, 2)
        scale = 1 + 0.06 * clamp(t / 1.9)
        draw_blur_in(layer, self.title, W / 2, 455, t, 0.4, stagger=0.05, dur=0.6, scale=scale,
                     exit_t=1.5, exit_dur=0.35)
        self.date.draw(layer, W / 2, 598, t, 0.9, fi, 0.85 * fade, anchor="center")


class EndCard:
    def __init__(self):
        self.stops = [("ГАРНИ", 190), ("СЕВАН", 540), ("СЕВАНАВАНК", 890)]
        self.names = [Word(n, sans(24, 600), WHITE, 0.5, tracking=4, shadow_blur=6) for n, _ in self.stops]
        # Stories have no "save" button for viewers, so the reel CTA is off by default
        cta = os.environ.get("CTA", "")
        self.cta = Word(cta, sans(28, 500), WHITE, 0.5, tracking=7, shadow_blur=6) if cta else None

    def draw(self, layer, t):
        if t < 26.9:
            return
        y = 1215
        x0, x1 = self.stops[0][1], self.stops[-1][1]
        lp = cubic_out((t - 27.0) / 1.0)
        if lp > 0:
            layer.line(x0, y, x0 + (x1 - x0) * lp, y, 0.8, 2)
        for i, (name, x) in enumerate(self.stops):
            reach = (x - x0) / (x1 - x0)
            t_reach = 27.0 + 1 - (1 - reach) ** (1 / 3)
            p = expo_out((t - t_reach + 0.04) / 0.4)
            if p > 0:
                layer.dot(x, y, 7 * p + 3 * math.exp(-max(0, t - t_reach) / 0.12))
                draw_type_on(layer, self.names[i], x, y + 52, t, t_reach, 1.0, per=0.012, anchor="center")
        if self.cta and t > 28.1:
            draw_type_on(layer, self.cta, W / 2, 1370, t, 28.1, 0.95, per=0.02, anchor="center")
            half = self.cta.width() / 2 * expo_out((t - 28.1) / 0.8)
            layer.line(W / 2 - half, 1392, W / 2 + half, 1392, 0.6, 2)


# ---------------------------------------------------------------- frame fx
def affine(img, scale=1.0, angle=0.0, dx=0.0, dy=0.0):
    if abs(scale - 1) < 1e-4 and abs(angle) < 1e-4 and abs(dx) < 0.5 and abs(dy) < 0.5:
        return img
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def zoom_blur(img, scale, amount, n=6):
    if amount < 0.004:
        return affine(img, scale)
    acc = np.zeros((H, W, 3), np.float32)
    for i in range(n):
        acc += affine(img, scale * (1 + amount * i / (n - 1)))
    return (acc / n).astype(np.uint8)


def motion_blur(img, dx, dy):
    k = int(abs(dx) + abs(dy))
    if k < 3:
        return img
    k = min(k, 301) | 1
    return cv2.blur(img, (k, 1) if abs(dx) >= abs(dy) else (1, k))


def rgb_split(img, px):
    px = int(round(px))
    if px == 0:
        return img
    out = img.copy()
    out[..., 2] = np.roll(img[..., 2], px, axis=1)
    out[..., 0] = np.roll(img[..., 0], -px, axis=1)
    return out


def flash(img, a):
    if a <= 0.004:
        return img
    return (img.astype(np.float32) * (1 - a) + 255 * a).astype(np.uint8)


def light_leak(t_rel):
    yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    cx = W * (0.1 + 0.9 * t_rel)
    cy = H * (0.35 - 0.15 * t_rel)
    g = np.exp(-(((xx - cx) / (W * 0.55)) ** 2 + ((yy - cy) / (H * 0.45)) ** 2))
    g = cv2.resize(g, (W, H))
    return g[..., None] * (np.array([60, 150, 255], np.float32) / 255)


def build_vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return (1 - 0.3 * np.clip(r / 1.3, 0, 1) ** 2.4)[..., None]


def build_bottom_shade():
    y = np.linspace(0, 1, H, dtype=np.float32)
    g = np.clip((y - 0.55) / 0.2, 0, 1) ** 1.4 * 0.45
    g = g * np.clip((0.86 - y) / 0.1, 0.4, 1)
    return g[:, None, None]


VIGN = build_vignette()
SHADE = build_bottom_shade()
_grain_rng = np.random.default_rng(7)
GRAIN = [cv2.resize(_grain_rng.normal(0, 4.2, (H // 2, W // 2)).astype(np.float32), (W, H))
         for _ in range(6)]


def transition_fx(frame, fi):
    post = {"flash": 0.0, "rgb": 0.0, "shake": 0.0, "leak": None}
    for cut, tr in TRANSITIONS.items():
        k = fi - int(round(cut * FPS))  # <0 outgoing, >=0 incoming
        kind = tr["kind"]
        if kind == "zoom":
            n_out, n_in = tr["n"]
            amp = tr["amp"]
            if -n_out <= k < 0:
                p = (n_out + k + 1) / n_out
                frame = zoom_blur(frame, 1 + amp * p ** 2, 0.18 * p)
                post["rgb"] = max(post["rgb"], 10 * p)
            elif 0 <= k < n_in:
                q = 1 - k / n_in
                frame = zoom_blur(frame, 1 + amp * 0.8 * q ** 2, 0.14 * q)
                post["rgb"] = max(post["rgb"], 12 * q)
                if tr.get("flash"):
                    post["flash"] = max(post["flash"], tr["flash"] * [1, 0.55, 0.25, 0.1, 0][min(k, 4)])
                if tr.get("shake"):
                    post["shake"] = max(post["shake"], tr["shake"])
        elif kind == "whip":
            dx, dy = tr["dir"]
            n, L = 3, (W if dx else H)
            if -n <= k < 0:
                p = (n + k + 1) / n
                off = L * 0.45 * p ** 2
                frame = motion_blur(affine(frame, 1.0, 0, dx * off, dy * off), dx * 260 * p, dy * 260 * p)
            elif 0 <= k < n:
                q = 1 - (k + 1) / (n + 1)
                off = -L * 0.45 * q ** 2
                frame = motion_blur(affine(frame, 1.0, 0, dx * off, dy * off), dx * 260 * q, dy * 260 * q)
        elif kind == "spin":
            n = 4
            if -n <= k < 0:
                p = (n + k + 1) / n
                frame = zoom_blur(affine(frame, 1 + 0.25 * p, 55 * p ** 2), 1.0, 0.12 * p, 4)
                post["rgb"] = max(post["rgb"], 8 * p)
            elif 0 <= k < n:
                q = 1 - k / n
                frame = zoom_blur(affine(frame, 1 + 0.25 * q, -55 * q ** 2), 1.0, 0.12 * q, 4)
                post["rgb"] = max(post["rgb"], 8 * q)
        elif kind == "leak":
            if 0 <= k < 16:
                q = k / 16
                frame = affine(frame, 1 + 0.2 * (1 - cubic_out(q)))
                post["leak"] = (q, 0.85 * (1 - q) ** 1.5)
                if k < 3:
                    post["flash"] = max(post["flash"], [0.35, 0.15, 0.05][k])
            elif -3 <= k < 0:
                post["leak"] = (0.0, 0.3 * (4 + k) / 3)
    return frame, post


def beat_pulse(t):
    if t < DROP or t >= 26.5:
        return 1.0
    b = math.floor(t / BEAT + 1e-6) * BEAT
    strong = abs(b / 2.0 - round(b / 2.0)) < 1e-6
    return 1 + (0.04 if strong else 0.022) * math.exp(-(t - b) / 0.11)


# ---------------------------------------------------------------- 3D transitions
def _project(pts, f, z_off):
    """Pinhole projection of Nx3 points (camera at origin looking +z, y down)."""
    z = pts[:, 2] + z_off
    return np.stack([W / 2 + f * pts[:, 0] / z, H / 2 + f * pts[:, 1] / z], 1).astype(np.float32)


def _backdrop(A, B, e):
    small = cv2.resize(cv2.addWeighted(A, 1 - e, B, e, 0), (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    return (cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (W, H)).astype(np.float32) * 0.35).astype(np.uint8)


def _paint_face(canvas, img, quad, shade):
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    M = cv2.getPerspectiveTransform(src, quad)
    face = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    m = cv2.warpPerspective(np.full((H, W), 255, np.uint8), M, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT).astype(np.float32)[..., None] / 255
    return (canvas * (1 - m) + face.astype(np.float32) * shade * m).astype(np.uint8)


def box_transition(A, B, p, axis):
    """A on the front face of a box that turns 90 degrees to show B on the next face.
    axis 'y': A leaves to the left, B arrives from the right; 'x': A goes up, B from below."""
    e = 0.5 - 0.5 * math.cos(math.pi * clamp(p))
    th = e * math.pi / 2
    depth = W if axis == "y" else H
    f = 2.2 * depth                       # front face fills the frame at rest
    z_off = f + depth / 2 + 0.45 * depth * math.sin(math.pi * e)   # dolly out mid-turn
    hw, hh, hd = W / 2, H / 2, depth / 2
    front = np.array([[-hw, -hh, -hd], [hw, -hh, -hd], [hw, hh, -hd], [-hw, hh, -hd]], np.float64)
    if axis == "y":
        def rot(v, a):
            ca, sa = math.cos(a), math.sin(a)
            return np.stack([v[:, 0] * ca + v[:, 2] * sa, v[:, 1], -v[:, 0] * sa + v[:, 2] * ca], 1)
    else:
        def rot(v, a):
            ca, sa = math.cos(a), math.sin(a)
            return np.stack([v[:, 0], v[:, 1] * ca + v[:, 2] * sa, -v[:, 1] * sa + v[:, 2] * ca], 1)
    faces = [(A, rot(front, th)), (B, rot(rot(front, -math.pi / 2), th))]
    out = _backdrop(A, B, e)
    order = sorted(faces, key=lambda fc: -fc[1][:, 2].mean())   # far first
    for img, pts in order:
        n = np.cross(pts[1] - pts[0], pts[3] - pts[0])
        centre = pts.mean(0) + np.array([0, 0, z_off])
        if np.dot(n, centre) <= 0:        # n points into the box: visible when it points away from us
            continue
        cosang = abs(np.dot(n / np.linalg.norm(n), centre / np.linalg.norm(centre)))
        out = _paint_face(out, img, _project(pts, f, z_off), 0.45 + 0.55 * cosang)
    return out


def flip_transition(A, B, p):
    """Card flip around the horizontal axis with a dolly out; A on the front, B on the back."""
    e = 0.5 - 0.5 * math.cos(math.pi * clamp(p))
    th = e * math.pi
    f = 2.4 * H
    z_off = f + 0.5 * H * math.sin(math.pi * e)
    hw, hh = W / 2, H / 2
    card = np.array([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]], np.float64)
    img = A if th < math.pi / 2 else B
    a = th if th < math.pi / 2 else th - math.pi
    ca, sa = math.cos(a), math.sin(a)
    pts = np.stack([card[:, 0], card[:, 1] * ca, card[:, 1] * sa], 1)
    out = _backdrop(A, B, e)
    return _paint_face(out, img, _project(pts, f, z_off), 0.5 + 0.5 * abs(ca))


# ---------------------------------------------------------------- sources
_SEG = None
_DEPTH = None


def sky_segmenter():
    global _SEG
    if _SEG is None:
        _SEG = SkySegmenter(os.path.join(WORK, "models", "segformer_b2.onnx"), os.path.join(WORK, "skycache"))
    return _SEG


def depth_estimator():
    global _DEPTH
    if _DEPTH is None:
        _DEPTH = DepthEstimator(os.path.join(WORK, "models", "depth_anything_v2_small.onnx"),
                                os.path.join(WORK, "depthcache"))
    return _DEPTH


class SegmentReader:
    def __init__(self, clip, src_in):
        self.clip, self.src_in = clip, src_in
        if clip == "map":
            path, vf, self.fps = os.path.join(OUT, "map.mp4"), "null", 30
        else:
            path, vf, self.fps = os.path.join(WORK, "proxy", clip + ".mp4"), GRADE, 60
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{src_in:.3f}", "-i", path, "-vf", vf, "-an",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 3 * 4)
        self.idx = -1
        self.frame = None

    def get(self, src_t):
        want = max(0, int(round((src_t - self.src_in) * self.fps)))
        while self.idx < want:
            buf = self.proc.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            self.frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            self.idx += 1
        return self.frame.copy()

    def key(self):
        return f"{self.clip}:{self.src_in:.3f}:{self.idx}"

    def close(self):
        self.proc.stdout.close()
        self.proc.kill()
        self.proc.wait()


def src_time(seg, lt):
    t = seg[1]
    for d, sp in seg[4]:
        step = min(lt, d)
        t += step * sp
        lt -= step
        if lt <= 0:
            break
    return t


def seg_at(t):
    return next(s for s in SEGMENTS if s[2] <= t + 1e-9 < s[2] + s[3])


class FrameSource:
    """Frames of the edit at reel time t; frozen segments get a depth-parallax camera move."""

    def __init__(self):
        self.reader, self.cur = None, None
        self.still = None

    def get(self, t):
        seg = seg_at(t)
        if seg is not self.cur:
            if self.reader:
                self.reader.close()
                self.reader = None
            self.cur = seg
            if len(seg) > 5:
                r = SegmentReader(seg[0], seg[1])
                img = r.get(seg[1])
                r.close()
                k = f"{seg[0]}:{seg[1]:.3f}:still"
                disp = depth_estimator().disparity(img, k, sky_segmenter().mask(img, k))
                self.still = (img, disp)
            else:
                self.reader = SegmentReader(seg[0], seg[1])
        if len(seg) > 5:
            mv = seg[5]
            u = clamp((t - seg[2]) / seg[3])
            e = 0.5 - 0.5 * math.cos(math.pi * u)

            def lerp(v):
                return v[0] + (v[1] - v[0]) * e if isinstance(v, tuple) else v
            img, disp = self.still
            frame = parallax_warp(img, disp, lerp(mv["zoom"]), mv["kz"], lerp(mv["tx"]), lerp(mv["ty"]))
            return frame, f"{seg[0]}:{seg[1]:.3f}:f3d:{int(round(u * 1000))}"
        return self.reader.get(src_time(seg, t - seg[2])), self.reader.key()

    def close(self):
        if self.reader:
            self.reader.close()


# ---------------------------------------------------------------- title placement
def place_titles(titles, seg):
    """Put each depth title where the scene hides the lower part of the letters
    (about `vis` of the glyph area stays visible on average) while their upper
    part stays in open sky in every frame of the shot, even as the camera moves."""
    src = FrameSource()
    f = 6
    for tt in titles:
        s = tt.spec
        masks = []
        t = s["start"] + 0.05
        while t < s["end"] - 0.02:
            frame, key = src.get(t)
            masks.append(cv2.resize(seg.mask(frame, key), (W // f, H // f), interpolation=cv2.INTER_AREA))
            t += 3 / FPS
        target = s.get("vis", 0.8)
        best = None
        # first try the full size with strict, then relaxed per-frame limits; then shrink
        plan = [(0.9, 0.15), (0.8, 0.3)] + [(0.9, 0.15)] * 5
        for attempt, (top_min, slack) in enumerate(plan):
            if attempt >= 2:
                tt.resize(0.88)  # does not fit anywhere: shrink and retry
            fa, ox, oy = tt.alpha_footprint()
            fa_s = cv2.resize(fa, (max(1, fa.shape[1] // f), max(1, fa.shape[0] // f)),
                              interpolation=cv2.INTER_AREA).astype(np.float32)
            hs, ws = fa_s.shape
            top = fa_s.copy()
            top[int(hs * 0.55):] = 0
            # correlation maps: value at (y0, x0) = coverage of the text box whose top-left is there
            vis = np.stack([cv2.filter2D(m, -1, fa_s, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
                            for m in masks]) / fa_s.sum()
            vtop = np.stack([cv2.filter2D(m, -1, top, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
                             for m in masks]) / top.sum()
            v_mean, v_min, t_min = vis.mean(0), vis.min(0), vtop.min(0)
            for y0 in range(240 // f, 1250 // f - hs):
                for x0 in range(50 // f, (W - 50) // f - ws):
                    if t_min[y0, x0] < top_min or v_min[y0, x0] < target - slack:
                        continue
                    cx = (x0 - ox / f) * f
                    cy = (y0 - oy / f) * f
                    vm = v_mean[y0, x0]
                    if vm > 0.97:  # nothing in front: open sky, a little above the middle
                        score = -0.4 - abs(cy - 620) / 3000 - abs(cx - W / 2) / 2000
                    else:
                        score = -abs(vm - target) - abs(cx - W / 2) / 3000
                    if best is None or score > best[0]:
                        best = (score, int(cx), int(cy), vm, v_min[y0, x0], tt.word.fnt.size)
            if best is not None:
                break
        if best is None:
            best = (0, W // 2, 620, -1, -1, tt.word.fnt.size)
        tt.pos = (best[1], best[2])
        print(f"title {s['text']}: size={best[5]} pos={tt.pos} visible mean={best[3]:.2f} "
              f"min={best[4]:.2f}", flush=True)
    src.close()


# ---------------------------------------------------------------- render
def render_video(path):
    seg = sky_segmenter()
    titles = [Title(s) for s in TITLES]
    place_titles(titles, seg)
    chapters = [Chapter(c) for c in CHAPTERS]
    intro, end = Intro(), EndCard()
    scene, hud = Layer(), Layer()
    boxes = {int(round(c * FPS)): tr for c, tr in TRANSITIONS.items() if tr["kind"] in ("cube", "flip")}
    held, peeked = {}, {}
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
         "-maxrate", "16M", "-bufsize", "32M", "-pix_fmt", "yuv420p", "-profile:v", "high",
         "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    src = FrameSource()
    shake = 0.0
    chrome_f0 = int(round(CHROME_AT * FPS))
    frames = range(NFRAMES)
    if os.environ.get("FRAMES"):
        a, b = (int(v) for v in os.environ["FRAMES"].split(":"))
        frames = range(a, b)
    for fi in frames:
        t = fi / FPS
        frame, key = src.get(t)

        # 1) scene text: composited into the footage, so transitions move it too
        live = [tt for tt in titles if tt.alive(t)]
        if live:
            for tt in live:
                tt.draw(scene, t)
            mask = seg.mask(frame, key) if any(tt.spec["depth"] for tt in live) else None
            frame = scene.composite(frame, mask)

        # 2) 3D box / flip transitions need both shots at once
        for cf, tr in boxes.items():
            n = tr["n"]
            if fi == cf - 1:
                held[cf] = frame.copy()
            if cf - n <= fi < cf + n:
                if fi < cf:
                    if cf not in peeked:
                        ps = FrameSource()
                        peeked[cf] = ps.get(cf / FPS)[0]
                        ps.close()
                    A, B = frame, peeked[cf]
                else:
                    A, B = held.get(cf, frame), frame
                p = (fi - (cf - n) + 0.5) / (2 * n)
                frame = box_transition(A, B, p, tr["axis"]) if tr["kind"] == "cube" else flip_transition(A, B, p)

        # 3) cut transitions, beat pulse, shake
        frame, post = transition_fx(frame, fi)
        if 28.0 <= t < 28.2:
            post["flash"] = max(post["flash"], 0.3 * (1 - (t - 28.0) / 0.2))
            post["shake"] = max(post["shake"], 0.6)
        shake = max(shake * 0.72, post["shake"])
        if shake > 0.05:
            rng = np.random.default_rng(fi * 7919)
            sx, sy, sa = rng.normal(0, 13 * shake), rng.normal(0, 13 * shake), rng.normal(0, 0.6 * shake)
            frame = affine(frame, beat_pulse(t) * 1.02, sa, sx, sy)
        else:
            frame = affine(frame, beat_pulse(t))

        # 4) look
        if t >= DROP:
            frame = (frame.astype(np.float32) * VIGN).astype(np.uint8)
        if post["leak"] is not None:
            q, a = post["leak"]
            f = frame.astype(np.float32) / 255
            frame = ((1 - (1 - f) * (1 - light_leak(q) * a)) * 255).astype(np.uint8)
        if t >= 26.5:
            d = 0.22 * clamp((t - 26.5) / 0.4)
            frame = (frame.astype(np.float32) * (1 - d)).astype(np.uint8)

        # 5) chrome 3D word (rendered in Blender with alpha)
        ci = fi - chrome_f0
        cpath = os.path.join(WORK, "render3d", "chrome", f"{ci:04d}.png")
        if 0 <= ci and os.path.exists(cpath):
            spr = cv2.imread(cpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if spr.shape[:2] != (H, W):
                spr = cv2.resize(spr, (W, H))
            spr = np.roll(spr, CHROME_DY, axis=0)
            if CHROME_DY < 0:
                spr[CHROME_DY:] = 0
            # soft contact shadow so the chrome reads against a bright sky
            sh = np.zeros_like(spr)
            sh[..., 3] = np.roll(cv2.GaussianBlur(spr[..., 3], (0, 0), 16), 14, axis=0) * 0.55
            sh[..., :3] = 10
            hud.draw(sh, 0, 0)
            hud.draw(spr, 0, 0)
            frame = hud.composite(frame)

        # 6) HUD text on top
        vis = max([c.visibility(t) for c in chapters] + [clamp((t - 26.9) / 0.4)])
        if vis > 0:
            frame = (frame.astype(np.float32) * (1 - SHADE * vis)).astype(np.uint8)
        intro.draw(hud, t, fi)
        for c in chapters:
            c.draw(hud, t, fi)
        end.draw(hud, t)
        frame = hud.composite(frame)

        # 7) post
        if post["rgb"] > 0.5:
            frame = rgb_split(frame, post["rgb"])
        frame = flash(frame, post["flash"])
        frame = (frame.astype(np.float32) + GRAIN[fi % len(GRAIN)][..., None]).clip(0, 255).astype(np.uint8)
        enc.stdin.write(np.ascontiguousarray(frame).tobytes())
        if fi % 60 == 0:
            print(f"frame {fi}/{NFRAMES}", flush=True)
    src.close()
    enc.stdin.close()
    enc.wait()


# ---------------------------------------------------------------- 3D map intro
def compose_map(path):
    """Globe (frames 0-60) + terrain (45-180) Blender renders -> 6 s intro clip.
    Adds a star field, denoises the low-sample renders and dives from one into the other."""
    gdir = os.path.join(WORK, "render3d", "globe")
    tdir = os.path.join(WORK, "render3d", "terrain")
    rng = np.random.default_rng(11)
    ns = 1400
    sx, sy = rng.uniform(-1, 1, ns) * W * 0.9, rng.uniform(-1, 1, ns) * H * 0.9
    sb = rng.uniform(0, 1, ns) ** 3 * 255
    sph = rng.uniform(0, 6.28, ns)

    def load(p):
        im = cv2.imread(p)
        im = cv2.fastNlMeansDenoisingColored(im, None, 3, 3, 5, 13)
        return cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)

    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "13", "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE)
    for fi in range(180):
        g = None
        if fi <= 60:
            g = load(os.path.join(gdir, f"{fi:04d}.png"))
            zoom = 1 + 0.35 * fi / 60
            stars = np.zeros((H, W), np.float32)
            for x, y, b, ph in zip(sx, sy, sb, sph):
                px, py = int(W / 2 + x * zoom), int(H / 2 + y * zoom)
                if 0 <= px < W and 0 <= py < H:
                    stars[py, px] = max(stars[py, px], b * (0.75 + 0.25 * math.sin(ph + fi * 0.3)))
            stars = cv2.GaussianBlur(stars, (0, 0), 0.8) * 2.2
            space = (g.max(2) < 14).astype(np.float32)
            space = cv2.GaussianBlur(space, (0, 0), 2)
            g = np.clip(g.astype(np.float32) + (stars * space)[..., None], 0, 255).astype(np.uint8)
        if fi < 45:
            out = g
        else:
            tf = load(os.path.join(tdir, f"{fi:04d}.png"))
            if fi <= 60:
                u = (fi - 45) / 15
                w_ = u * u * (3 - 2 * u)
                A = zoom_blur(g, 1 + 0.9 * u * u, 0.2 * u)
                B = zoom_blur(tf, 1.0, 0.16 * (1 - u))
                out = cv2.addWeighted(A, 1 - w_, B, w_, 0)
                out = flash(out, 0.3 * math.sin(math.pi * u))
            else:
                out = tf
        enc.stdin.write(np.ascontiguousarray(out).tobytes())
    enc.stdin.close()
    enc.wait()


# ---------------------------------------------------------------- audio
def whoosh(dur=0.42, peak=0.72, f0=350, f1=5200, pan=(-0.8, 0.8), seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    f, tt, Z = signal.stft(rng.standard_normal(n), SR, nperseg=1024)
    fc = f0 * (f1 / f0) ** (tt / tt.max())
    mask = np.exp(-0.5 * ((np.log(f[:, None] + 1) - np.log(fc[None, :])) / 0.45) ** 2)
    _, y = signal.istft(Z * mask, SR, nperseg=1024)
    y = y[:n]
    x = np.linspace(0, 1, n)
    y = y * np.where(x < peak, (x / peak) ** 2.5, np.exp(-(x - peak) / 0.05))
    y /= np.abs(y).max() + 1e-9
    pl = np.linspace(pan[0], pan[1], n)
    return np.stack([y * np.sqrt((1 - pl) / 2), y * np.sqrt((1 + pl) / 2)], 1), peak * dur


def boom():
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    fr = 38 + 95 * np.exp(-t / 0.07)
    y = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.28)
    y[:220] += np.random.default_rng(3).standard_normal(220) * np.linspace(1, 0, 220) * 0.6
    y /= np.abs(y).max()
    return np.stack([y, y], 1), 0.0


def render_audio(path):
    sr, music = wavfile.read(os.path.join(WORK, "music", "823.wav"))
    assert sr == SR
    s0 = int(MUSIC_START * SR)
    mix = music.astype(np.float32)[s0:s0 + int(DUR * SR)].copy()
    n = len(mix)
    mix[:int(0.03 * SR)] *= np.linspace(0, 1, int(0.03 * SR))[:, None]
    fo = int(1.3 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5
    sfx = np.zeros_like(mix)

    def place(clip_peak, at, gain):
        clip, pk = clip_peak
        i0 = int((at - pk) * SR)
        a, b = max(0, i0), min(n, i0 + len(clip))
        if b > a:
            sfx[a:b] += clip[a - i0:b - i0] * gain

    for i, (cut, tr) in enumerate(sorted(TRANSITIONS.items())):
        if tr["kind"] == "whip":
            pan = (-0.9, 0.9) if tr["dir"][0] <= 0 else (0.9, -0.9)
            place(whoosh(0.38, 0.75, 500, 6000, pan, seed=i), cut, 0.28)
        else:
            gain = 0.24 if tr.get("amp", 0.5) >= 0.4 else 0.16
            place(whoosh(0.6, 0.85, 250, 4200, (0, 0), seed=i), cut, gain)
    place(whoosh(0.9, 0.8, 200, 3800, (0, 0), seed=91), 1.85, 0.3)            # dive into the terrain
    place(whoosh(0.5, 0.8, 300, 5000, (0.6, -0.6), seed=92), CHROME_AT + 0.5, 0.26)  # chrome word flies in
    for at, g in ((DROP, 0.42), (18.0, 0.3), (28.0, 0.4)):
        place(boom(), at, g)
    out = mix + sfx
    out = np.tanh(out / max(np.abs(out).max(), 1e-6) * 1.15) / np.tanh(1.15) * 0.93
    wavfile.write(path, SR, out.astype(np.float32))


if __name__ == "__main__":
    only = os.environ.get("ONLY", "")
    vid = os.path.join(OUT, "video.mp4")
    aud = os.path.join(OUT, "audio.wav")
    if only in ("", "audio"):
        render_audio(aud)
    if only in ("", "video", "map") and (only == "map" or not os.path.exists(os.path.join(OUT, "map.mp4"))):
        compose_map(os.path.join(OUT, "map.mp4"))
    if only in ("", "video"):
        render_video(vid)
    if only == "":
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-i", aud, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-shortest",
                        "-movflags", "+faststart", os.path.join(OUT, "reel_music.mp4")], check=True)
        print("done")
