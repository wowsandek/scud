#!/usr/bin/env python3
"""Beat-synced travel reel renderer (1080x1920, 30 fps).

Reads 1080x1920 proxies of the source clips, cuts them on a 120 BPM grid,
adds transitions (zoom punch, whip pan, spin, glitch, light leak), beat
pulses, animated location labels and an end card, then encodes the video
and muxes it with music + synthesized whoosh/boom SFX.

Usage: python3 render.py <work_dir>
  work_dir must contain proxy/c1..c5.mp4, music/823.wav, fonts/*.ttf
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

WORK = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(WORK, "out")
os.makedirs(OUT, exist_ok=True)

W, H, FPS = 1080, 1920, 30
BEAT = 0.5
DUR = 27.5
NFRAMES = int(DUR * FPS)
MUSIC_START = 50.1336  # track time of reel t=0 (bar start, drop lands at t=4.0)
SR = 44100

ACCENT = (255, 154, 31)  # apricot orange (RGB)
DARK = (17, 17, 17)
WHITE = (255, 255, 255)

GRADE = ("eq=contrast=1.08:saturation=1.25:gamma=0.97,"
         "colorbalance=rs=-0.02:bs=0.03:rh=0.03:bh=-0.02,"
         "unsharp=5:5:0.5")

# ---------------------------------------------------------------- edit list
# clip, source in-point, reel start, reel duration, speed ramp [(dur, speed)]
SEGMENTS = [
    # intro (build-up, title on screen)
    ("c4", 0.30, 0.0, 2.0, [(2.0, 1.0)]),
    ("c3", 23.0, 2.0, 2.0, [(1.5, 1.0), (0.5, 1.6)]),
    # 01 Garni  (drop at 4.0)
    ("c1", 0.50, 4.0, 1.0, [(1.0, 1.5)]),
    ("c1", 4.60, 5.0, 1.0, [(1.0, 1.5)]),
    ("c1", 7.60, 6.0, 1.0, [(1.0, 1.0)]),
    ("c1", 12.6, 7.0, 1.0, [(1.0, 1.2)]),
    # 02 Sevan beach
    ("c2", 1.60, 8.0, 1.5, [(1.5, 1.3)]),
    ("c2", 17.0, 9.5, 1.5, [(1.5, 1.0)]),
    # 03 jet ski
    ("c3", 12.8, 11.0, 1.5, [(1.5, 1.5)]),
    ("c3", 26.6, 12.5, 1.0, [(1.0, 1.5)]),
    ("c3", 41.0, 13.5, 2.5, [(0.5, 2.0), (1.5, 0.5), (0.5, 2.0)]),
    # 04 Sevanavank
    ("c2", 12.2, 16.0, 1.5, [(1.5, 1.0)]),
    ("c4", 11.3, 17.5, 1.5, [(1.5, 1.0)]),
    ("c4", 4.50, 19.0, 1.0, [(1.0, 1.5)]),
    # finale panoramas
    ("c5", 3.20, 20.0, 2.0, [(2.0, 1.25)]),
    ("c5", 15.4, 22.0, 2.5, [(2.5, 1.0)]),
    # end card background (slow-mo)
    ("c5", 17.95, 24.5, 3.0, [(3.0, 0.45)]),
]

# transition at each cut time
TRANSITIONS = {
    2.0: ("zoomsoft",),
    4.0: ("zoom", True),        # the drop: zoom punch + flash
    5.0: ("whip", (0, -1)),
    6.0: ("whip", (-1, 0)),
    7.0: ("zoom", False),
    8.0: ("spin",),
    9.5: ("whip", (-1, 0)),
    11.0: ("glitch",),
    12.5: ("whip", (1, 0)),
    13.5: ("zoom", False),
    16.0: ("zoom", True),
    17.5: ("whip", (0, -1)),
    19.0: ("whip", (-1, 0)),
    20.0: ("leak",),
    22.0: ("whip", (-1, 0)),
    24.5: ("glitch",),
}

LABELS = [
    # start, end, number, name, subtitle
    (4.0, 8.0, "01", "ГАРНИ", "Симфония камней"),
    (8.0, 11.0, "02", "СЕВАН", "озеро на высоте 1900 м"),
    (11.0, 16.0, "03", "ГИДРОЦИКЛ", "по волнам Севана"),
    (16.0, 20.0, "04", "СЕВАНАВАНК", "монастырь IX века"),
]


# ---------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out_cubic(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3


def ease_in_cubic(p):
    p = clamp(p)
    return p ** 3


def ease_out_back(p, s=1.9):
    p = clamp(p) - 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def font(name, size, weight=900):
    f = ImageFont.truetype(os.path.join(WORK, "fonts", name), size)
    f.set_variation_by_axes([weight])
    return f


def text_sprite(text, fnt, fill=WHITE, shadow=0.55, tracking=0, trim_y=True):
    """Render text to a BGRA uint8 sprite with a soft drop shadow."""
    widths = [fnt.getlength(ch) for ch in text]
    tw = int(sum(widths) + tracking * max(0, len(text) - 1))
    asc, desc = fnt.getmetrics()
    pad = 40
    w, h = tw + 2 * pad, asc + desc + 2 * pad
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    x = pad
    for ch, cw in zip(text, widths):
        d.text((x, pad), ch, font=fnt, fill=255)
        x += cw + tracking
    return compose_sprite(mask, fill, shadow, trim_y)


def compose_sprite(mask, fill, shadow, trim_y=True):
    w, h = mask.size
    m = np.asarray(mask, np.float32) / 255
    if shadow > 0:
        sh = np.asarray(mask.filter(ImageFilter.GaussianBlur(10)), np.float32) / 255
        sh = np.roll(sh, 6, axis=0) * shadow
    else:
        sh = np.zeros_like(m)
    a = m + sh * (1 - m)
    col = np.zeros((h, w, 3), np.float32)
    rgb = np.array(fill[::-1], np.float32)  # to BGR
    with np.errstate(invalid="ignore", divide="ignore"):
        col[:] = rgb * (m / np.maximum(a, 1e-6))[..., None]
    out = np.dstack([col, a * 255]).clip(0, 255).astype(np.uint8)
    return trim(out, trim_y=trim_y)


def trim(spr, keep=2, trim_y=True):
    ys, xs = np.nonzero(spr[..., 3] > 0)
    if len(xs) == 0:
        return spr
    y0, y1 = max(0, ys.min() - keep), min(spr.shape[0], ys.max() + keep + 1)
    if not trim_y:
        y0, y1 = 0, spr.shape[0]
    x0, x1 = max(0, xs.min() - keep), min(spr.shape[1], xs.max() + keep + 1)
    return spr[y0:y1, x0:x1].copy()


def rounded_rect_sprite(w, h, r, fill):
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=255)
    m = np.asarray(img, np.float32)
    col = np.zeros((h, w, 3), np.float32)
    col[:] = np.array(fill[::-1], np.float32)
    return np.dstack([col, m]).astype(np.uint8)


def pin_sprite(size, fill):
    s = size * 4
    img = Image.new("L", (s, int(s * 1.35)), 0)
    d = ImageDraw.Draw(img)
    r = s // 2
    d.ellipse([0, 0, s - 1, s - 1], fill=255)
    d.polygon([(s * 0.12, r * 1.35), (s * 0.88, r * 1.35), (s / 2, s * 1.33)], fill=255)
    d.ellipse([r * 0.55, r * 0.55, s - r * 0.55, s - r * 0.55], fill=0)
    img = img.resize((size, int(size * 1.35)), Image.LANCZOS)
    return compose_sprite(img, fill, 0.45)


def blit(dst, spr, x, y, alpha=1.0, scale=1.0, anchor=(0.5, 0.5), clip=None):
    if alpha <= 0.004 or spr is None:
        return
    if abs(scale - 1.0) > 1e-3:
        if scale <= 0.02:
            return
        interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        spr = cv2.resize(spr, None, fx=scale, fy=scale, interpolation=interp)
    h, w = spr.shape[:2]
    x0 = int(round(x - anchor[0] * w))
    y0 = int(round(y - anchor[1] * h))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if clip:
        X0, Y0 = max(X0, int(clip[0])), max(Y0, int(clip[1]))
        X1, Y1 = min(X1, int(clip[2])), min(Y1, int(clip[3]))
    if X1 <= X0 or Y1 <= Y0:
        return
    s = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].astype(np.float32)
    a = s[..., 3:4] / 255.0 * alpha
    d = dst[Y0:Y1, X0:X1, :3].astype(np.float32)
    dst[Y0:Y1, X0:X1, :3] = (d * (1 - a) + s[..., :3] * a).astype(np.uint8)


def affine(img, scale=1.0, angle=0.0, dx=0.0, dy=0.0, border=cv2.BORDER_REFLECT101):
    if abs(scale - 1) < 1e-4 and abs(angle) < 1e-4 and abs(dx) < 0.5 and abs(dy) < 0.5:
        return img
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border)


def zoom_blur(img, scale, amount, n=6):
    """Radial blur: average copies scaled between scale and scale*(1+amount)."""
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


def glitch(img, strength, seed):
    rng = np.random.default_rng(seed)
    out = img.copy()
    for _ in range(int(6 + 10 * strength)):
        y = rng.integers(0, H - 20)
        hgt = int(rng.integers(8, 90))
        sh = int(rng.normal(0, 90 * strength))
        out[y:y + hgt] = np.roll(out[y:y + hgt], sh, axis=1)
    return rgb_split(out, 18 * strength * (1 if rng.random() > 0.5 else -1))


def flash(img, a, color=(255, 255, 255)):
    if a <= 0.004:
        return img
    c = np.array(color, np.float32)
    return (img.astype(np.float32) * (1 - a) + c * a).astype(np.uint8)


# ---------------------------------------------------------------- sources
class SegmentReader:
    def __init__(self, clip, src_in):
        self.src_in = src_in
        path = os.path.join(WORK, "proxy", clip + ".mp4")
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{src_in:.3f}", "-i", path, "-vf", GRADE,
             "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
        self.idx = -1
        self.frame = None

    def get(self, src_t):
        want = max(0, int(round((src_t - self.src_in) * 60)))
        while self.idx < want:
            buf = self.proc.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break  # clip ended: hold the last frame
            self.frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            self.idx += 1
        return self.frame.copy()

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


# ---------------------------------------------------------------- graphics
F_HEAD = "Unbounded.ttf"
F_BODY = "Montserrat.ttf"


def fit_font(name, text, max_w, size, weight=900):
    while size > 20:
        f = font(name, size, weight)
        if f.getlength(text) <= max_w:
            return f
        size -= 4
    return font(name, size, weight)


class Label:
    def __init__(self, start, end, num, name, sub):
        self.start, self.end = start, end
        self.num = text_sprite(num, font(F_BODY, 40, 800), ACCENT, shadow=0.5)
        self.name = text_sprite(name, fit_font(F_HEAD, name, 820, 112), WHITE, shadow=0.6)
        self.pin = pin_sprite(46, ACCENT)
        st = text_sprite(sub, font(F_BODY, 40, 700), DARK, shadow=0)
        pw, ph = st.shape[1] + 56, st.shape[0] + 30
        pill = rounded_rect_sprite(pw, ph, ph // 2, ACCENT)
        blit(pill, st, pw / 2, ph / 2)
        self.pill = pill

    def draw(self, img, t):
        lt = t - self.start
        if lt < 0 or t >= self.end:
            return
        out_p = clamp((t - (self.end - 0.16)) / 0.16)
        a = 1 - out_p
        dx = -90 * ease_in_cubic(out_p)
        x = 72 + dx
        base_y = 1330
        # number + pin
        pp = ease_out_back((lt - 0.08) / 0.25)
        if lt > 0.08:
            blit(img, self.pin, x + 23, base_y - 175, a, max(0.02, pp), (0.5, 1.0))
            na = clamp((lt - 0.14) / 0.12) * a
            blit(img, self.num, x + 62 - 30 * (1 - ease_out_cubic((lt - 0.14) / 0.25)),
                 base_y - 196, na, 1.0, (0.0, 0.5))
        # name: rises out of a mask
        p = ease_out_back((lt - 0.1) / 0.35, 1.4)
        nh = self.name.shape[0]
        y_top = base_y - nh
        blit(img, self.name, x, y_top + (1 - p) * nh * 1.1, a, 1.0, (0.0, 0.0),
             clip=(0, y_top - 40, W, base_y + 8))
        # pill wipes in from the left
        wp = ease_out_cubic((lt - 0.3) / 0.3)
        if wp > 0:
            pw = self.pill.shape[1]
            blit(img, self.pill, x, base_y + 28, a, 1.0, (0.0, 0.0),
                 clip=(x, 0, x + pw * wp, H))


def bottom_shade(img, a):
    if a <= 0:
        return img
    g = SHADE * a
    return (img.astype(np.float32) * (1 - g)).astype(np.uint8)


def build_shade():
    y = np.linspace(0, 1, H, dtype=np.float32)
    g = np.clip((y - 0.45) / 0.3, 0, 1) ** 1.3 * 0.62
    g = g * np.clip((0.92 - y) / 0.12, 0.35, 1)
    return np.repeat(g[:, None, None], W, axis=1)


def build_vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return (1 - 0.32 * np.clip(r / 1.3, 0, 1) ** 2.4)[..., None]


SHADE = build_shade()
VIGN = build_vignette()


class Intro:
    def __init__(self):
        f1 = fit_font(F_HEAD, "ОДИН ДЕНЬ", 900, 118)
        self.l1 = [text_sprite(ch, f1, WHITE, 0.6, trim_y=False) if ch != " " else None
                   for ch in "ОДИН ДЕНЬ"]
        adv = [f1.getlength(ch) for ch in "ОДИН ДЕНЬ"]
        total = sum(adv)
        self.l1x = []
        x = W / 2 - total / 2
        for a in adv:
            self.l1x.append(x + a / 2)
            x += a
        f2 = fit_font(F_HEAD, "В АРМЕНИИ", 820, 104)
        self.l2 = text_sprite("В АРМЕНИИ", f2, DARK, 0)
        bw, bh = self.l2.shape[1] + 64, self.l2.shape[0] + 44
        self.box = rounded_rect_sprite(bw, bh, 18, ACCENT)
        self.top = text_sprite("27 · 09", font(F_BODY, 46, 700), WHITE, 0.5, tracking=6)

    def draw(self, img, t):
        if t >= 4.12:
            return
        push = 1.0 + 0.9 * ease_in_cubic((t - 3.75) / 0.37)  # flies into camera at the drop
        fade = 1 - clamp((t - 3.95) / 0.17)
        build = 1 + 0.04 * clamp((t - 2.0) / 1.75)
        s = push * build
        cy1, cy2 = 800, 960

        def pos(x, y):
            return W / 2 + (x - W / 2) * s, H / 2 + (y - H / 2) * s

        for i, (spr, cx) in enumerate(zip(self.l1, self.l1x)):
            if spr is None:
                continue
            p = ease_out_back((t - 0.05 - i * 0.045) / 0.28, 2.2)
            if p <= 0:
                continue
            x, y = pos(cx, cy1)
            blit(img, spr, x, y - (1 - p) * 60, fade * clamp(p * 2), s * max(0.02, p))
        bp = ease_out_cubic((t - 0.95) / 0.28)
        if bp > 0:
            x, y = pos(W / 2, cy2)
            bw = self.box.shape[1] * s
            blit(img, self.box, x, y, fade, s, clip=(x - bw / 2, 0, x - bw / 2 + bw * bp, H))
            tp = clamp((t - 1.1) / 0.2)
            blit(img, self.l2, x, y + (1 - tp) * 25, fade * tp, s,
                 clip=(x - bw / 2, 0, x - bw / 2 + bw * bp, H))
        ta = clamp((t - 1.6) / 0.3) * fade
        x, y = pos(W / 2, 660)
        blit(img, self.top, x, y, ta, s)


class EndCard:
    def __init__(self):
        self.title = text_sprite("АРМЕНИЯ", fit_font(F_HEAD, "АРМЕНИЯ", 940, 150), WHITE, 0.6)
        self.stops = [("Гарни", 150), ("Севан", 500), ("Севанаванк", 850)]
        fs = font(F_BODY, 44, 700)
        self.names = [text_sprite(n, fs, WHITE, 0.6) for n, _ in self.stops]
        self.dot = pin_sprite(40, ACCENT)
        cta = text_sprite("сохрани, чтобы не потерять", font(F_BODY, 40, 700), DARK, 0)
        pw, ph = cta.shape[1] + 60, cta.shape[0] + 34
        pill = rounded_rect_sprite(pw, ph, ph // 2, WHITE)
        blit(pill, cta, pw / 2, ph / 2)
        self.cta = pill

    def draw(self, img, t):
        lt = t - 24.5
        if lt < 0:
            return img
        dark = 0.42 * clamp(lt / 0.25)
        img = (img.astype(np.float32) * (1 - dark)).astype(np.uint8)
        hit = 1 + 0.12 * math.exp(-max(0, t - 26.0) / 0.12) if t >= 26.0 else 1.0
        p = ease_out_back(lt / 0.35, 1.6)
        if p > 0:
            blit(img, self.title, W / 2, 800, clamp(p * 1.5), (0.85 + 0.15 * p) * hit)
        # route line draws between 25.0 and 26.0
        lp = ease_out_cubic((t - 25.0) / 1.0)
        y = 1010
        x0, x1 = self.stops[0][1], self.stops[-1][1]
        if lp > 0:
            xe = int(x0 + (x1 - x0) * lp)
            cv2.line(img, (x0, y), (xe, y), (255, 255, 255), 5, cv2.LINE_AA)
            for i, (n, x) in enumerate(self.stops):
                reach = (x - x0) / (x1 - x0)
                t_reach = 25.0 + 1 - (1 - reach) ** (1 / 3)  # inverse of ease_out_cubic
                dp = ease_out_back((t - t_reach + 0.05) / 0.3, 2.0)
                if dp > 0:
                    blit(img, self.dot, x, y + 8, 1.0, max(0.02, dp), (0.5, 1.0))
                    blit(img, self.names[i], x, y + 58, clamp(dp), 1.0)
        cp = ease_out_back((t - 26.15) / 0.35, 1.5)
        if cp > 0:
            blit(img, self.cta, W / 2, 1245 + (1 - cp) * 40, clamp(cp * 1.4), 1.0)
        return img


def light_leak(t_rel):
    """Warm moving light leak (BGR float, 0..1 intensity)."""
    yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    cx = W * (0.1 + 0.9 * t_rel)
    cy = H * (0.35 - 0.15 * t_rel)
    g = np.exp(-(((xx - cx) / (W * 0.55)) ** 2 + ((yy - cy) / (H * 0.45)) ** 2))
    g = cv2.resize(g, (W, H))
    col = np.array([60, 150, 255], np.float32) / 255  # BGR warm orange
    return g[..., None] * col


# ---------------------------------------------------------------- render
def transition_fx(frame, t, fi):
    """Geometric / blur part of transitions. Returns frame, post-fx dict."""
    post = {"flash": 0.0, "rgb": 0.0, "glitch": 0.0, "shake": 0.0, "leak": None}
    for cut, spec in TRANSITIONS.items():
        cf = int(round(cut * FPS))
        k = fi - cf  # <0: outgoing clip, >=0: incoming clip
        kind = spec[0]
        if kind in ("zoom", "zoomsoft"):
            big = kind == "zoom"
            n_out, n_in = (4, 5) if big else (3, 3)
            amp = 0.55 if big else 0.25
            if -n_out <= k < 0:
                p = (n_out + k + 1) / n_out
                frame = zoom_blur(frame, 1 + amp * p ** 2, 0.18 * p)
                post["rgb"] = max(post["rgb"], 10 * p)
            elif 0 <= k < n_in:
                q = 1 - k / n_in
                frame = zoom_blur(frame, 1 + amp * 0.8 * q ** 2, 0.14 * q)
                post["rgb"] = max(post["rgb"], 12 * q)
                if big and len(spec) > 1 and spec[1]:
                    post["flash"] = max(post["flash"], [0.85, 0.5, 0.25, 0.1, 0][min(k, 4)])
                    post["shake"] = max(post["shake"], 1.0)
                elif big:
                    post["shake"] = max(post["shake"], 0.5)
        elif kind == "whip":
            dx, dy = spec[1]
            n = 3
            L = W if dx else H
            if -n <= k < 0:
                p = (n + k + 1) / n
                off = L * 0.45 * p ** 2
                fr = affine(frame, 1.0, 0, dx * off, dy * off)
                frame = motion_blur(fr, dx * 260 * p, dy * 260 * p)
            elif 0 <= k < n:
                q = 1 - (k + 1) / (n + 1)
                off = -L * 0.45 * q ** 2
                fr = affine(frame, 1.0, 0, dx * off, dy * off)
                frame = motion_blur(fr, dx * 260 * q, dy * 260 * q)
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
        elif kind == "glitch":
            if -2 <= k < 4:
                post["glitch"] = max(post["glitch"], 1.0 - abs(k + 0.5) / 4.5)
                if k in (0, 1):
                    post["flash"] = max(post["flash"], 0.35 if k == 0 else 0.12)
                post["shake"] = max(post["shake"], 0.6)
        elif kind == "leak":
            if 0 <= k < 16:
                q = k / 16
                frame = affine(frame, 1 + 0.22 * (1 - ease_out_cubic(q)))
                post["leak"] = (q, 0.9 * (1 - q) ** 1.5)
                post["flash"] = max(post["flash"], [0.45, 0.2, 0.08][k] if k < 3 else 0)
            elif -3 <= k < 0:
                post["leak"] = (0.0, 0.35 * (4 + k) / 3)
    return frame, post


def beat_pulse(t):
    if t < 4.0 or t >= 24.5:
        return 1.0
    b = math.floor(t / BEAT + 1e-6) * BEAT
    dt = t - b
    strong = abs((b / 2.0) - round(b / 2.0)) < 1e-6
    return 1 + (0.045 if strong else 0.025) * math.exp(-dt / 0.11)


def shake_offsets(fi, strength):
    rng = np.random.default_rng(fi * 7919)
    return rng.normal(0, 14 * strength), rng.normal(0, 14 * strength), rng.normal(0, 0.7 * strength)


def render_video(path):
    intro, end = Intro(), EndCard()
    labels = [Label(*l) for l in LABELS]
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
         "-maxrate", "16M", "-bufsize", "32M", "-pix_fmt", "yuv420p", "-profile:v", "high",
         "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    reader, cur = None, None
    shake_left = 0.0
    for fi in range(NFRAMES):
        t = fi / FPS
        seg = next(s for s in SEGMENTS if s[2] <= t + 1e-9 < s[2] + s[3])
        if seg is not cur:
            if reader:
                reader.close()
            reader, cur = SegmentReader(seg[0], seg[1]), seg
        frame = reader.get(src_time(seg, t - seg[2]))

        frame, post = transition_fx(frame, t, fi)
        # end-card hit on the last big beat
        if 26.0 <= t < 26.2:
            post["flash"] = max(post["flash"], 0.6 * (1 - (t - 26.0) / 0.2))
            post["shake"] = max(post["shake"], 1.0)
        shake_left = max(shake_left * 0.72, post["shake"])
        sx, sy, sa = shake_offsets(fi, shake_left) if shake_left > 0.05 else (0, 0, 0)
        frame = affine(frame, beat_pulse(t) * (1.02 if shake_left > 0.05 else 1.0), sa, sx, sy)

        frame = (frame.astype(np.float32) * VIGN).astype(np.uint8)
        if post["leak"] is not None:
            q, a = post["leak"]
            lk = light_leak(q) * a
            f = frame.astype(np.float32) / 255
            frame = ((1 - (1 - f) * (1 - lk)) * 255).astype(np.uint8)  # screen blend

        # text layers
        la = 0.0
        for lb in labels:
            if lb.start <= t < lb.end:
                la = clamp((t - lb.start - 0.05) / 0.25) * (1 - clamp((t - (lb.end - 0.16)) / 0.16))
        if t < 4.12:
            frame = (frame.astype(np.float32) * (1 - 0.22 * (1 - clamp((t - 3.9) / 0.2)))).astype(np.uint8)
        frame = bottom_shade(frame, la)
        intro.draw(frame, t)
        for lb in labels:
            lb.draw(frame, t)
        frame = end.draw(frame, t)

        if post["glitch"] > 0:
            frame = glitch(frame, post["glitch"], fi)
        if post["rgb"] > 0.5:
            frame = rgb_split(frame, post["rgb"])
        frame = flash(frame, post["flash"])
        enc.stdin.write(np.ascontiguousarray(frame).tobytes())
        if fi % 60 == 0:
            print(f"frame {fi}/{NFRAMES}", flush=True)
    reader.close()
    enc.stdin.close()
    enc.wait()


# ---------------------------------------------------------------- audio
def whoosh(dur=0.42, peak=0.72, f0=350, f1=5200, pan=(-0.8, 0.8), seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    f, tt, Z = signal.stft(noise, SR, nperseg=1024)
    fc = f0 * (f1 / f0) ** (tt / tt.max())
    mask = np.exp(-0.5 * ((np.log(f[:, None] + 1) - np.log(fc[None, :])) / 0.45) ** 2)
    _, y = signal.istft(Z * mask, SR, nperseg=1024)
    y = y[:n]
    x = np.linspace(0, 1, n)
    env = np.where(x < peak, (x / peak) ** 2.5, np.exp(-(x - peak) / 0.05))
    y = y * env
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


def glitch_sfx(seed):
    rng = np.random.default_rng(seed)
    n = int(0.22 * SR)
    y = rng.standard_normal(n)
    gate = (np.sin(2 * np.pi * 34 * np.arange(n) / SR) > 0).astype(float)
    y = np.round(y * gate * 6) / 6
    b, a = signal.butter(2, [900, 6000], "bandpass", fs=SR)
    y = signal.lfilter(b, a, y) * np.exp(-np.arange(n) / SR / 0.1)
    y /= np.abs(y).max()
    return np.stack([y, y], 1), 0.02


def render_audio(path):
    sr, music = wavfile.read(os.path.join(WORK, "music", "823.wav"))
    assert sr == SR
    music = music.astype(np.float32)
    s0 = int(MUSIC_START * SR)
    mix = music[s0:s0 + int(DUR * SR)].copy()
    n = len(mix)
    fade_in = int(0.03 * SR)
    mix[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    fo = int(1.3 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5

    sfx = np.zeros_like(mix)

    def place(clip_peak, at, gain):
        clip, pk = clip_peak
        i0 = int((at - pk) * SR)
        a, b = max(0, i0), min(n, i0 + len(clip))
        if b > a:
            sfx[a:b] += clip[a - i0:b - i0] * gain

    for i, (cut, spec) in enumerate(sorted(TRANSITIONS.items())):
        kind = spec[0]
        if kind == "whip":
            dx = spec[1][0]
            pan = (-0.9, 0.9) if dx <= 0 else (0.9, -0.9)
            place(whoosh(0.38, 0.75, 500, 6000, pan, seed=i), cut, 0.30)
        elif kind in ("zoom", "zoomsoft", "spin", "leak"):
            place(whoosh(0.6, 0.85, 250, 4200, (0, 0), seed=i), cut, 0.26 if kind != "zoomsoft" else 0.18)
        elif kind == "glitch":
            place(glitch_sfx(i), cut, 0.16)
    for at in (4.0, 16.0, 26.0):
        place(boom(), at, 0.42)

    out = mix + sfx
    peak = np.abs(out).max()
    out = np.tanh(out / max(peak, 1e-6) * 1.15) / np.tanh(1.15) * 0.93
    wavfile.write(path, SR, out.astype(np.float32))


if __name__ == "__main__":
    only = os.environ.get("ONLY", "")
    vid = os.path.join(OUT, "video.mp4")
    aud = os.path.join(OUT, "audio.wav")
    if only in ("", "audio"):
        render_audio(aud)
    if only in ("", "video"):
        render_video(vid)
    if only == "":
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-i", aud, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-shortest",
                        "-movflags", "+faststart", os.path.join(OUT, "reel_music.mp4")], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-c", "copy", "-an",
                        "-movflags", "+faststart", os.path.join(OUT, "reel_no_music.mp4")], check=True)
        print("done")
