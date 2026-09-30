#!/usr/bin/env python3
"""Pre-cut the footage for the engraving plates.

For every shot, the frames the video needs are extracted at the video's own timing (30 fps,
speed ramps baked in) as a two-channel 8-bit raster:
  R = luminance with local contrast (CLAHE) — what the engraving shader hatches,
  G = "signal" mask: saturated orange/red surfaces and small blown highlights (glitter on the water;
      large blown areas such as the sky around the sun are left out) — drawn in the palette's orange.
Each shot becomes app/public/footage/<id>.bin (frames * H * W * 2 bytes, row 0 = top) plus an entry
in app/public/footage/index.json, with per frame the signal's centroid (0..1, y down) and area
fraction ("sig"), which the plates use to put callouts on it.

Usage: python3 make_footage.py <proxy_dir> <repo_root>
  proxies: 1080x1920 60 fps, c1..c9 (see reel/render.py for what each clip is)
"""
import json
import os
import sys

import cv2
import numpy as np

PROXY, ROOT = sys.argv[1], sys.argv[2]
OUT = os.path.join(ROOT, "app", "public", "footage")
os.makedirs(OUT, exist_ok=True)
FPS = 30
W, H = 432, 768

# id: (clip, source in-point, [(output duration, speed), ...])
SHOTS = {
    "road1": ("c6", 0.5, [(2.0, 2.0)]),
    "garni_cols": ("c7", 7.4, [(2.0, 1.0)]),
    "garni_orange": ("c7", 11.2, [(1.0, 1.2)]),
    "garni_bath": ("c7", 40.3, [(1.0, 1.3)]),
    "basalt_arch": ("c1", 7.45, [(1.5, 1.0)]),
    "basalt_walk": ("c1", 0.5, [(1.5, 1.5)]),
    "geghard_cliff": ("c8", 0.0, [(2.0, 1.0)]),
    "geghard_yard": ("c8", 7.2, [(1.0, 1.4)]),
    "geghard_tunnel": ("c8", 15.0, [(1.0, 2.0)]),
    "road2": ("c9", 2.0, [(4.0, 3.0)]),
    "sevan_beach": ("c2", 1.6, [(2.0, 1.3)]),
    "sevan_sun": ("c2", 17.0, [(2.0, 1.0)]),
    "jet_tilt": ("c3", 12.8, [(1.5, 1.5)]),
    "jet_spray": ("c3", 41.0, [(0.5, 2.0), (1.5, 0.5), (0.5, 2.0)]),
    "vank_church": ("c4", 11.35, [(2.0, 0.75)]),
    "vank_zoom": ("c2", 12.4, [(1.0, 1.2)]),
    "vank_lake": ("c4", 4.5, [(1.0, 1.5)]),
    "pen_clouds": ("c5", 3.2, [(2.0, 1.25)]),
    "pen_ridge": ("c5", 15.4, [(2.0, 1.0)]),
    "pen_walk": ("c5", 17.6, [(6.0, 0.28)]),
}

clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 10))


def src_times(src_in, ramp):
    ts, t = [], src_in
    for dur, sp in ramp:
        for i in range(int(round(dur * FPS))):
            ts.append(t + i / FPS * sp)
        t += dur * sp
    return ts


def encode(bgr):
    small = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    lum = clahe.apply(lab[..., 0])
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
    orange = np.exp(-((((h + 20) % 360) - 40) / 22) ** 2) * np.clip((s - 0.45) / 0.3, 0, 1) * np.clip((v - 0.3) / 0.3, 0, 1)
    glint = np.clip((v - 0.965) / 0.03, 0, 1) * np.clip((0.25 - s) / 0.2, 0, 1)
    # keep glints smaller than ~9 px (a white top-hat): the sun's disk and blown sky are not signal
    glint = cv2.morphologyEx(glint.astype(np.float32), cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    sig = np.clip(np.maximum(orange, glint), 0, 1)
    sig = cv2.GaussianBlur(sig, (0, 0), 1.2)
    return np.dstack([lum, (sig * 255).astype(np.uint8)])


index = {}
for sid, (clip, src_in, ramp) in SHOTS.items():
    cap = cv2.VideoCapture(os.path.join(PROXY, clip + ".mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ts = src_times(src_in, ramp)
    want = [int(round(t * fps)) for t in ts]
    cap.set(cv2.CAP_PROP_POS_FRAMES, want[0])
    pos, frame, frames = want[0] - 1, None, []
    for w_ in want:
        while pos < w_:
            ok, f = cap.read()
            if not ok:
                break
            frame, pos = f, pos + 1
        frames.append(encode(frame))
    cap.release()
    arr = np.stack(frames).astype(np.uint8)
    arr.tofile(os.path.join(OUT, sid + ".bin"))
    sig = []
    ys, xs = np.mgrid[0:H, 0:W]
    for f in arr:
        m = (f[..., 1] > 128).astype(np.float32)
        a = m.sum()
        sig.append([round(float((m * xs).sum() / a / W), 4), round(float((m * ys).sum() / a / H), 4), round(float(a / (W * H)), 5)]
                   if a > 0 else [0.5, 0.5, 0.0])
    index[sid] = {"frames": len(frames), "w": W, "h": H, "fps": FPS, "clip": clip, "in": src_in, "sig": sig}
    print(sid, arr.shape, flush=True)
json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
