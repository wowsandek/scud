#!/usr/bin/env python3
"""Cut the music segment for the video and write data/audio.json in the engine's schema.

Track: "Swish Swed" (Mixkit, free licence), 120 BPM. The segment starts on a kick/downbeat
(track 18.524 s) so that the drop (track 40.524 s) lands on beat 44 of the video (22.0 s),
the arrival at Lake Sevan, and runs 92 beats (23 bars). Grid measured on the kick attacks
(steepest rise of the <120 Hz envelope): drift < 5 ms over the segment.

Usage: python3 make_audio.py <track.wav> <repo_root>
"""
import json
import os
import sys

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

SRC, ROOT = sys.argv[1], sys.argv[2]
PERIOD = 0.5
M0 = 18.524            # track time of the segment's first beat (a kick on a downbeat)
NBEATS = 92
DUR = NBEATS * PERIOD
SR = 44100
FPS = 100

sr, x = wavfile.read(SRC)
assert sr == SR
x = x.astype(np.float32)
seg = x[int(M0 * SR): int((M0 + DUR) * SR)].copy()
n = len(seg)
fi = int(0.01 * SR)
seg[:fi] *= np.linspace(0, 1, fi)[:, None]
fo = int(1.4 * SR)
seg[-fo:] *= (np.linspace(1, 0, fo) ** 1.6)[:, None]
os.makedirs(os.path.join(ROOT, "audio"), exist_ok=True)
wavfile.write(os.path.join(ROOT, "audio", "track.wav"), SR, seg)

mono = seg.mean(1)
beats = [round(i * PERIOD, 4) for i in range(NBEATS)]
downbeats = beats[::4]


def band(lo, hi):
    sos = signal.butter(4, [lo, hi] if lo else hi, "bandpass" if lo else "lowpass", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, mono)


def env(sig):
    hop = SR // FPS
    frames = len(sig) // hop
    r = np.sqrt(np.array([np.mean(sig[i * hop:(i + 1) * hop] ** 2) for i in range(frames)]) + 1e-12)
    out = np.zeros_like(r)
    a_att, a_rel = np.exp(-1 / (0.01 * FPS)), np.exp(-1 / (0.09 * FPS))
    v = 0.0
    for i, s in enumerate(r):
        a = a_att if s > v else a_rel
        v = a * v + (1 - a) * s
        out[i] = v
    return np.clip(out / (np.percentile(out, 99) + 1e-9), 0, 1)


low, mid, high = band(None, 150), band(150, 2000), band(4000, 16000)
e = {"rms": env(mono), "low": env(low), "mid": env(mid), "high": env(high)}
e["bass"], e["other"], e["drums"] = e["low"], e["mid"], np.maximum(e["low"], e["high"])
e["vocal"] = np.zeros_like(e["rms"])


def onsets(sig, delta):
    o = librosa.onset.onset_strength(y=sig.astype(np.float32), sr=SR, hop_length=256)
    t = librosa.onset.onset_detect(onset_envelope=o, sr=SR, hop_length=256, units="time", delta=delta)
    frames = librosa.time_to_frames(t, sr=SR, hop_length=256)
    s = o[frames] / (o.max() + 1e-9)
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(t, s) if b > 0.08]


kick = onsets(band(None, 120), 0.2)
snare = onsets(band(1500, 5000), 0.25)
hat = onsets(band(7000, 16000), 0.2)
DROP = 44 * PERIOD
data = {
    "duration": round(DUR, 3),
    "bpm": round(60 / PERIOD, 3),
    "beat_period": PERIOD,
    "time_signature": 4,
    "beats": beats,
    "downbeats": downbeats,
    "sections": [
        {"name": "groove", "start": 0.0, "end": round(12 * PERIOD, 3)},
        {"name": "breakdown", "start": round(12 * PERIOD, 3), "end": round(28 * PERIOD, 3)},
        {"name": "riser", "start": round(28 * PERIOD, 3), "end": round(DROP, 3)},
        {"name": "drop", "start": round(DROP, 3), "end": round(DUR, 3)},
    ],
    "fps": FPS,
    **{k: [round(float(v), 3) for v in arr] for k, arr in e.items()},
    "onsets": {"kick": kick, "snare": snare, "hat": hat, "vocal": []},
    "notes": ("Swish Swed (Mixkit) from track time %.3f s, %d beats at %.3f BPM. Beat 0 = segment start (a "
              "downbeat); the drop is beat 44 (%.3f s). Envelopes: 100 fps RMS, 10/90 ms attack/release, "
              "divided by the 99th percentile. drums = max(low, high), bass = low, other = mid, no vocal."
              % (M0, NBEATS, 60 / PERIOD, DROP)),
}
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
json.dump(data, open(os.path.join(ROOT, "data", "audio.json"), "w"))
json.dump({"lines": []}, open(os.path.join(ROOT, "data", "lyrics.json"), "w"))
print(f"segment {M0:.3f}-{M0 + DUR:.3f}, dur {DUR:.3f}, drop {DROP:.3f}; kicks {len(kick)} snares {len(snare)} hats {len(hat)}")
