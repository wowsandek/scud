"""2.5D "3D photo" camera moves on a still frame.

Depth Anything V2 (small, ONNX) gives relative inverse depth; the sky (from the
SegFormer mask) is pushed to infinity. A virtual camera move is applied by
backward-warping every output pixel with a displacement proportional to its
disparity, so near objects slide and grow faster than far ones.
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class DepthEstimator:
    def __init__(self, model_path, cache_dir):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def disparity(self, img, key, sky=None):
        """Near = 1, far/sky = 0, float32 HxW, edge-aligned to the image."""
        path = os.path.join(self.cache_dir, key.replace(":", "_").replace("/", "_") + ".npy")
        if os.path.exists(path):
            return np.load(path)
        h, w = img.shape[:2]
        x = cv2.resize(img, (518, 924), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
        d = self.sess.run(None, {"pixel_values": x})[0][0]
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
        lo, hi = np.percentile(d, 2), np.percentile(d, 99.5)
        d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        if sky is not None:
            d = d * (1 - sky)
        d = cv2.ximgproc.guidedFilter(guide=img, src=d, radius=8, eps=1e-3)
        # fatten foreground a little so its edges do not smear into the background
        d = np.maximum(d, cv2.dilate(d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) - 0.02)
        d = cv2.GaussianBlur(d, (0, 0), 1.5).clip(0, 1).astype(np.float32)
        np.save(path, d)
        return d


def warp(img, disp, zoom=1.0, kz=0.0, tx=0.0, ty=0.0, iters=4, extra=()):
    """Render a virtual camera move.

    Forward model for a source pixel p with disparity d:
        q = c + (p - c) * zoom * (1 + kz * d) + (tx, ty) * d
    solved backwards per output pixel by fixed-point iteration.
    `extra` are more HxW(xC) arrays warped with the same field (e.g. masks).
    """
    h, w = disp.shape
    cx, cy = w / 2, h / 2
    qx, qy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    px, py = qx.copy(), qy.copy()
    for _ in range(iters):
        d = cv2.remap(disp, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        s = zoom * (1 + kz * d)
        px = cx + (qx - cx - tx * d) / s
        py = cy + (qy - cy - ty * d) / s
    out = cv2.remap(img, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    rest = [cv2.remap(e, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101) for e in extra]
    return (out, *rest) if extra else out
