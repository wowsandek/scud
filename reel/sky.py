"""Sky segmentation (SegFormer-B2 / ADE20k, ONNX) with edge refinement and disk cache.

The mask is 1.0 where the pixel is sky; text drawn "behind" the scene is
multiplied by it so hills, cliffs and buildings occlude the letters.
"""
import hashlib
import os

import cv2
import numpy as np
import onnxruntime as ort

SKY_CLASS = 2
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class SkySegmenter:
    def __init__(self, model_path, cache_dir):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _infer(self, img):
        x = cv2.resize(img, (512, 896), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
        logits = self.sess.run(None, {"pixel_values": x})[0][0]
        e = np.exp(logits - logits.max(0))
        return e[SKY_CLASS] / e.sum(0)

    def mask(self, img, key):
        """Refined sky mask (float32 HxW, 0..1). `key` identifies the frame for caching."""
        path = os.path.join(self.cache_dir, hashlib.md5(key.encode()).hexdigest() + ".png")
        if os.path.exists(path):
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        h, w = img.shape[:2]
        p = cv2.resize(self._infer(img), (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
        r = cv2.ximgproc.guidedFilter(guide=img, src=p.astype(np.float32), radius=6, eps=1e-4)
        r = np.clip((r - 0.3) / 0.4, 0, 1).astype(np.float32)
        cv2.imwrite(path, (r * 255).astype(np.uint8))
        return r
