"""Depth estimation backends.

* MiDaSDepth   - MiDaS DPT-Hybrid via torch.hub (PyTorch). Used when torch is installed.
* ShadingDepth - classical OpenCV luminance proxy. NOT real depth: a dependency-free fallback so the
                 pipeline, API and tests run anywhere. Expect false positives on real photos.
"""
import os

import cv2
import numpy as np


def normalize01(arr):
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class ShadingDepth:
    name = "classical-shading-proxy"

    def estimate(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return normalize01(cv2.GaussianBlur(gray, (5, 5), 0))


class MiDaSDepth:
    name = "midas-dpt-hybrid"

    def __init__(self, model_type="DPT_Hybrid"):
        import torch  # optional dependency
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True).to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform

    def estimate(self, bgr):
        torch = self.torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # MiDaS expects RGB (the original passed BGR)
        batch = self.transform(rgb).to(self.device)
        with torch.no_grad():
            pred = self.model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=bgr.shape[:2], mode="bicubic", align_corners=False).squeeze()
        return normalize01(pred.cpu().numpy())


def get_depth_estimator():
    """DEPTH_BACKEND = auto (default) | midas | classical."""
    backend = os.getenv("DEPTH_BACKEND", "auto").lower()
    if backend in ("auto", "midas"):
        try:
            return MiDaSDepth()
        except Exception:
            if backend == "midas":
                raise
    return ShadingDepth()
