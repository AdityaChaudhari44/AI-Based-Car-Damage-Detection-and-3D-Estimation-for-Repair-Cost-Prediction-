"""Dent / damage region detection.

Depth route: a dent is where depth deviates from the smooth panel surface. The original code thresholded
the gradient with THRESH_BINARY_INV, which marks the *flattest* areas; this version marks high deviation.
Optional YOLO route: if YOLO_WEIGHTS points at a fine-tuned YOLOv8-seg model (see scripts/train_yolo.py),
its polygons are used as the damage mask instead.
"""
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .depth import get_depth_estimator


@dataclass
class Detection:
    mask: np.ndarray       # uint8 {0,1}
    overlay: np.ndarray    # BGR image with damage highlighted
    depth: np.ndarray      # float32 [0,1]
    backend: str


def depth_deviation(depth, sigma_frac=0.1):
    h, w = depth.shape
    sigma = max(3.0, min(h, w) * sigma_frac)
    background = cv2.GaussianBlur(depth, (0, 0), sigma)          # smooth panel surface
    return np.abs(cv2.GaussianBlur(depth, (5, 5), 0) - background)


def clean_mask(mask, min_area_frac=0.002):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = min_area_frac * mask.size
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    return keep


def make_overlay(bgr, mask):
    heat = np.zeros_like(bgr)
    heat[mask == 1] = (0, 0, 255)
    out = cv2.addWeighted(bgr, 1.0, heat, 0.45, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    return out


class YoloMasks:
    name = "yolo-seg"

    def __init__(self, weights):
        from ultralytics import YOLO  # optional dependency
        self.model = YOLO(weights)

    def mask(self, bgr, conf=0.25):
        mask = np.zeros(bgr.shape[:2], np.uint8)
        res = self.model(bgr, conf=conf, verbose=False)[0]
        if res.masks is not None:
            for poly in res.masks.xy:
                cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
        return mask


class DentDetector:
    def __init__(self, depth_estimator=None, yolo=None, k_sigma=2.5, min_deviation=0.05):
        self.depth_estimator = depth_estimator or get_depth_estimator()
        weights = os.getenv("YOLO_WEIGHTS")
        self.yolo = yolo or (YoloMasks(weights) if weights else None)
        self.k_sigma, self.min_deviation = k_sigma, min_deviation

    def detect(self, bgr):
        if bgr is None or bgr.ndim != 3:
            raise ValueError("Expected a BGR colour image.")
        depth = self.depth_estimator.estimate(bgr)
        if self.yolo is not None:
            mask, backend = self.yolo.mask(bgr), f"{self.yolo.name}+{self.depth_estimator.name}"
        else:
            dev = depth_deviation(depth)
            thr = max(self.min_deviation, float(dev.mean() + self.k_sigma * dev.std()))
            mask, backend = clean_mask((dev > thr).astype(np.uint8)), self.depth_estimator.name
        return Detection(mask, make_overlay(bgr, mask), depth, backend)
