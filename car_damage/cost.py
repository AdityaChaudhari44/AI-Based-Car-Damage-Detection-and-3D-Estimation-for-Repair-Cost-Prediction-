"""Repair-cost heuristic (INR) from damage mask + depth. Rates are configurable and UNCALIBRATED:
px_per_cm2 and base_rate come from the project report, not from real garage quotes."""
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class CostConfig:
    px_per_cm2: float = 500.0
    base_rate_inr_per_cm2: float = 150.0
    min_cost: float = 500.0
    max_cost: float = 50000.0


def locate_region(mask):
    """Panel guess from the damage centroid's position in the frame (not true car-part detection)."""
    h, w = mask.shape
    m = cv2.moments(mask, binaryImage=True)
    cx, cy = m["m10"] / (m["m00"] + 1e-6), m["m01"] / (m["m00"] + 1e-6)
    if cy < h * 0.35:
        return "bonnet", 1.3
    if cy > h * 0.7:
        return "bumper", 1.1
    if cx < w * 0.3 or cx > w * 0.7:
        return "door/fender", 1.2
    return "side_panel", 1.0


def estimate_repair_cost(depth, mask, cfg=None):
    cfg = cfg or CostConfig()
    area_px = int(mask.sum())
    if area_px == 0:
        return {"damage_detected": False, "estimated_cost_inr": 0.0, "area_cm2": 0.0,
                "avg_depth": 0.0, "region": None}
    area_cm2 = area_px / cfg.px_per_cm2
    avg_depth = float(np.mean(depth[mask == 1]))
    region, region_factor = locate_region(mask)
    cost = (area_cm2 * cfg.base_rate_inr_per_cm2 * region_factor
            * (1 + avg_depth * 2.5) * (1 + area_cm2 / 800.0))
    return {"damage_detected": True, "estimated_cost_inr": round(float(np.clip(cost, cfg.min_cost, cfg.max_cost)), 2),
            "area_cm2": round(area_cm2, 2), "avg_depth": round(avg_depth, 3), "region": region}
