import os, sys
from pathlib import Path
os.environ["DEPTH_BACKEND"] = "classical"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from car_damage.api import app
from car_damage.cost import estimate_repair_cost, locate_region, CostConfig
from car_damage.detector import DentDetector
from car_damage.depth import ShadingDepth


def panel(blob=True, center=(160, 120)):
    """Synthetic flat grey panel; optional smooth shading blob standing in for a dent."""
    img = np.full((240, 320), 128, np.float32)
    if blob:
        yy, xx = np.mgrid[:240, :320]
        img += 70 * np.exp(-(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * 14 ** 2)))
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_flat_panel_has_no_damage():
    det = DentDetector(ShadingDepth()).detect(panel(blob=False))
    assert det.mask.sum() == 0
    assert estimate_repair_cost(det.depth, det.mask)["damage_detected"] is False


def test_blob_is_localised():
    det = DentDetector(ShadingDepth()).detect(panel(center=(250, 60)))
    ys, xs = np.nonzero(det.mask)
    assert len(xs) > 0
    assert abs(xs.mean() - 250) < 25 and abs(ys.mean() - 60) < 25     # centroid near the blob
    assert det.overlay.shape == det.mask.shape + (3,)


def test_cost_increases_with_area_and_is_clamped():
    depth = np.full((100, 100), 0.5, np.float32)
    small, big = np.zeros((100, 100), np.uint8), np.zeros((100, 100), np.uint8)
    small[45:50, 45:50] = 1; big[30:70, 30:70] = 1
    cs = estimate_repair_cost(depth, small)["estimated_cost_inr"]
    cb = estimate_repair_cost(depth, big)["estimated_cost_inr"]
    assert 500 <= cs < cb <= 50000
    assert estimate_repair_cost(depth, big, CostConfig(max_cost=1000))["estimated_cost_inr"] == 1000


def test_region_heuristic():
    for (y, x), expected in {(10, 50): "bonnet", (90, 50): "bumper", (50, 5): "door/fender", (50, 50): "side_panel"}.items():
        m = np.zeros((100, 100), np.uint8); m[y:y + 3, x:x + 3] = 1
        assert locate_region(m)[0] == expected


client = TestClient(app)

def png(img):
    return cv2.imencode(".png", img)[1].tobytes()

def test_api_health_and_analyze():
    assert client.get("/health").json()["backend"] == "classical-shading-proxy"
    r = client.post("/analyze?overlay=true", files={"file": ("a.png", png(panel()), "image/png")})
    body = r.json()
    assert r.status_code == 200 and body["damage_detected"] and body["estimated_cost_inr"] >= 500
    assert body["overlay_png_base64"]
    r2 = client.post("/analyze", files={"file": ("b.png", png(panel(blob=False)), "image/png")})
    assert r2.json()["damage_detected"] is False

def test_api_rejects_bad_input():
    assert client.post("/analyze", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415
    assert client.post("/analyze", files={"file": ("x.png", b"notanimage", "image/png")}).status_code == 400
    assert client.post("/analyze").status_code == 422


def test_api_returns_3d_surface_grid():
    r = client.post("/analyze?include_3d=true", files={"file": ("3d.png", png(panel()), "image/png")})
    body = r.json()
    assert r.status_code == 200
    assert "3d_surface" in body
    surface = body["3d_surface"]
    assert len(surface["depth"]) == surface["width"] * surface["height"]
    assert len(surface["colors"]) == surface["width"] * surface["height"]
    assert surface["depth_type"] == "relative"
