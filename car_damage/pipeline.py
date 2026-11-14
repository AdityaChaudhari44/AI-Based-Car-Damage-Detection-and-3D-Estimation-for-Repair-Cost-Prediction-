import base64

import cv2
import numpy as np

from .cost import estimate_repair_cost
from .detector import DentDetector

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = DentDetector()
    return _detector


def analyze_image(bgr, detector=None, include_overlay=False, include_3d=False, classifier=None):
    detection = (detector or get_detector()).detect(bgr)
    result = estimate_repair_cost(detection.depth, detection.mask)
    result["backend"] = detection.backend
    if classifier is not None:
        result.update(classifier.predict(bgr))
    if include_overlay:
        ok, png = cv2.imencode(".png", detection.overlay)
        result["overlay_png_base64"] = base64.b64encode(png.tobytes()).decode() if ok else None
    if include_3d:
        # Downsample for a responsive browser-side 3D mesh. MiDaS returns relative
        # depth, not metric distance; the surface is a visual reconstruction only.
        grid_w = min(72, max(24, bgr.shape[1] // 8))
        grid_h = min(72, max(24, bgr.shape[0] // 8))
        depth_grid = cv2.resize(detection.depth.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        mask_grid = cv2.resize(detection.mask.astype(np.uint8), (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_grid = cv2.resize(rgb, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        # Highlight detected damage in red on the reconstructed mesh.
        colors = rgb_grid.reshape(-1, 3).copy()
        colors[mask_grid.reshape(-1) > 0] = np.array([255, 55, 55], dtype=np.uint8)
        result["3d_surface"] = {
            "width": int(grid_w), "height": int(grid_h),
            "depth": np.round(depth_grid, 4).reshape(-1).tolist(),
            "colors": [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors],
            "damage_mask": mask_grid.reshape(-1).astype(int).tolist(),
            "depth_type": "relative",
            "note": "Approximate single-image surface visualization. Depth is relative, not calibrated in centimeters; it is not a measurement-grade reconstruction."
        }
    return result


def analyze_bytes(data, **kwargs):
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image (send JPEG or PNG).")
    return analyze_image(bgr, **kwargs)
