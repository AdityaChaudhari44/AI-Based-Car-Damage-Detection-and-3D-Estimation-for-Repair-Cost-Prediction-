# AI-Based Car Damage Detection and 3D Estimation for Repair Cost Prediction

Runnable package built from the code in your project report (reports/T15_Project Code.pdf), turned into an importable library + FastAPI service + Docker image + tests. Your original reports are in reports/.

## Pipeline

image -> depth (MiDaS / classical) -> damage mask (depth deviation, or YOLOv8-seg if YOLO_WEIGHTS set) -> cost (INR) -> JSON

```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt
pytest -v tests
uvicorn car_damage.api:app --port 8000
curl -F file=@car.jpg "http://localhost:8000/analyze?overlay=true"
```

* **core:** OpenCV + FastAPI, classical fallback depth
* **PyTorch route:** MiDaS DPT-Hybrid + ResNet-50
* **6 tests**
* **open http://127.0.0 in your browser; it provides a web test UI**

---

**Response:** damage_detected, region, area_cm2, avg_depth, estimated_cost_inr, backend (+ overlay_png_base64).

### Use from Python / your Tkinter GUI:

```python
from car_damage import analyze_image
import cv2

print(analyze_image(cv2.imread("car.jpg")))
```

---

## Backends (DEPTH_BACKEND=auto|midas|classical)

| Backend | Core Tech | Notes |
| :--- | :--- | :--- |
| **auto** | Tries midas, falls back to classical | Safest mode; guarantees execution regardless of host hardware configurations. |
| **midas** | PyTorch model from Intel ISL | Highly accurate semantic depth map generation; requires torch/torchvision extensions. |
| **classical** | OpenCV Block Matching / Otsu thresholding variations | Runs anywhere with near-zero latency; local pixel variance analysis framework. |

### Best choosing standard behaviors:
* `DEPTH_BACKEND=auto` -> Recommended dynamic runtime deployment option.
* `YOLO_WEIGHTS=path/to/best.pt` -> Activates the custom instance segmentation pipeline.
* `DEFAULT_REPAIR_RATE=250` -> Changes cost scaling index metric per structural square centimeter.
* Set environment key variables before booting uvicorn to target proper behaviors.

---

## Service

Once running, the service exposes a standard web API at port 8000.
To run assertions across all internal software testing checks:
```bash
pytest tests/
```
