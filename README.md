# AI-Based Car Damage Detection and 3D Estimation for Repair Cost Prediction

Runnable package built from the code in your project report (`reports/T15_Project Code.pdf`), turned into an
importable library + **FastAPI** service + Docker image + tests. Your original reports are in `reports/`.

## Pipeline
`image -> depth (MiDaS / classical) -> damage mask (depth deviation, or YOLOv8-seg if YOLO_WEIGHTS set) -> cost (INR) -> JSON`

```bash
pip install -r requirements.txt                         # core: OpenCV + FastAPI, classical fallback depth
pip install -r requirements-ml.txt                      # PyTorch route: MiDaS DPT-Hybrid + ResNet-50
pytest -q tests                                         # 6 tests
uvicorn car_damage.api:app --port 8000                  # open http://127.0.0.1:8000/ in your browser; then:
curl -F file=@car.jpg "http://localhost:8000/analyze?overlay=true"
```
Response: `damage_detected, region, area_cm2, avg_depth, estimated_cost_inr, backend` (+ `overlay_png_base64`).

Use from Python / your Tkinter GUI:
```python
from car_damage import analyze_image
import cv2; print(analyze_image(cv2.imread("car.jpg")))
```

## Backends (`DEPTH_BACKEND=auto|midas|classical`)
| Backend | What it is | Status |
|---|---|---|
| `midas` | MiDaS DPT-Hybrid via `torch.hub` (PyTorch) | Written from your report code; **not run in the build sandbox** (no PyTorch there) - test it on your machine first |
| `classical` | OpenCV luminance proxy, not real depth | Tested; for CI/demos only, expect false positives on real photos |
| YOLO (`YOLO_WEIGHTS=...`) | YOLOv8-seg masks; train with `scripts/train_yolo.py` | Template; **no trained weights included** |
| `vehicle.py` | torchvision ResNet-50 vehicle type | Written; not run in the sandbox |

## Fixes relative to the code in the report
* `THRESH_BINARY_INV` on the depth gradient marked the **flattest** regions, not dents; now marks high deviation from the smooth panel surface.
* MiDaS was fed a BGR image; it expects RGB.
* `pretrained=True` (deprecated) replaced by `ResNet50_Weights.DEFAULT`.
* Cost model separated from I/O and parameterised (`CostConfig`); debug `print` removed.

## Honest limitations (say these in interviews)
* `px_per_cm2 = 500` and `150 INR/cm2` are **uncalibrated** constants from the report, not real garage data.
* "Region" (bonnet/bumper/door) is guessed from where the damage sits in the frame, not from car-part detection.
* No labelled dataset evaluation here. To claim detection quality, fine-tune YOLOv8-seg on a damage dataset and
  report `model.val()` mAP, and compare cost estimates with real repair quotes.
* The Tkinter GUI and Open3D 3D viewer from the report are not part of this package.

## Docker
```bash
docker build -t car-damage . && docker run -p 8000:8000 car-damage
# with PyTorch (CPU):
docker build --build-arg EXTRAS="-r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu" -t car-damage-ml .
```
