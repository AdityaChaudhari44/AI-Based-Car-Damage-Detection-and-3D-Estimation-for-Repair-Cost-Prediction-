"""Fine-tune a YOLOv8 segmentation model on a car-damage dataset (Roboflow / CarDD / your own labels).
Template: needs `pip install ultralytics` and a dataset YAML. Not run in this repo's tests.

    python scripts/train_yolo.py --data data/car_damage.yaml --epochs 50
Then:  export YOLO_WEIGHTS=runs/segment/train/weights/best.pt   (the API will use it automatically)
"""
import argparse

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--base", default="yolov8n-seg.pt")
a = p.parse_args()

model = YOLO(a.base)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz)
print(model.val())            # mAP50 / mAP50-95 for boxes and masks: put these numbers on your resume
