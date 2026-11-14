"""Vehicle-type classifier: torchvision ResNet-50 (ImageNet) with keyword mapping. Optional (needs torch)."""
KEYWORDS = {
    "car": ["car", "sports car", "limousine", "cab", "convertible", "minivan", "beach wagon", "jeep", "racer"],
    "truck": ["truck", "pickup", "trailer", "tow truck", "moving van"],
    "bus": ["bus", "trolleybus", "school bus"],
    "motorcycle": ["moped", "motor scooter", "motorcycle"],
    "bicycle": ["bicycle", "mountain bike", "tricycle"],
}


class VehicleClassifier:
    def __init__(self):
        import torch
        from torchvision import models
        self.torch = torch
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights).eval()
        self.preprocess, self.categories = weights.transforms(), weights.meta["categories"]

    def predict(self, bgr):
        import cv2
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with self.torch.no_grad():
            probs = self.model(self.preprocess(img).unsqueeze(0)).softmax(1)[0]
        label = self.categories[int(probs.argmax())].lower()
        for vtype, words in KEYWORDS.items():
            if any(w in label for w in words):
                return {"vehicle_type": vtype, "imagenet_label": label, "confidence": round(float(probs.max()), 3)}
        return {"vehicle_type": "unknown", "imagenet_label": label, "confidence": round(float(probs.max()), 3)}
