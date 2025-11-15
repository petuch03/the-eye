from typing import List, Dict
import numpy as np
from ultralytics import YOLO


class Detection:

    def __init__(self, x1: float, y1: float, x2: float, y2: float,
                 conf: float, cls: int, label: str = ""):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf
        self.cls = cls
        self.label = label

    @property
    def bbox(self) -> tuple:
        """Return bbox as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)


class FireDetector:
    def __init__(self, model_weights: str, device: str = "cpu", class_map: Dict[int, str] = None):
        self.model_weights = model_weights
        self.device = device
        self.class_map = class_map or {}

        print(f"Loading model from {model_weights}...")
        self.model = YOLO(model_weights)
        self.model.to(device)
        print("Model loaded successfully")

    def detect(self, frame: np.ndarray, conf: float = 0.25) -> List[Detection]:
        results = self.model.predict(frame, conf=conf, verbose=False)

        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    cls_id = int(class_ids[i])
                    label = self.class_map.get(cls_id, f"class_{cls_id}")

                    detection = Detection(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        conf=float(confidences[i]),
                        cls=cls_id,
                        label=label,
                    )
                    detections.append(detection)

        return detections
