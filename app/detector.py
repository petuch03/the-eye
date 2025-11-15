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
    def __init__(self, model_weights: str, device: str = "cpu"):
        self.model_weights = model_weights
        self.device = device

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
                    detection = Detection(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        conf=float(confidences[i]),
                        cls=int(class_ids[i]),
                    )
                    detections.append(detection)

        return detections
