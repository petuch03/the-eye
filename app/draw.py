from typing import List
import cv2
import numpy as np
from app.detector import Detection


# Colors for different classes (BGR format)
COLORS = {
    "fire": (0, 69, 255),      # Orange-red
    "smoke": (128, 128, 128),  # Gray
    "default": (0, 255, 0),    # Green
}


def draw_boxes(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

        # Get color
        label = det.label if det.label else "default"
        color = COLORS.get(label.lower(), COLORS["default"])

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = f"{label} {det.conf:.2f}"
        cv2.putText(
            annotated,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated
