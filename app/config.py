"""Basic configuration management."""

import os


class Config:
    def __init__(self):
        self.source = os.getenv("SOURCE", "./samples/video.mp4")
        self.model_weights = os.getenv("MODEL_WEIGHTS", "models/best.pt")
        self.conf_thresh = float(os.getenv("CONF_THRESH", "0.25"))
        self.img_size = int(os.getenv("IMG_SIZE", "640"))
        self.display = int(os.getenv("DISPLAY", "1"))
        self.device = "cpu"

    @classmethod
    def from_env(cls):
        return cls()
