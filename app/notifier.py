"""Telegram notification system."""

import io
from typing import List
import numpy as np
import cv2
import requests
from PIL import Image
from app.detector import Detection


class TelegramNotifier:
    """Sends alerts via Telegram bot."""

    def __init__(self, bot_token: str, chat_id: str):
        """Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token
            chat_id: Target chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"

    def send_alert(
        self,
        image_bgr: np.ndarray,
        detections: List[Detection],
        source: str,
    ) -> bool:
        """Send alert with image to Telegram.

        Args:
            image_bgr: Image in BGR format
            detections: List of Detection objects
            source: Video source identifier

        Returns:
            True if sent successfully
        """
        try:
            # Prepare caption
            detection_count = len(detections)
            conf_values = [f"{det.conf:.2f}" for det in detections]
            conf_str = ", ".join(conf_values)

            # Get primary label
            if detections:
                labels = [det.label for det in detections]
                primary_label = max(set(labels), key=labels.count)
            else:
                primary_label = "fire"

            caption = (
                f"🚨 ALERT: {primary_label} detected\n"
                f"Count: {detection_count}\n"
                f"Confidence: {conf_str}\n"
                f"Source: {source}"
            )

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Convert to bytes
            bio = io.BytesIO()
            pil_image.save(bio, format="JPEG", quality=85)
            bio.seek(0)

            # Send to Telegram
            url = f"{self.api_url}/sendPhoto"
            files = {"photo": ("alert.jpg", bio, "image/jpeg")}
            data = {"chat_id": self.chat_id, "caption": caption}

            print(f"Sending Telegram alert to chat {self.chat_id}...")
            response = requests.post(url, files=files, data=data, timeout=30)

            if response.status_code == 200:
                print("✓ Telegram alert sent successfully")
                return True
            else:
                print(f"✗ Telegram error: {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Error sending Telegram alert: {e}")
            return False
