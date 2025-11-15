"""Telegram notification system."""

import io
import json
import threading
import time
from typing import List, Optional, Callable
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
        self.last_update_id = 0
        self.callback_handler: Optional[Callable] = None
        self.polling_thread = None
        self.polling_active = False

    def send_alert(
        self,
        image_bgr: np.ndarray,
        detections: List[Detection],
        source: str,
        alert_id: Optional[int] = None,
    ) -> bool:
        """Send alert with image to Telegram.

        Args:
            image_bgr: Image in BGR format
            detections: List of Detection objects
            source: Video source identifier
            alert_id: Optional alert ID for callback buttons

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

            if alert_id:
                caption += f"\nAlert ID: #{alert_id}"

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Convert to bytes
            bio = io.BytesIO()
            pil_image.save(bio, format="JPEG", quality=85)
            bio.seek(0)

            # Prepare inline keyboard with Confirm/Reject buttons
            keyboard = {
                "inline_keyboard": [
                    [
                        {
                            "text": "✅ Confirm",
                            "callback_data": f"confirm_{alert_id or 0}"
                        },
                        {
                            "text": "❌ Reject",
                            "callback_data": f"reject_{alert_id or 0}"
                        }
                    ]
                ]
            }

            # Send to Telegram
            url = f"{self.api_url}/sendPhoto"
            files = {"photo": ("alert.jpg", bio, "image/jpeg")}
            data = {
                "chat_id": self.chat_id,
                "caption": caption,
                "reply_markup": json.dumps(keyboard)
            }

            print(f"Sending Telegram alert to chat {self.chat_id}...")
            response = requests.post(url, files=files, data=data, timeout=30)

            if response.status_code == 200:
                print("✓ Telegram alert sent successfully")
                return True
            else:
                print(f"✗ Telegram error: {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except Exception as e:
            print(f"✗ Error sending Telegram alert: {e}")
            return False

    def set_callback_handler(self, handler: Callable):
        """Set callback handler for button clicks.

        Args:
            handler: Function to call with (alert_id, action) when button is clicked
        """
        self.callback_handler = handler

    def start_polling(self):
        """Start polling for callback queries in background thread."""
        if self.polling_thread and self.polling_active:
            print("Telegram polling already active")
            return

        self.polling_active = True
        self.polling_thread = threading.Thread(target=self._poll_updates, daemon=True)
        self.polling_thread.start()
        print("Telegram callback polling started")

    def stop_polling(self):
        """Stop polling for callback queries."""
        self.polling_active = False
        if self.polling_thread:
            self.polling_thread.join(timeout=5)
        print("Telegram callback polling stopped")

    def _poll_updates(self):
        """Poll for updates (button clicks) from Telegram."""
        while self.polling_active:
            try:
                url = f"{self.api_url}/getUpdates"
                params = {
                    "offset": self.last_update_id + 1,
                    "timeout": 10,
                    "allowed_updates": ["callback_query"]
                }

                response = requests.get(url, params=params, timeout=15)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok") and data.get("result"):
                        for update in data["result"]:
                            self.last_update_id = max(self.last_update_id, update["update_id"])
                            self._handle_update(update)

            except Exception as e:
                print(f"Error polling Telegram updates: {e}")
                time.sleep(5)

    def _handle_update(self, update: dict):
        """Handle a single update from Telegram.

        Args:
            update: Update dictionary from Telegram API
        """
        if "callback_query" in update:
            callback_query = update["callback_query"]
            callback_data = callback_query.get("data", "")
            message_id = callback_query["message"]["message_id"]
            chat_id = callback_query["message"]["chat"]["id"]

            # Parse callback data (format: "action_alertid")
            if "_" in callback_data:
                action, alert_id_str = callback_data.split("_", 1)
                try:
                    alert_id = int(alert_id_str)

                    # Call the callback handler if set
                    if self.callback_handler:
                        self.callback_handler(alert_id, action)

                    # Update the message to show the action taken
                    status_emoji = "✅" if action == "confirm" else "❌"
                    status_text = "CONFIRMED" if action == "confirm" else "REJECTED"

                    self._answer_callback_query(
                        callback_query["id"],
                        f"{status_emoji} Alert #{alert_id} {status_text}"
                    )

                    # Edit message to remove buttons and add status
                    self._edit_message_caption(
                        chat_id,
                        message_id,
                        callback_query["message"]["caption"] + f"\n\n{status_emoji} Status: {status_text}"
                    )

                except ValueError:
                    pass

    def _answer_callback_query(self, callback_query_id: str, text: str):
        """Answer a callback query (show notification to user).

        Args:
            callback_query_id: ID of the callback query
            text: Text to show in notification
        """
        try:
            url = f"{self.api_url}/answerCallbackQuery"
            data = {"callback_query_id": callback_query_id, "text": text}
            requests.post(url, json=data, timeout=10)
        except Exception as e:
            print(f"Error answering callback query: {e}")

    def _edit_message_caption(self, chat_id: int, message_id: int, caption: str):
        """Edit message caption to remove buttons.

        Args:
            chat_id: Chat ID
            message_id: Message ID
            caption: New caption text
        """
        try:
            url = f"{self.api_url}/editMessageCaption"
            data = {
                "chat_id": chat_id,
                "message_id": message_id,
                "caption": caption
            }
            requests.post(url, json=data, timeout=10)
        except Exception as e:
            print(f"Error editing message caption: {e}")
