#!/usr/bin/env python3
"""Test Telegram connection."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.notifier import TelegramNotifier
import cv2
import numpy as np

# Load from .env
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

if not token or not chat_id:
    print("Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in .env")
    sys.exit(1)

print(f"Token: {token[:20]}...")
print(f"Chat ID: {chat_id}")

# Create a test image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(img, "Test Alert", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

# Create fake detection
from app.detector import Detection
det = Detection(100, 100, 200, 200, 0.95, 0, "fire")

# Send test
notifier = TelegramNotifier(token, chat_id)
success = notifier.send_alert(img, [det], "test_source")

if success:
    print("✓ Test alert sent successfully!")
else:
    print("✗ Failed to send test alert")
