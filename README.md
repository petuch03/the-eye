# Fire Detection System

Real-time fire and smoke detection using YOLOv8.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run detection on a video file:

```bash
python scripts/run.py \
    --source ./samples/v4.mp4 \
    --model-weights models/best.pt \
    --class-map '{"0":"fire","1":"smoke"}' \
    --target-classes '[0]' \
    --display 1
```

### Telegram Alerts

Get real-time alerts via Telegram:

```bash
python scripts/run.py \
    --source ./samples/v4.mp4 \
    --model-weights models/best.pt \
    --class-map '{"0":"fire","1":"smoke"}' \
    --target-classes '[0]' \
    --telegram-bot-token "YOUR_BOT_TOKEN" \
    --telegram-chat-id "YOUR_CHAT_ID" \
    --consecutive 3 \
    --cooldown 30
```

**Setup Telegram:**
1. Talk to [@BotFather](https://t.me/botfather) on Telegram
2. Create a new bot with `/newbot` and get your token
3. Get your chat ID from [@userinfobot](https://t.me/userinfobot)

### Options

- `--source` - Video file path or stream URL
- `--model-weights` - Path to YOLOv8 model weights
- `--class-map` - JSON mapping of class IDs to labels
- `--target-classes` - JSON array of class IDs to detect
- `--display` - Show video window (1=yes, 0=no)
- `--conf` - Confidence threshold (default: 0.25)
- `--imgsz` - Input image size (default: 640)
- `--telegram-bot-token` - Telegram bot token for alerts
- `--telegram-chat-id` - Telegram chat ID for alerts
- `--consecutive` - Consecutive detections before alert (default: 3)
- `--cooldown` - Alert cooldown in seconds (default: 30)

### Configuration File

Alternatively, create a `.env` file:

```bash
cp .env.example .env
# Edit .env with your settings
```

CLI arguments override `.env` settings.

## Project Structure

```
the-eye/
├── app/               # Core modules
│   ├── config.py      # Configuration
│   ├── detector.py    # YOLOv8 detector
│   ├── streamer.py    # Video reader
│   ├── pipeline.py    # Detection pipeline
│   ├── draw.py        # Visualization
│   ├── notifier.py    # Telegram alerts
│   └── utils.py       # Utilities
├── scripts/
│   └── run.py         # Main script
├── models/            # Model weights
├── samples/           # Test videos
└── requirements.txt
```

## Controls

- Press `q` to quit during video playback

## Requirements

- Python 3.10+
- YOLOv8
- OpenCV
