#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import Config
from app.pipeline import FireDetectionPipeline
from app.utils import check_source_exists


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fire Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/run.py \\
      --source ./samples/video.mp4 \\
      --model-weights models/best.pt \\
      --class-map '{"0":"fire","1":"smoke"}' \\
      --target-classes '[0]' \\
      --display 1
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        help="Video source (file path or URL)",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        help="Path to YOLOv8 model weights",
    )
    parser.add_argument(
        "--conf",
        type=float,
        help="Confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        help="Input image size for inference",
    )
    parser.add_argument(
        "--class-map",
        type=str,
        help='Class ID to label mapping (JSON, e.g., \'{"0":"fire","1":"smoke"}\')',
    )
    parser.add_argument(
        "--target-classes",
        type=str,
        help="Target class IDs (JSON array, e.g., '[0,1]')",
    )
    parser.add_argument(
        "--display",
        type=int,
        choices=[0, 1],
        help="Display video window (1=yes, 0=no)",
    )
    parser.add_argument(
        "--telegram-bot-token",
        type=str,
        help="Telegram bot token for alerts",
    )
    parser.add_argument(
        "--telegram-chat-id",
        type=str,
        help="Telegram chat ID for alerts",
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        help="Consecutive detections before alert (default: 3)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        help="Alert cooldown in seconds (default: 30)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load base config from env
    config = Config.from_env()

    # Apply CLI overrides
    if args.source:
        config.source = args.source
    if args.model_weights:
        config.model_weights = args.model_weights
    if args.conf is not None:
        config.conf_thresh = args.conf
    if args.imgsz:
        config.img_size = args.imgsz
    if args.display is not None:
        config.display = args.display
    if args.telegram_bot_token:
        config.telegram_bot_token = args.telegram_bot_token
    if args.telegram_chat_id:
        config.telegram_chat_id = args.telegram_chat_id
    if args.consecutive:
        config.consecutive_detections = args.consecutive
    if args.cooldown:
        config.alert_cooldown = args.cooldown

    # Parse class map
    if args.class_map:
        try:
            class_map_dict = json.loads(args.class_map)
            config.class_map = {int(k): v for k, v in class_map_dict.items()}
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing --class-map: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        config.class_map = {}

    # Parse target classes
    if args.target_classes:
        try:
            target_list = json.loads(args.target_classes)
            config.target_classes = [int(x) for x in target_list]
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing --target-classes: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        config.target_classes = []

    # Validate source
    if not config.source:
        print("Error: No video source specified. Use --source or set SOURCE in .env", file=sys.stderr)
        sys.exit(1)

    if not check_source_exists(config.source):
        print(f"Error: Video source not found: {config.source}", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    try:
        pipeline = FireDetectionPipeline(config)
        pipeline.run()
        sys.exit(0)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
