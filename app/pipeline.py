import time
import cv2
from app.config import Config
from app.detector import FireDetector
from app.streamer import VideoStreamer
from app.draw import draw_boxes
from app.notifier import TelegramNotifier
from app.web_notifier import WebNotifier


class FireDetectionPipeline:

    def __init__(self, config: Config):
        self.config = config
        self.frame_count = 0
        self.detection_count = 0
        self.alert_count = 0

        # Alert tracking
        self.consecutive_count = 0
        self.last_alert_time = 0.0

        # Initialize components
        print("Initializing detector...")
        self.detector = FireDetector(
            model_weights=config.model_weights,
            device=config.device,
            class_map=config.class_map,
        )

        print(f"Initializing video streamer: {config.source}")
        self.streamer = VideoStreamer(source=config.source)

        # Initialize Web dashboard if enabled (initialize first to get AlertStore)
        self.web = None
        if config.web_dashboard_enabled:
            print("Initializing web dashboard...")
            self.web = WebNotifier(
                host=config.web_host,
                port=config.web_port,
            )
            self.web.start()

        # Initialize Telegram if configured
        self.telegram = None
        if config.telegram_bot_token and config.telegram_chat_id:
            print("Initializing Telegram notifier...")
            self.telegram = TelegramNotifier(
                bot_token=config.telegram_bot_token,
                chat_id=config.telegram_chat_id,
            )

            # Set up callback handler to sync Telegram actions with web dashboard
            if self.web:
                def handle_telegram_callback(alert_id: int, action: str):
                    """Handle Telegram button clicks and sync with web dashboard."""
                    status = "confirmed" if action == "confirm" else "rejected"
                    success = self.web.store.update_status(alert_id, status)
                    if success:
                        print(f"✓ Alert #{alert_id} {status} via Telegram")
                    else:
                        print(f"✗ Failed to update alert #{alert_id}")

                self.telegram.set_callback_handler(handle_telegram_callback)
                self.telegram.start_polling()

    def run(self):
        """Run the detection pipeline."""
        print("Starting detection pipeline...")
        print(f"Display: {self.config.display}, Confidence: {self.config.conf_thresh}")

        try:
            while True:
                # Read frame
                ret, frame = self.streamer.read()
                if not ret or frame is None:
                    print("End of video")
                    break

                self.frame_count += 1

                # Run detection
                detections = self.detector.detect(frame, conf=self.config.conf_thresh)

                # Filter by target classes if specified
                if self.config.target_classes:
                    detections = [d for d in detections if d.cls in self.config.target_classes]

                if detections:
                    self.detection_count += len(detections)
                    print(f"Frame {self.frame_count}: Detected {len(detections)} object(s)")

                # Check alert logic
                should_alert = self._should_alert(len(detections) > 0)

                # Draw boxes for display or alert
                annotated = draw_boxes(frame, detections) if detections or self.config.display else frame

                # Send alert if triggered
                if should_alert and detections:
                    print(f"🚨 ALERT triggered! (consecutive: {self.consecutive_count})")

                    # Create alert and get ID (web dashboard first to get alert_id)
                    alert_id = None
                    if self.web:
                        alert_id = self.web.send_alert(annotated, detections, self.config.source)

                    # Send to Telegram with alert_id for button callbacks
                    if self.telegram:
                        self.telegram.send_alert(annotated, detections, self.config.source, alert_id)

                    self.alert_count += 1
                    self.last_alert_time = time.time()
                    self.consecutive_count = 0

                # Display if enabled
                if self.config.display:
                    cv2.imshow("Fire Detection", annotated)

                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User quit")
                        break

                # Progress logging
                if self.frame_count % 100 == 0:
                    print(f"Processed {self.frame_count} frames")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self._cleanup()

    def _should_alert(self, has_detections: bool) -> bool:
        """Check if alert should be triggered.

        Args:
            has_detections: Whether current frame has detections

        Returns:
            True if alert should be triggered
        """
        if has_detections:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0
            return False

        # Check consecutive threshold
        if self.consecutive_count < self.config.consecutive_detections:
            return False

        # Check cooldown
        current_time = time.time()
        if current_time - self.last_alert_time < self.config.alert_cooldown:
            return False

        return True

    def _cleanup(self):
        print("Cleaning up...")
        self.streamer.release()
        if self.config.display:
            cv2.destroyAllWindows()

        print(f"\nTotal frames: {self.frame_count}")
        print(f"Total detections: {self.detection_count}")
        print(f"Total alerts: {self.alert_count}")
