import cv2
from app.config import Config
from app.detector import FireDetector
from app.streamer import VideoStreamer
from app.draw import draw_boxes


class FireDetectionPipeline:

    def __init__(self, config: Config):
        self.config = config
        self.frame_count = 0
        self.detection_count = 0

        # Initialize components
        print("Initializing detector...")
        self.detector = FireDetector(
            model_weights=config.model_weights,
            device=config.device,
            class_map=config.class_map,
        )

        print(f"Initializing video streamer: {config.source}")
        self.streamer = VideoStreamer(source=config.source)

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

                # Display if enabled
                if self.config.display:
                    annotated = draw_boxes(frame, detections)
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

    def _cleanup(self):
        print("Cleaning up...")
        self.streamer.release()
        if self.config.display:
            cv2.destroyAllWindows()

        print(f"\nTotal frames: {self.frame_count}")
        print(f"Total detections: {self.detection_count}")
