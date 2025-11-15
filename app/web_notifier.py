"""Web-based alert dashboard."""

import base64
import time
from datetime import datetime
from typing import List, Dict, Optional
import threading
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request
from app.detector import Detection


class AlertStore:
    """Thread-safe storage for alerts."""

    def __init__(self):
        self.alerts: List[Dict] = []
        self.lock = threading.Lock()
        self._id_counter = 0

    def add_alert(
        self,
        image_bgr: np.ndarray,
        detections: List[Detection],
        source: str,
    ) -> int:
        """Add a new alert.

        Args:
            image_bgr: Alert image in BGR format
            detections: List of detections
            source: Video source identifier

        Returns:
            Alert ID
        """
        with self.lock:
            self._id_counter += 1
            alert_id = self._id_counter

            # Encode image to base64
            _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode("utf-8")

            # Get primary label
            if detections:
                labels = [det.label for det in detections]
                primary_label = max(set(labels), key=labels.count)
            else:
                primary_label = "fire"

            alert = {
                "id": alert_id,
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "label": primary_label,
                "count": len(detections),
                "confidences": [f"{det.conf:.2f}" for det in detections],
                "image": image_base64,
                "status": "pending",  # pending, confirmed, rejected
            }

            self.alerts.append(alert)
            return alert_id

    def get_pending_alerts(self) -> List[Dict]:
        """Get all pending alerts."""
        with self.lock:
            return [a for a in self.alerts if a["status"] == "pending"]

    def get_all_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts."""
        with self.lock:
            return self.alerts[-limit:][::-1]  # Most recent first

    def update_status(self, alert_id: int, status: str) -> bool:
        """Update alert status.

        Args:
            alert_id: Alert ID
            status: New status (confirmed/rejected)

        Returns:
            True if updated successfully
        """
        with self.lock:
            for alert in self.alerts:
                if alert["id"] == alert_id:
                    alert["status"] = status
                    return True
            return False


class WebNotifier:
    """Web-based alert notification system."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        """Initialize web notifier.

        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.store = AlertStore()
        self.app = Flask(
            __name__,
            template_folder="/Users/Egor.Safronov/Desktop/the-eye/app/web/templates",
            static_folder="/Users/Egor.Safronov/Desktop/the-eye/app/web/static",
        )
        self._setup_routes()
        self.server_thread = None

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            return render_template("dashboard.html")

        @self.app.route("/api/alerts")
        def get_alerts():
            status_filter = request.args.get("status", "all")
            if status_filter == "pending":
                alerts = self.store.get_pending_alerts()
            else:
                alerts = self.store.get_all_alerts()
            return jsonify(alerts)

        @self.app.route("/api/alerts/<int:alert_id>", methods=["POST"])
        def update_alert(alert_id):
            data = request.get_json()
            status = data.get("status")

            if status not in ["confirmed", "rejected"]:
                return jsonify({"error": "Invalid status"}), 400

            success = self.store.update_status(alert_id, status)
            if success:
                return jsonify({"success": True})
            else:
                return jsonify({"error": "Alert not found"}), 404

    def send_alert(
        self,
        image_bgr: np.ndarray,
        detections: List[Detection],
        source: str,
    ) -> int:
        """Add alert to the dashboard.

        Args:
            image_bgr: Alert image in BGR format
            detections: List of detections
            source: Video source identifier

        Returns:
            Alert ID
        """
        alert_id = self.store.add_alert(image_bgr, detections, source)
        print(f"✓ Alert #{alert_id} added to web dashboard")
        return alert_id

    def start(self):
        """Start the web server in a background thread."""
        if self.server_thread is not None:
            print("Web server already running")
            return

        def run_server():
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True,
            )

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        print(f"Web dashboard started at http://{self.host}:{self.port}")

    def stop(self):
        """Stop the web server."""
        # Flask doesn't have a clean shutdown in threaded mode
        # The daemon thread will terminate when main program exits
        self.server_thread = None
