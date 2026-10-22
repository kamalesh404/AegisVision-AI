"""
Violation & Incident Alert Manager:
- Manages security intrusions, safety violations, and spatial events
- Captures & persists snapshot evidence thumbnails
- Maintains in-memory ring buffer & exports enterprise audit CSV reports
"""
import os
import csv
import time
import cv2
from datetime import datetime

class IncidentManager:
    def __init__(self, alerts_dir: str = "alerts", max_history: int = 150):
        self.alerts_dir = alerts_dir
        self.max_history = max_history
        self.events = []
        os.makedirs(self.alerts_dir, exist_ok=True)
        self.csv_log_path = os.path.join(self.alerts_dir, "incident_audit_log.csv")
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Event_ID", "Type", "Severity", "Object_ID", "Class", "Confidence", "Zone_or_Line", "Snapshot_File"])

    def recent(self, limit: int = 30):
        return self.get_recent_events(limit)

    def record(self, event_type: str, label: str, track_id=None, direction: str = "", zone_type: str = "intrusion", frame=None, track=None, severity: str = None):
        """Flexible record helper compatible with pipeline calls."""
        if severity is None:
            severity = "CRITICAL" if ("danger" in label.lower() or "restricted" in label.lower() or zone_type == "danger") else "WARNING"
        
        # If a track object is passed directly
        if track is not None:
            return self.record_incident(event_type=event_type, severity=severity, track=track, zone_or_line=label, frame=frame)
        
        # Otherwise create a lightweight mock track representation
        class MockTrack:
            def __init__(self, tid):
                self.track_id = tid
                self.cls_name = "object"
                self.confidence = 0.9
                self.bbox = [20, 20, 100, 100]
                self.dwell_time_sec = 1.0

        mt = MockTrack(track_id or 1)
        return self.record_incident(event_type=f"{event_type}_{direction}" if direction else event_type, severity=severity, track=mt, zone_or_line=label, frame=frame)

    def record_incident(self, event_type: str, severity: str, track, zone_or_line: str, frame=None):
        """
        Records an incident, optionally saves snapshot image of the violation.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        event_id = f"EVT-{int(time.time() * 1000) % 1000000:06d}"
        
        snapshot_filename = ""
        if frame is not None:
            snapshot_filename = f"{event_id}_{track.cls_name}_track{track.track_id}.jpg"
            full_path = os.path.join(self.alerts_dir, snapshot_filename)
            
            # Draw highlight on snapshot before saving
            snap_img = frame.copy()
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            cv2.rectangle(snap_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"VIOLATION: {event_type} [#{track.track_id}]"
            cv2.putText(snap_img, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(full_path, snap_img)

        event_data = {
            "id": event_id,
            "timestamp": timestamp_str,
            "type": event_type,
            "severity": severity,  # "CRITICAL", "WARNING", "INFO"
            "track_id": track.track_id,
            "cls_name": track.cls_name,
            "confidence": round(track.confidence, 2),
            "zone_or_line": zone_or_line,
            "snapshot": snapshot_filename,
            "dwell_sec": track.dwell_time_sec
        }

        # In-memory rolling history
        self.events.insert(0, event_data)
        if len(self.events) > self.max_history:
            self.events.pop()

        # Append to audit CSV
        try:
            with open(self.csv_log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp_str, event_id, event_type, severity, 
                    track.track_id, track.cls_name, f"{track.confidence:.2f}", 
                    zone_or_line, snapshot_filename
                ])
        except Exception as e:
            print(f"Error writing to audit CSV: {e}")

        return event_data

    def get_recent_events(self, limit: int = 30):
        return self.events[:limit]

    def clear(self):
        self.events.clear()
