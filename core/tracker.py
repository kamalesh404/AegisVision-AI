"""
Core Object Tracker with ID Persistence, Motion Trails, and Kinematics.
Supports Centroid + IoU tracking with smooth trajectory history.
"""
import numpy as np
from collections import deque
import math
import time

class TrackedObject:
    def __init__(self, track_id: int, bbox: list, cls_id: int, cls_name: str, confidence: float, max_history: int = 30):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.confidence = confidence
        self.history = deque(maxlen=max_history)
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.missed_frames = 0
        
        # Spatial analytics flags
        self.crossed_lines = set()
        self.current_zones = set()
        self.violation_triggered = False

        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.history.append((cx, cy))

    @property
    def centroid(self):
        return self.history[-1]

    @property
    def dwell_time_sec(self):
        return round(time.time() - self.first_seen, 1)

    @property
    def velocity_vector(self):
        if len(self.history) < 2:
            return (0, 0)
        p1 = self.history[0]
        p2 = self.history[-1]
        dt = max(1, len(self.history) - 1)
        return ((p2[0] - p1[0]) / dt, (p2[1] - p1[1]) / dt)

    def update(self, bbox: list, cls_id: int, cls_name: str, confidence: float):
        self.bbox = bbox
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.confidence = confidence
        self.last_seen = time.time()
        self.missed_frames = 0
        
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.history.append((cx, cy))


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


class CentroidIoUTracker:
    def __init__(self, max_distance: float = 65.0, max_missed: int = 15, max_history: int = 30):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.max_history = max_history
        self.next_id = 1
        self.tracks = {}  # track_id -> TrackedObject

    def active_count(self) -> int:
        return len(self.tracks)

    def reset(self):
        self.next_id = 1
        self.tracks.clear()

    def update(self, detections: list):
        """
        detections: list of dicts:
          {'bbox': [x1, y1, x2, y2], 'cls_id': int, 'cls_name': str, 'conf': float}
        Returns:
          list of TrackedObject
        """
        if not detections:
            # Mark all existing as missed
            to_delete = []
            for tid, track in self.tracks.items():
                track.missed_frames += 1
                if track.missed_frames > self.max_missed:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return list(self.tracks.values())

        if not self.tracks:
            # All detections become new tracks
            for det in detections:
                track = TrackedObject(
                    self.next_id, det['bbox'], det['cls_id'], det['cls_name'], det['conf'], self.max_history
                )
                self.tracks[self.next_id] = track
                self.next_id += 1
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        prev_tracks = [self.tracks[tid] for tid in track_ids]

        # Calculate cost matrix based on Euclidean distance + IoU
        cost_matrix = np.zeros((len(prev_tracks), len(detections)), dtype=np.float32)

        for i, track in enumerate(prev_tracks):
            tcx, tcy = track.centroid
            for j, det in enumerate(detections):
                dcx = (det['bbox'][0] + det['bbox'][2]) / 2.0
                dcy = (det['bbox'][1] + det['bbox'][3]) / 2.0
                dist = math.hypot(tcx - dcx, tcy - dcy)
                iou = calculate_iou(track.bbox, det['bbox'])
                
                # If same class, boost matching
                class_penalty = 0.0 if track.cls_name == det['cls_name'] else 40.0
                
                cost = dist - (iou * 50.0) + class_penalty
                cost_matrix[i, j] = cost

        # Greedy matching
        matched_tracks = set()
        matched_detections = set()

        flat_indices = np.argsort(cost_matrix, axis=None)
        for idx in flat_indices:
            row = idx // len(detections)
            col = idx % len(detections)

            if row in matched_tracks or col in matched_detections:
                continue

            # Check threshold
            track = prev_tracks[row]
            det = detections[col]
            tcx, tcy = track.centroid
            dcx = (det['bbox'][0] + det['bbox'][2]) / 2.0
            dcy = (det['bbox'][1] + det['bbox'][3]) / 2.0
            dist = math.hypot(tcx - dcx, tcy - dcy)

            if dist < self.max_distance:
                track.update(det['bbox'], det['cls_id'], det['cls_name'], det['conf'])
                matched_tracks.add(row)
                matched_detections.add(col)

        # Handle unmatched existing tracks
        to_delete = []
        for i, track in enumerate(prev_tracks):
            if i not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed:
                    to_delete.append(track.track_id)
        for tid in to_delete:
            del self.tracks[tid]

        # Handle new detections
        for j, det in enumerate(detections):
            if j not in matched_detections:
                track = TrackedObject(
                    self.next_id, det['bbox'], det['cls_id'], det['cls_name'], det['conf'], self.max_history
                )
                self.tracks[self.next_id] = track
                self.next_id += 1

        return list(self.tracks.values())
