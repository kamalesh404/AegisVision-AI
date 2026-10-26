"""
Spatial Analytics Engine:
- Bi-directional Line Crossing / Tripwires (IN vs OUT counting)
- Polygon Geofencing / Intrusion Zone Detection
- Dwell Time & Loitering Analytics
- Dynamic Zone Occupancy & Capacity Management
- Real-time Spatial Heatmap Generation
- 2D Kinematics & Velocity Heading Estimation
"""
import time
import cv2
import numpy as np

def ccw(A, B, C):
    """Checks if three points are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def line_segments_intersect(A, B, C, D):
    """
    Returns True if line segment AB intersects line segment CD.
    A, B are points on segment 1; C, D are points on segment 2.
    """
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

def get_crossing_direction(prev_pt, curr_pt, line_pt1, line_pt2):
    """
    Calculates whether the movement from prev_pt to curr_pt crossed line (line_pt1 -> line_pt2)
    from Left-to-Right (IN) or Right-to-Left (OUT) using the 2D cross product.
    """
    # Vector along the line
    line_vec = (line_pt2[0] - line_pt1[0], line_pt2[1] - line_pt1[1])
    # Vector of movement
    move_vec = (curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
    
    # 2D cross product (line_vec x move_vec)
    cross_product = line_vec[0] * move_vec[1] - line_vec[1] * move_vec[0]
    return "IN" if cross_product > 0 else "OUT"


class TripwireLine:
    def __init__(self, pt1: tuple = None, pt2: tuple = None, label: str = "Tripwire", line_id: str = None, direction: str = "both"):
        self.line_id = line_id or f"tw_{int(time.time()*1000)%100000}"
        self.pt1 = (int(pt1[0]), int(pt1[1])) if pt1 is not None else (0, 0)
        self.pt2 = (int(pt2[0]), int(pt2[1])) if pt2 is not None else (0, 0)
        self.label = label
        self.direction = direction
        self.in_count = 0
        self.out_count = 0
        self.history_events = []

    def get_counts(self):
        return {"in_count": self.in_count, "out_count": self.out_count}

    def reset_counts(self):
        self.in_count = 0
        self.out_count = 0
        self.history_events.clear()

    def check_crossings(self, tracks: list):
        events = []
        for t in tracks:
            ev = self.check_crossing(t)
            if ev:
                events.append(ev)
        return events

    def check_crossing(self, track):
        """
        Check if tracked object trajectory crossed this tripwire line.
        """
        if len(track.history) < 2:
            return None

        # Check latest movement segment
        p_prev = track.history[-2]
        p_curr = track.history[-1]

        if line_segments_intersect(p_prev, p_curr, self.pt1, self.pt2):
            key = f"{self.line_id}_{len(track.history)}"
            if key not in track.crossed_lines:
                track.crossed_lines.add(key)
                direction = get_crossing_direction(p_prev, p_curr, self.pt1, self.pt2)
                if direction == "IN":
                    self.in_count += 1
                else:
                    self.out_count += 1
                
                event = {
                    "line_id": self.line_id,
                    "label": self.label,
                    "direction": direction,
                    "track_id": track.track_id,
                    "cls_name": track.cls_name,
                    "timestamp": round(track.last_seen, 2)
                }
                self.history_events.append(event)
                return event
        return None


class GeofenceZone:
    def __init__(
        self,
        points: list = None,
        label: str = "Restricted Zone",
        zone_type: str = "intrusion",
        zone_id: str = None,
        color=(0, 0, 255),
        polygon_points: list = None,
        dwell_threshold_sec: float = 0.0,
        max_occupancy: int = 0
    ):
        self.zone_id = zone_id or f"zone_{int(time.time()*1000)%100000}"
        raw_pts = points if points is not None else (polygon_points if polygon_points is not None else [])
        self.points = np.array(raw_pts, dtype=np.int32)
        self.label = label
        self.zone_type = zone_type
        self.color = color  # BGR
        self.active_intrusions = set()
        self.total_intrusions = 0
        
        # Enterprise Dwell Time & Capacity Analytics
        self.dwell_threshold_sec = float(dwell_threshold_sec)
        self.max_occupancy = int(max_occupancy)
        self.track_entry_times = {}  # track_id -> timestamp (float)
        self.loitering_incidents = 0
        self._notified_loitering_tracks = set()

    @property
    def area(self) -> float:
        """Calculate polygon surface area in pixels using Green's theorem."""
        if len(self.points) < 3:
            return 0.0
        return float(cv2.contourArea(self.points))

    @property
    def centroid(self) -> tuple[int, int]:
        """Compute the spatial centroid of the polygon zone."""
        if len(self.points) == 0:
            return (0, 0)
        M = cv2.moments(self.points)
        if M["m00"] != 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return (int(np.mean(self.points[:, 0])), int(np.mean(self.points[:, 1])))

    def get_stats(self):
        return {
            "total_intrusions": self.total_intrusions,
            "active_count": len(self.active_intrusions),
            "max_occupancy": self.max_occupancy,
            "dwell_threshold_sec": self.dwell_threshold_sec,
            "loitering_incidents": self.loitering_incidents,
            "area_px": round(self.area, 1),
        }

    def reset_stats(self):
        self.total_intrusions = 0
        self.active_intrusions.clear()
        self.track_entry_times.clear()
        self.loitering_incidents = 0
        self._notified_loitering_tracks.clear()

    def get_dwell_time(self, track_id: int) -> float:
        """Returns dwell time in seconds for a track currently inside the zone."""
        if track_id in self.track_entry_times:
            return round(time.time() - self.track_entry_times[track_id], 2)
        return 0.0

    def check_intrusions(self, tracks: list):
        intrusions = self.update_intrusions(tracks)
        return [{"track_id": t.track_id, "track": t} for t in intrusions]

    def contains_point(self, point: tuple) -> bool:
        """
        Point in Polygon check: returns True if point is strictly inside or on edge.
        """
        if len(self.points) < 3:
            return False
        res = cv2.pointPolygonTest(self.points, (float(point[0]), float(point[1])), measureDist=False)
        return res >= 0

    def update_intrusions(self, tracks: list):
        """
        Evaluates which tracks are currently inside the zone.
        Updates dwell timers and returns newly triggered intrusion tracks.
        """
        new_intrusions = []
        currently_inside = set()
        now = time.time()

        for track in tracks:
            footprint = (int((track.bbox[0] + track.bbox[2]) / 2), int(track.bbox[3]))
            centroid = track.centroid

            if self.contains_point(footprint) or self.contains_point(centroid):
                currently_inside.add(track.track_id)
                track.current_zones.add(self.zone_id)
                
                # Track entry timestamp for dwell tracking
                if track.track_id not in self.track_entry_times:
                    self.track_entry_times[track.track_id] = now

                if track.track_id not in self.active_intrusions:
                    self.total_intrusions += 1
                    new_intrusions.append(track)
            else:
                track.current_zones.discard(self.zone_id)
                # Cleanup dwell timer if departed
                if track.track_id in self.track_entry_times:
                    del self.track_entry_times[track.track_id]
                self._notified_loitering_tracks.discard(track.track_id)

        # Cleanup tracks that were missed or purged
        purged = set(self.track_entry_times.keys()) - currently_inside
        for tid in purged:
            del self.track_entry_times[tid]
            self._notified_loitering_tracks.discard(tid)

        self.active_intrusions = currently_inside
        return new_intrusions

    def check_loitering(self, tracks: list) -> list[dict]:
        """
        Identifies tracks that have dwelled beyond the configured loitering threshold.
        """
        if self.dwell_threshold_sec <= 0:
            return []

        alerts = []
        for track in tracks:
            if track.track_id in self.active_intrusions:
                dwell = self.get_dwell_time(track.track_id)
                if dwell >= self.dwell_threshold_sec and track.track_id not in self._notified_loitering_tracks:
                    self.loitering_incidents += 1
                    self._notified_loitering_tracks.add(track.track_id)
                    alerts.append({
                        "zone_id": self.zone_id,
                        "label": self.label,
                        "track_id": track.track_id,
                        "cls_name": track.cls_name,
                        "dwell_sec": dwell,
                        "threshold_sec": self.dwell_threshold_sec,
                        "timestamp": round(time.time(), 2)
                    })
        return alerts

    def check_occupancy_breach(self) -> dict | None:
        """
        Checks if the current number of active occupants exceeds the maximum capacity threshold.
        """
        if self.max_occupancy <= 0:
            return None

        current_count = len(self.active_intrusions)
        if current_count > self.max_occupancy:
            return {
                "zone_id": self.zone_id,
                "label": self.label,
                "current_occupancy": current_count,
                "max_capacity": self.max_occupancy,
                "breach_count": current_count - self.max_occupancy,
                "timestamp": round(time.time(), 2)
            }
        return None


class VelocityEstimator:
    """
    Estimates 2D trajectory kinematics, velocity vectors, scalar speeds,
    and 8-way compass heading orientation.
    """
    COMPASS_SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    @staticmethod
    def calculate_velocity(pt_prev: tuple, pt_curr: tuple, dt_sec: float) -> tuple[float, float]:
        """Computes velocity vector (vx, vy) in pixels/sec."""
        if dt_sec <= 0:
            return (0.0, 0.0)
        vx = (pt_curr[0] - pt_prev[0]) / dt_sec
        vy = (pt_curr[1] - pt_prev[1]) / dt_sec
        return (round(float(vx), 2), round(float(vy), 2))

    @staticmethod
    def calculate_speed(vx: float, vy: float) -> float:
        """Computes scalar Euclidean speed from velocity components."""
        return round(float(np.hypot(vx, vy)), 2)

    @classmethod
    def calculate_heading(cls, pt_prev: tuple, pt_curr: tuple) -> tuple[float, str]:
        """
        Computes angular bearing (0-360 degrees) and cardinal compass heading.
        0° is North (up), 90° is East (right), 180° is South, 270° is West.
        """
        dx = pt_curr[0] - pt_prev[0]
        dy = pt_curr[1] - pt_prev[1]
        if dx == 0 and dy == 0:
            return (0.0, "STATIONARY")

        # In image coordinates, y increases downwards, so North is -dy
        angle_rad = np.arctan2(dx, -dy)
        angle_deg = (np.degrees(angle_rad) + 360.0) % 360.0

        idx = int((angle_deg + 22.5) // 45) % 8
        return (round(float(angle_deg), 1), cls.COMPASS_SECTORS[idx])

    @staticmethod
    def detect_anomaly(current_speed: float, baseline_speed: float, threshold_factor: float = 2.5) -> bool:
        """Detects sudden acceleration or high-speed anomaly relative to baseline."""
        if baseline_speed <= 0:
            return False
        return current_speed >= (baseline_speed * threshold_factor)


class HeatmapAccumulator:
    def __init__(self, width: int = 640, height: int = 480, decay_rate: float = 0.995):
        self.width = width
        self.height = height
        self.decay_rate = decay_rate
        self.heatmap = np.zeros((height, width), dtype=np.float32)

    def reset(self, width: int = None, height: int = None):
        if width and height:
            self.width = width
            self.height = height
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)

    def accumulate(self, points: list, radius: int = 15, intensity: float = 2.0):
        self.add_points(points, radius=radius, intensity=intensity)

    def add_points(self, points: list, radius: int = 15, intensity: float = 2.0):
        # Gradual decay of past heat
        self.heatmap *= self.decay_rate

        for (x, y) in points:
            if 0 <= x < self.width and 0 <= y < self.height:
                cv2.circle(self.heatmap, (int(x), int(y)), radius, intensity, -1)

    def get_color_overlay(self, alpha: float = 0.5):
        """
        Returns BGR color mapped heatmap ready to blend with frame.
        """
        if self.heatmap.max() > 0:
            norm_heat = np.clip(self.heatmap / (self.heatmap.max() + 1e-5) * 255, 0, 255).astype(np.uint8)
        else:
            norm_heat = np.zeros((self.height, self.width), dtype=np.uint8)

        # Apply Jet or Inferno colormap
        colored_heatmap = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
        
        # Mask out very low heat values to keep background visible
        mask = norm_heat < 20
        colored_heatmap[mask] = 0
        return colored_heatmap
