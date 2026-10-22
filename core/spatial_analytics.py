"""
Spatial Analytics Engine:
- Bi-directional Line Crossing / Tripwires (IN vs OUT counting)
- Polygon Geofencing / Intrusion Zone Detection
- Real-time Spatial Heatmap Generation
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
    def __init__(self, points: list = None, label: str = "Restricted Zone", zone_type: str = "intrusion", zone_id: str = None, color=(0, 0, 255), polygon_points: list = None):
        self.zone_id = zone_id or f"zone_{int(time.time()*1000)%100000}"
        raw_pts = points if points is not None else (polygon_points if polygon_points is not None else [])
        self.points = np.array(raw_pts, dtype=np.int32)
        self.label = label
        self.zone_type = zone_type
        self.color = color  # BGR
        self.active_intrusions = set()
        self.total_intrusions = 0

    def get_stats(self):
        return {"total_intrusions": self.total_intrusions, "active_count": len(self.active_intrusions)}

    def reset_stats(self):
        self.total_intrusions = 0
        self.active_intrusions.clear()

    def check_intrusions(self, tracks: list):
        intrusions = self.update_intrusions(tracks)
        return [{"track_id": t.track_id, "track": t} for t in intrusions]

    def contains_point(self, point: tuple) -> bool:
        """
        Point in Polygon check: returns True if point is strictly inside or on edge.
        """
        # cv2.pointPolygonTest returns positive for inside, 0 on edge, negative outside
        res = cv2.pointPolygonTest(self.points, (float(point[0]), float(point[1])), measureDist=False)
        return res >= 0

    def update_intrusions(self, tracks: list):
        """
        Evaluates which tracks are currently inside the zone.
        Returns newly triggered intrusion tracks.
        """
        new_intrusions = []
        currently_inside = set()

        for track in tracks:
            # Check bottom center of bounding box (footprint) or centroid
            footprint = (int((track.bbox[0] + track.bbox[2]) / 2), int(track.bbox[3]))
            centroid = track.centroid

            if self.contains_point(footprint) or self.contains_point(centroid):
                currently_inside.add(track.track_id)
                track.current_zones.add(self.zone_id)
                
                if track.track_id not in self.active_intrusions:
                    self.total_intrusions += 1
                    new_intrusions.append(track)
            else:
                track.current_zones.discard(self.zone_id)

        self.active_intrusions = currently_inside
        return new_intrusions


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
