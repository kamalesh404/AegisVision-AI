"""
Advanced HUD Overlay Renderer — draws enterprise-grade visual overlays on video frames.
Includes: bounding boxes with confidence arcs, motion trails, tripwire lines,
geofence polygons, heatmap blending, and a telemetry HUD panel.
"""
import cv2
import numpy as np
import math

# ── Color palette (BGR) ──────────────────────────────────────────
PALETTE = [
    (255, 76, 76),    # Red
    (76, 255, 76),    # Green
    (76, 165, 255),   # Orange
    (255, 255, 76),   # Cyan
    (255, 76, 255),   # Magenta
    (76, 255, 255),   # Yellow
    (200, 130, 255),  # Pink
    (130, 255, 200),  # Mint
    (255, 200, 130),  # Light Blue
    (180, 180, 255),  # Light Red
    (255, 180, 180),  # Light Cyan
    (180, 255, 180),  # Light Green
]

def get_color(cls_id: int):
    return PALETTE[cls_id % len(PALETTE)]


class HUDRenderer:
    def __init__(self):
        self.frame_count = 0

    def draw_tracked_objects(self, frame, tracks, show_trails=True, show_confidence=True, show_id=True):
        """
        Draw bounding boxes with rounded corners, ID tags, confidence bars, and motion trails.
        """
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            color = get_color(track.cls_id)
            w = x2 - x1
            h = y2 - y1

            # ── Rounded corner box ──
            corner_len = min(20, w // 4, h // 4)
            t = 2
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, t)

            # Faint inner rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # ── ID + Class label ──
            if show_id:
                label = f"#{track.track_id} {track.cls_name}"
                if show_confidence:
                    label += f" {track.confidence:.0%}"

                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                tag_y = max(0, y1 - 6)
                cv2.rectangle(frame, (x1, tag_y - lh - 6), (x1 + lw + 8, tag_y), color, -1)
                cv2.putText(frame, label, (x1 + 4, tag_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # ── Confidence bar ──
            if show_confidence:
                bar_width = w
                bar_height = 3
                filled = int(bar_width * track.confidence)
                bar_y = y2 + 4
                cv2.rectangle(frame, (x1, bar_y), (x1 + bar_width, bar_y + bar_height), (60, 60, 60), -1)
                cv2.rectangle(frame, (x1, bar_y), (x1 + filled, bar_y + bar_height), color, -1)

            # ── Motion trails ──
            if show_trails and len(track.history) > 1:
                pts = list(track.history)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    thickness = max(1, int(alpha * 3))
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(frame, pts[i - 1], pts[i], trail_color, thickness, cv2.LINE_AA)

            # ── Centroid dot ──
            cx, cy = track.centroid
            cv2.circle(frame, (cx, cy), 4, color, -1)

        return frame

    def draw_tripwires(self, frame, tripwires):
        """
        Draws tripwire lines with directional arrows and IN/OUT counters.
        """
        for tw in tripwires:
            # Main line
            cv2.line(frame, tw.pt1, tw.pt2, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Direction arrows along the line
            mx = (tw.pt1[0] + tw.pt2[0]) // 2
            my = (tw.pt1[1] + tw.pt2[1]) // 2
            
            # Perpendicular direction indicators
            dx = tw.pt2[0] - tw.pt1[0]
            dy = tw.pt2[1] - tw.pt1[1]
            length = math.hypot(dx, dy)
            if length > 0:
                nx, ny = -dy / length, dx / length  # perpendicular normal
                arrow_len = 15
                # IN arrow
                in_tip = (int(mx + nx * arrow_len), int(my + ny * arrow_len))
                cv2.arrowedLine(frame, (mx, my), in_tip, (0, 200, 0), 2, tipLength=0.4)
                # OUT arrow
                out_tip = (int(mx - nx * arrow_len), int(my - ny * arrow_len))
                cv2.arrowedLine(frame, (mx, my), out_tip, (0, 100, 255), 2, tipLength=0.4)

            # Label and counters
            label = f"{tw.label}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lx = min(tw.pt1[0], tw.pt2[0])
            ly = min(tw.pt1[1], tw.pt2[1]) - 30

            cv2.rectangle(frame, (lx - 2, ly - lh - 4), (lx + lw + 6, ly + 4), (0, 0, 0), -1)
            cv2.putText(frame, label, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # IN/OUT counter badges
            in_text = f"IN: {tw.in_count}"
            out_text = f"OUT: {tw.out_count}"
            cv2.putText(frame, in_text, (lx + 2, ly + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, out_text, (lx + 80, ly + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)

        return frame

    def draw_geofences(self, frame, zones, tracks):
        """
        Draws geofence zones with transparent fill, border, and intrusion flash.
        """
        overlay = frame.copy()
        for zone in zones:
            # Fill polygon with transparency
            fill_color = zone.color
            cv2.fillPoly(overlay, [zone.points], fill_color)
            
            # Label
            centroid = zone.points.mean(axis=0).astype(int)
            (lw, lh), _ = cv2.getTextSize(zone.label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (centroid[0] - lw // 2 - 4, centroid[1] - lh - 6), 
                         (centroid[0] + lw // 2 + 4, centroid[1] + 4), (0, 0, 0), -1)
            cv2.putText(frame, zone.label, (centroid[0] - lw // 2, centroid[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Intrusion & capacity telemetry
            is_breached = zone.check_occupancy_breach() is not None
            if zone.max_occupancy > 0:
                count_text = f"Occ: {len(zone.active_intrusions)}/{zone.max_occupancy}"
                if is_breached:
                    count_text += " [BREACH]"
            else:
                count_text = f"Active: {len(zone.active_intrusions)}"
            
            if zone.dwell_threshold_sec > 0:
                count_text += f" | Limit: {int(zone.dwell_threshold_sec)}s"

            text_color = (0, 0, 255) if is_breached else (200, 200, 200)
            cv2.putText(frame, count_text, (centroid[0] - lw // 2, centroid[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

            # If breached or active intrusion, flash border
            if is_breached:
                border_color = (0, 0, 255)
                border_thickness = 4
            elif zone.active_intrusions:
                border_color = (0, 0, 255)
                border_thickness = 3
            else:
                border_color = fill_color
                border_thickness = 2
            cv2.polylines(frame, [zone.points], isClosed=True, color=border_color, thickness=border_thickness)

        # Blend fill
        alpha = 0.15
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_hud_telemetry(self, frame, fps, inference_ms, track_count, mode_name, model_label, 
                            tripwire_stats=None, zone_stats=None):
        """
        Draws the top-left HUD panel with system telemetry.
        """
        h, w = frame.shape[:2]
        
        # ── Top-left HUD panel ──
        panel_w = 320
        panel_h = 155
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (0, 255, 200), 1)

        # Title
        cv2.putText(frame, "AEGIS VISION AI", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)
        
        y = 48
        lines = [
            f"Mode: {mode_name}",
            f"Model: {model_label}",
            f"FPS: {fps:.1f}  |  Inference: {inference_ms}ms",
            f"Active Tracks: {track_count}",
        ]
        
        if tripwire_stats:
            total_in = sum(t.in_count for t in tripwire_stats)
            total_out = sum(t.out_count for t in tripwire_stats)
            lines.append(f"Crossings  IN: {total_in}  |  OUT: {total_out}")

        if zone_stats:
            total_intrusions = sum(z.total_intrusions for z in zone_stats)
            active = sum(len(z.active_intrusions) for z in zone_stats)
            lines.append(f"Zone Intrusions: {total_intrusions}  Active: {active}")

        for line in lines:
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            y += 18

        # ── Timestamp bottom-right ──
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (tw_text, th_text), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(frame, ts, (w - tw_text - 12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # ── Recording dot (blinking) ──
        self.frame_count += 1
        if (self.frame_count // 15) % 2 == 0:
            cv2.circle(frame, (w - tw_text - 28, h - 17), 5, (0, 0, 255), -1)

        return frame

    def draw_heatmap(self, frame, heatmap_acc, alpha=0.4):
        overlay = heatmap_acc.get_color_overlay(alpha=alpha)
        return self.blend_heatmap(frame, overlay, alpha=alpha)

    def draw_zones(self, frame, zones, tracks=None):
        return self.draw_geofences(frame, zones, tracks or [])

    def draw_tracks(self, frame, tracks, show_ids=True, show_confidence=True, show_trails=True):
        return self.draw_tracked_objects(frame, tracks, show_trails=show_trails, show_confidence=show_confidence, show_id=show_ids)

    def draw_hud(self, frame, fps, mode, track_count, tripwire_stats=None, zone_stats=None, model_name="YOLOv8n", inference_ms=0.0):
        tw_list = tripwire_stats if isinstance(tripwire_stats, list) else None
        z_list = zone_stats if isinstance(zone_stats, list) else None
        return self.draw_hud_telemetry(frame, fps, inference_ms, track_count, str(mode), str(model_name), tw_list, z_list)

    def blend_heatmap(self, frame, heatmap_overlay, alpha=0.4):
        """
        Blend heatmap overlay onto frame.
        """
        if heatmap_overlay is not None and heatmap_overlay.shape[:2] == frame.shape[:2]:
            mask = np.any(heatmap_overlay > 0, axis=2)
            blended = frame.copy()
            blended[mask] = cv2.addWeighted(frame, 1 - alpha, heatmap_overlay, alpha, 0)[mask]
            return blended
        return frame
