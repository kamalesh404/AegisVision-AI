"""
Demo Video Simulator:
Generates a realistic traffic & surveillance scene with moving vehicles and pedestrians.
Guarantees out-of-the-box demonstration even without an active webcam.
"""
import cv2
import numpy as np
import math
import time

class SyntheticFeedGenerator:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame_idx = 0
        
        # Simulated entities: [x, y, dx, dy, width, height, color, class_name]
        self.entities = [
            # Cars moving across lanes
            {"x": -80, "y": 280, "dx": 4.5, "dy": 0.0, "w": 110, "h": 55, "color": (40, 180, 240), "cls": "car"},
            {"x": 680, "y": 360, "dx": -5.0, "dy": 0.0, "w": 120, "h": 60, "color": (180, 50, 40), "cls": "car"},
            {"x": -250, "y": 270, "dx": 3.8, "dy": 0.0, "w": 180, "h": 75, "color": (60, 140, 60), "cls": "truck"},
            # Pedestrians crossing
            {"x": 220, "y": 140, "dx": 0.0, "dy": 2.2, "w": 28, "h": 65, "color": (220, 220, 100), "cls": "person"},
            {"x": 420, "y": 460, "dx": 0.0, "dy": -2.0, "w": 28, "h": 65, "color": (200, 100, 220), "cls": "person"},
            {"x": 520, "y": 180, "dx": -0.8, "dy": 1.5, "w": 26, "h": 60, "color": (100, 220, 200), "cls": "person"},
        ]

    def get_frame(self):
        self.frame_idx += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. Background scene: Road & sidewalk
        frame[0:220, :] = [45, 45, 50]       # Upper sidewalk / plaza
        frame[220:440, :] = [30, 30, 32]     # Main asphalt road
        frame[440:self.height, :] = [45, 45, 50] # Lower sidewalk

        # Road dividing lane markings
        dash_offset = int((self.frame_idx * 4) % 60)
        for x in range(-60 + dash_offset, self.width + 60, 60):
            cv2.line(frame, (x, 330), (x + 30, 330), (180, 180, 180), 2)

        # Crosswalk zebra stripes
        for x in range(200, 270, 16):
            cv2.rectangle(frame, (x, 220), (x + 8, 440), (120, 120, 120), -1)

        # 2. Draw moving entities with pseudo-realistic geometry
        for ent in self.entities:
            ent["x"] += ent["dx"]
            ent["y"] += ent["dy"]

            # Wrap around boundaries
            if ent["dx"] > 0 and ent["x"] > self.width + 100:
                ent["x"] = -150
            elif ent["dx"] < 0 and ent["x"] < -150:
                ent["x"] = self.width + 100
            if ent["dy"] > 0 and ent["y"] > self.height + 60:
                ent["y"] = 120
            elif ent["dy"] < 0 and ent["y"] < 120:
                ent["y"] = self.height + 60

            ix = int(ent["x"])
            iy = int(ent["y"])
            w = ent["w"]
            h = ent["h"]

            # Draw vehicle or person
            if ent["cls"] in ("car", "truck"):
                # Vehicle body
                cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), ent["color"], -1)
                # Windshield / Cabin
                cabin_w = int(w * 0.5)
                cabin_x = ix + int(w * 0.25)
                cv2.rectangle(frame, (cabin_x, iy + 6), (cabin_x + cabin_w, iy + h - 6), (220, 240, 255), -1)
                # Wheels
                cv2.circle(frame, (ix + 18, iy + h + 2), 7, (20, 20, 20), -1)
                cv2.circle(frame, (ix + w - 18, iy + h + 2), 7, (20, 20, 20), -1)
                # Headlights
                hl_x = ix + w - 2 if ent["dx"] > 0 else ix + 2
                cv2.circle(frame, (hl_x, iy + 12), 4, (100, 255, 255), -1)
                cv2.circle(frame, (hl_x, iy + h - 12), 4, (100, 255, 255), -1)
            else:
                # Pedestrian body
                # Head
                cv2.circle(frame, (ix + w // 2, iy + 10), 9, (210, 180, 160), -1)
                # Torso
                cv2.rectangle(frame, (ix + 4, iy + 18), (ix + w - 4, iy + 45), ent["color"], -1)
                # Legs
                leg_swing = int(math.sin(self.frame_idx * 0.25) * 5)
                cv2.line(frame, (ix + 8, iy + 45), (ix + 6 + leg_swing, iy + h), (50, 50, 80), 3)
                cv2.line(frame, (ix + w - 8, iy + 45), (ix + w - 6 - leg_swing, iy + h), (50, 50, 80), 3)

        # Simulation watermark banner
        cv2.putText(frame, "[SIMULATED LIVE FEED - ULTRA-HD SYNTHETIC TEST STREAM]", 
                    (15, self.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

        return frame
