"""
Industry Mode Presets — pre-configured analytics profiles.
Each mode defines which COCO classes to detect, tripwire defaults,
zone behavior, and UI theme hints for the dashboard.
"""

COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    39: 'bottle', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 73: 'book', 74: 'clock',
}

MODE_PRESETS = {
    "safety": {
        "name": "🛡️ Industrial Safety & PPE Compliance",
        "description": "Monitor worker presence in hazardous zones, enforce PPE compliance areas, detect personnel near heavy machinery.",
        "detect_classes": [0],  # person
        "confidence_threshold": 0.45,
        "tracker_max_distance": 70,
        "tracker_max_missed": 20,
        "enable_tripwires": True,
        "enable_geofencing": True,
        "enable_heatmap": True,
        "enable_trails": True,
        "default_tripwires": [
            {"id": "safety_line_1", "label": "Hazard Boundary", "pt1_pct": [0.1, 0.5], "pt2_pct": [0.9, 0.5]}
        ],
        "default_zones": [
            {"id": "danger_zone_1", "label": "⚠️ Danger Zone", "polygon_pct": [[0.6, 0.3], [0.95, 0.3], [0.95, 0.85], [0.6, 0.85]], "color": [0, 0, 255], "dwell_threshold_sec": 4.0, "max_occupancy": 1}
        ],
        "violation_severity": "CRITICAL",
        "theme_accent": "#FF4444",
        "icon": "🛡️"
    },
    "traffic": {
        "name": "🚦 Smart Traffic & Fleet Analytics",
        "description": "Bi-directional vehicle counting, fleet classification (car/truck/bus/motorcycle), speed estimation via motion vectors.",
        "detect_classes": [0, 1, 2, 3, 5, 7],  # person, bicycle, car, motorcycle, bus, truck
        "confidence_threshold": 0.40,
        "tracker_max_distance": 80,
        "tracker_max_missed": 25,
        "enable_tripwires": True,
        "enable_geofencing": False,
        "enable_heatmap": True,
        "enable_trails": True,
        "default_tripwires": [
            {"id": "count_line_1", "label": "📊 Counting Line A", "pt1_pct": [0.05, 0.55], "pt2_pct": [0.95, 0.55]},
            {"id": "count_line_2", "label": "📊 Counting Line B", "pt1_pct": [0.05, 0.75], "pt2_pct": [0.95, 0.75]}
        ],
        "default_zones": [],
        "violation_severity": "INFO",
        "theme_accent": "#00C853",
        "icon": "🚦"
    },
    "retail": {
        "name": "👥 Retail & Crowd Spatial Intelligence",
        "description": "Customer foot-traffic analytics, dwell time heatmaps, zone occupancy monitoring for retail optimization.",
        "detect_classes": [0],  # person
        "confidence_threshold": 0.35,
        "tracker_max_distance": 60,
        "tracker_max_missed": 30,
        "enable_tripwires": True,
        "enable_geofencing": True,
        "enable_heatmap": True,
        "enable_trails": True,
        "default_tripwires": [
            {"id": "entrance_line", "label": "🚪 Store Entrance", "pt1_pct": [0.3, 0.1], "pt2_pct": [0.7, 0.1]}
        ],
        "default_zones": [
            {"id": "checkout_zone", "label": "💳 Checkout Area", "polygon_pct": [[0.05, 0.6], [0.4, 0.6], [0.4, 0.95], [0.05, 0.95]], "color": [255, 165, 0], "dwell_threshold_sec": 12.0, "max_occupancy": 3},
            {"id": "display_zone", "label": "🏷️ Display Zone", "polygon_pct": [[0.5, 0.2], [0.9, 0.2], [0.9, 0.55], [0.5, 0.55]], "color": [0, 200, 255], "dwell_threshold_sec": 8.0, "max_occupancy": 5}
        ],
        "violation_severity": "INFO",
        "theme_accent": "#2196F3",
        "icon": "👥"
    },
    "security": {
        "name": "🚨 Perimeter Defense & Intrusion Detection",
        "description": "Interactive polygon geofencing with real-time visual intrusion alerts, perimeter breach logging, and snapshot evidence.",
        "detect_classes": [0, 1, 2, 3, 5, 7, 15, 16],  # person, vehicles, cat, dog
        "confidence_threshold": 0.45,
        "tracker_max_distance": 75,
        "tracker_max_missed": 15,
        "enable_tripwires": True,
        "enable_geofencing": True,
        "enable_heatmap": False,
        "enable_trails": True,
        "default_tripwires": [
            {"id": "perimeter_line", "label": "🔒 Perimeter Line", "pt1_pct": [0.0, 0.45], "pt2_pct": [1.0, 0.45]}
        ],
        "default_zones": [
            {"id": "restricted_1", "label": "🚫 Restricted Area", "polygon_pct": [[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]], "color": [0, 0, 255], "dwell_threshold_sec": 5.0, "max_occupancy": 0}
        ],
        "violation_severity": "CRITICAL",
        "theme_accent": "#FF1744",
        "icon": "🚨"
    }
}

def get_mode(mode_name: str) -> dict:
    return MODE_PRESETS.get(mode_name, MODE_PRESETS["traffic"])

def list_modes():
    return [{"key": k, "name": v["name"], "description": v["description"], "icon": v["icon"]} for k, v in MODE_PRESETS.items()]
