# 🛡️ AegisVision AI — Enterprise Real-Time Spatial & Video Analytics

> **Industry-grade, competition-winning Computer Vision platform powered by YOLOv8 + FastAPI + Real-Time Multi-Object Tracking**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-5.x-5C3EE8?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏆 What Makes This Project Competition-Grade

| Feature | Description |
|---------|-------------|
| 🔄 **Real-Time MOT** | Centroid + IoU multi-object tracker with persistent ID assignment and motion trails |
| 📐 **Spatial Math** | Ray-segment intersection for bi-directional tripwire counting; Point-in-Polygon for geofence intrusion |
| 🧠 **Dynamic Model Switching** | Hot-swap between YOLOv8n/s/m/l/x without restart |
| 🎯 **4 Industry Modes** | Safety, Traffic, Retail, Security — each with pre-configured analytics profiles |
| 🔥 **Live Heatmaps** | Temporal decay heatmap accumulator with JET colormap overlay |
| 🚨 **Incident Management** | Auto-snapshot evidence capture, severity classification, CSV audit trail |
| 🖥️ **Control Room Dashboard** | Dark glassmorphism UI with real-time charts, stat cards, and incident table |
| ⌨️ **Keyboard Shortcuts** | Professional hotkeys (Space=pause, S=snapshot, H=heatmap, 1-4=modes) |

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────┐
│                   FastAPI Server                      │
│                                                      │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │ YOLO     │→ │ Centroid  │→ │ Spatial Engine   │  │
│  │ Engine   │  │ IoU       │  │ ├─ Tripwires     │  │
│  │ (detect) │  │ Tracker   │  │ ├─ Geofences     │  │
│  │          │  │ (MOT)     │  │ └─ Heatmap       │  │
│  └──────────┘  └───────────┘  └──────────────────┘  │
│       │              │               │               │
│       └──────────────┴───────────────┘               │
│                      │                               │
│              ┌───────┴────────┐                      │
│              │  HUD Renderer  │                      │
│              │  + Violation   │                      │
│              │    Manager     │                      │
│              └───────┬────────┘                      │
│                      │                               │
│              ┌───────┴────────┐                      │
│              │  MJPEG Stream  │ → Browser Dashboard  │
│              └────────────────┘                      │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python app.py
```

### 3. Open Dashboard
Navigate to **http://localhost:8000** in your browser.

---

## 🎮 Industry Modes

### 🛡️ Industrial Safety & PPE Compliance
- Detects workers (persons) near hazardous zones
- Geofenced danger areas with real-time intrusion alerts
- Auto-captures violation snapshots as evidence

### 🚦 Smart Traffic & Fleet Analytics
- Bi-directional vehicle counting (IN/OUT) via tripwire lines
- Vehicle classification: car, truck, bus, motorcycle, bicycle
- Motion vector trails for flow visualization

### 👥 Retail & Crowd Spatial Intelligence
- Customer foot-traffic counting at store entrances
- Dwell-time heatmaps for zone occupancy analysis
- Checkout and display area monitoring

### 🚨 Perimeter Defense & Intrusion Detection
- Interactive polygon geofencing for restricted areas
- Person + vehicle detection with perimeter breach alerts
- Snapshot evidence gallery with CSV audit log

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause / Resume |
| `S` | Download Snapshot |
| `R` | Reset All Counters |
| `H` | Toggle Heatmap |
| `T` | Toggle Motion Trails |
| `1` | Safety Mode |
| `2` | Traffic Mode |
| `3` | Retail Mode |
| `4` | Security Mode |

---

## 📁 Project Structure

```
yolo-vision-ai/
├── app.py                      # FastAPI application (REST + MJPEG stream)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── core/
│   ├── __init__.py
│   ├── detector.py             # YOLO inference engine wrapper
│   ├── tracker.py              # Centroid + IoU multi-object tracker
│   ├── spatial_analytics.py    # Tripwires, geofences, heatmaps
│   ├── violation_manager.py    # Incident & alert management
│   ├── renderer.py             # HUD overlay & visualization renderer
│   └── modes/
│       ├── __init__.py
│       └── presets.py          # Industry mode configurations
├── static/
│   ├── css/
│   │   └── style.css           # Dark control room theme
│   └── js/
│       └── app.js              # Dashboard frontend logic
├── templates/
│   └── index.html              # Control room dashboard template
├── alerts/                     # Auto-generated incident snapshots & CSV audit log
└── samples/                    # Sample videos for testing
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard UI |
| `GET` | `/video_feed` | MJPEG video stream |
| `GET` | `/api/status` | System status & metrics |
| `GET` | `/api/events` | Recent incident log |
| `GET` | `/api/modes` | Available mode presets |
| `GET` | `/api/snapshot` | Download current frame |
| `POST` | `/api/mode` | Switch analytics mode |
| `POST` | `/api/model` | Switch YOLO model |
| `POST` | `/api/source` | Change video source |
| `POST` | `/api/config` | Update display config |
| `POST` | `/api/reset_counters` | Reset all counters |

---

## 📊 Technical Highlights

### Multi-Object Tracking Algorithm
- **Centroid + IoU hybrid** cost matrix with class-aware penalty
- **Greedy assignment** with distance threshold gating
- **Persistent track IDs** with configurable miss tolerance
- **Motion trail** ring buffer (30-frame history per track)

### Spatial Analytics Math
- **Line-crossing detection**: CCW-based segment intersection test + 2D cross product for IN/OUT direction
- **Zone intrusion**: OpenCV `pointPolygonTest` on both footprint (bottom-center) and centroid
- **Heatmap**: Gaussian-accumulated temporal decay with JET colormap

### Performance
- YOLOv8n: ~25-40 FPS on CPU (640px inference)
- YOLOv8s: ~15-25 FPS on CPU
- GPU acceleration: automatic when CUDA available

---

## 📜 License

MIT License — free for commercial and academic use.

---

*Built with ❤️ 
