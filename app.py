"""
AegisVision AI — Enterprise Real-Time YOLO Video Analytics Platform
Main FastAPI application: MJPEG streaming, REST API, configuration, state management.
"""

import time
import threading
import logging
from pathlib import Path
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.detector import YOLOEngine
from core.tracker import CentroidIoUTracker
from core.spatial_analytics import TripwireLine, GeofenceZone, HeatmapAccumulator
from core.violation_manager import IncidentManager
from core.renderer import HUDRenderer
from core.modes.presets import get_mode, list_modes, MODE_PRESETS
from core.simulator import SyntheticFeedGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("aegisvision")

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
ALERTS_DIR = BASE_DIR / "alerts"
ALERTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class ModeRequest(BaseModel):
    mode: str


class ModelRequest(BaseModel):
    model: str


class SourceRequest(BaseModel):
    source: str


class ConfigRequest(BaseModel):
    confidence: Optional[float] = None
    show_trails: Optional[bool] = None
    show_heatmap: Optional[bool] = None
    show_ids: Optional[bool] = None
    show_confidence: Optional[bool] = None
    paused: Optional[bool] = None


class ZoneAddRequest(BaseModel):
    label: str = "Custom Restricted Zone"
    points: list[list[int]]
    color: Optional[list[int]] = [0, 0, 255]


class TripwireAddRequest(BaseModel):
    label: str = "Custom Tripwire"
    pt1: list[int]
    pt2: list[int]



# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    """Centralised mutable state for the entire analytics platform."""

    def __init__(self) -> None:
        self.engine: YOLOEngine = YOLOEngine(model_key="yolov8n", device="cpu")
        self.tracker: CentroidIoUTracker = CentroidIoUTracker()
        self.renderer: HUDRenderer = HUDRenderer()
        self.incident_mgr: IncidentManager = IncidentManager(alerts_dir=str(ALERTS_DIR))
        self.heatmap: HeatmapAccumulator = HeatmapAccumulator(width=640, height=480)
        self.simulator: SyntheticFeedGenerator = SyntheticFeedGenerator(width=640, height=480)
        self.tripwires: list[TripwireLine] = []
        self.zones: list[GeofenceZone] = []

        # Mode / video source
        self.current_mode: str = "traffic"
        self.video_source: object = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.cap_lock: threading.Lock = threading.Lock()

        # Config toggles
        self.show_trails: bool = True
        self.show_heatmap: bool = True
        self.show_ids: bool = True
        self.show_confidence: bool = True
        self.paused: bool = False

        # Metrics
        self.fps: float = 30.0
        self.frame_width: int = 640
        self.frame_height: int = 480
        self.total_frames: int = 0
        self.last_frame: Optional[np.ndarray] = None
        self._frame_times: deque = deque(maxlen=30)
        self.demo_mode: bool = False

    def open_source(self, source: object = None) -> bool:
        """Open video source (webcam index, file path, RTSP stream, or 'demo')."""
        if source is not None:
            self.video_source = source

        with self.cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            src_str = str(self.video_source).strip().lower()
            if src_str in ("demo", "sim", "simulation", "test"):
                self.demo_mode = True
                self.frame_width = 640
                self.frame_height = 480
                logger.info("Activated high-fidelity synthetic demo stream.")
                return True

            src = self.video_source
            try:
                src = int(src)
            except (ValueError, TypeError):
                pass

            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                logger.warning("Failed to open source %s — falling back to synthetic simulation mode.", src)
                self.demo_mode = True
                self.frame_width = 640
                self.frame_height = 480
                return False

            self.cap = cap
            self.demo_mode = False
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                self.frame_width = w
                self.frame_height = h
            logger.info("Opened video source %s (%dx%d)", src, self.frame_width, self.frame_height)
            return True

    def read_frame(self) -> np.ndarray:
        """Read frame from camera/video or return procedural simulation frame."""
        if self.demo_mode:
            return self.simulator.get_frame()

        with self.cap_lock:
            if self.cap is None or not self.cap.isOpened():
                return self.simulator.get_frame()

            ret, frame = self.cap.read()
            if not ret:
                # Loop video file if reaching EOF
                src = self.video_source
                is_num = False
                try:
                    int(src)
                    is_num = True
                except (ValueError, TypeError):
                    pass

                if not is_num and isinstance(src, str) and not src.startswith("rtsp"):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

                if not ret:
                    return self.simulator.get_frame()

            return frame

    def apply_mode(self, mode_name: str) -> None:
        """Switch analytics profile: reconfigures tripwires, zones, and tracking."""
        preset = get_mode(mode_name)
        if not preset:
            raise ValueError(f"Unknown mode: {mode_name}")

        self.current_mode = mode_name
        self.tracker.reset()
        self.tripwires.clear()
        self.zones.clear()
        self.heatmap.reset(width=self.frame_width, height=self.frame_height)

        w, h = self.frame_width, self.frame_height
        
        # Tripwires
        tw_defs = preset.get("default_tripwires") or preset.get("tripwires", [])
        for tw in tw_defs:
            if "pt1_pct" in tw and "pt2_pct" in tw:
                p1 = (int(tw["pt1_pct"][0] * w), int(tw["pt1_pct"][1] * h))
                p2 = (int(tw["pt2_pct"][0] * w), int(tw["pt2_pct"][1] * h))
            else:
                p1 = (int(tw.get("x1", 0.05) * w), int(tw.get("y1", 0.5) * h))
                p2 = (int(tw.get("x2", 0.95) * w), int(tw.get("y2", 0.5) * h))
            
            self.tripwires.append(
                TripwireLine(
                    line_id=tw.get("id", f"tw_{len(self.tripwires)+1}"),
                    pt1=p1,
                    pt2=p2,
                    label=tw.get("label", "Tripwire")
                )
            )

        # Zones
        zone_defs = preset.get("default_zones") or preset.get("zones", [])
        for zd in zone_defs:
            pts_pct = zd.get("polygon_pct") or zd.get("points", [])
            pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in pts_pct]
            if pixel_pts:
                color = tuple(zd.get("color", [0, 0, 255]))
                self.zones.append(
                    GeofenceZone(
                        zone_id=zd.get("id", f"zone_{len(self.zones)+1}"),
                        polygon_points=pixel_pts,
                        label=zd.get("label", "Restricted Area"),
                        color=color
                    )
                )

        logger.info("Applied mode '%s' (%d tripwires, %d zones)", mode_name, len(self.tripwires), len(self.zones))

    def update_fps(self) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                self.fps = round((len(self._frame_times) - 1) / dt, 1)


# Global singleton application state
state = AppState()

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Initializing AegisVision AI Enterprise Platform...")
    # Attempt opening camera; if unavailable, seamlessly activate simulator
    state.open_source(state.video_source)
    state.apply_mode(state.current_mode)
    yield
    logger.info("Shutting down AegisVision AI...")
    with state.cap_lock:
        if state.cap is not None:
            state.cap.release()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AegisVision AI",
    description="Enterprise Real-Time YOLO Video Analytics & Spatial Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ALERTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/alerts", StaticFiles(directory=str(ALERTS_DIR)), name="alerts")

TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Video Pipeline Generator
# ---------------------------------------------------------------------------

def _pipeline_generate():
    while True:
        if state.paused:
            if state.last_frame is not None:
                _, buf = cv2.imencode(".jpg", state.last_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.04)
            continue

        # 1. Grab Frame
        frame = state.read_frame()
        h, w = frame.shape[:2]
        if w != state.frame_width or h != state.frame_height:
            state.frame_width = w
            state.frame_height = h
            try:
                state.apply_mode(state.current_mode)
            except Exception:
                pass

        # 2. YOLO Object Detection
        preset = get_mode(state.current_mode) or {}
        target_classes = preset.get("detect_classes") or preset.get("classes", None)
        conf_thresh = preset.get("confidence_threshold") or preset.get("confidence", 0.40)

        raw_detections = state.engine.detect(
            frame,
            conf_threshold=conf_thresh,
            target_classes=target_classes,
            imgsz=640
        )

        # In simulation mode, complement YOLO detections with simulator entities for 100% reliable tracking
        if state.demo_mode and len(raw_detections) == 0:
            for ent in state.simulator.entities:
                ix = int(ent["x"])
                iy = int(ent["y"])
                if 0 <= ix < state.frame_width and 0 <= iy < state.frame_height:
                    raw_detections.append({
                        "bbox": [ix, iy, ix + ent["w"], iy + ent["h"]],
                        "cls_id": 0 if ent["cls"] == "person" else 2,
                        "cls_name": ent["cls"],
                        "conf": 0.88
                    })

        # 3. Multi-Object Tracking (Centroid + IoU with Track IDs & Trails)
        tracks = state.tracker.update(raw_detections)

        # 4. Spatial Tripwire Crossing Analysis
        for tw in state.tripwires:
            crossing_events = tw.check_crossings(tracks)
            for ev in crossing_events:
                # Record incident
                state.incident_mgr.record(
                    event_type=f"TRIPWIRE_CROSSING",
                    label=tw.label,
                    track_id=ev.get("track_id"),
                    direction=ev.get("direction", ""),
                    frame=frame,
                    severity="INFO"
                )

        # 5. Geofence Zone Intrusions
        for zone in state.zones:
            intrusions = zone.update_intrusions(tracks)
            for track in intrusions:
                severity = preset.get("violation_severity", "CRITICAL")
                state.incident_mgr.record_incident(
                    event_type=f"RESTRICTED_ZONE_BREACH",
                    severity=severity,
                    track=track,
                    zone_or_line=zone.label,
                    frame=frame
                )

        # 6. Spatial Heatmap Accumulator
        if state.show_heatmap:
            centroids = [t.centroid for t in tracks]
            if centroids:
                state.heatmap.accumulate(centroids)

        # 7. Render Overlays
        rendered = frame.copy()

        # Heatmap overlay
        if state.show_heatmap:
            rendered = state.renderer.draw_heatmap(rendered, state.heatmap, alpha=0.35)

        # Geofence zones
        rendered = state.renderer.draw_zones(rendered, state.zones, tracks)

        # Tripwire lines
        rendered = state.renderer.draw_tripwires(rendered, state.tripwires)

        # Bounding boxes, IDs, confidence bars, and motion trails
        rendered = state.renderer.draw_tracks(
            rendered,
            tracks,
            show_ids=state.show_ids,
            show_confidence=state.show_confidence,
            show_trails=state.show_trails
        )

        # Telemetry HUD
        rendered = state.renderer.draw_hud_telemetry(
            rendered,
            fps=state.fps,
            inference_ms=state.engine.last_inference_ms,
            track_count=len(tracks),
            mode_name=preset.get("name", state.current_mode),
            model_label=state.engine.model_name,
            tripwire_stats=state.tripwires,
            zone_stats=state.zones
        )

        state.last_frame = rendered
        state.total_frames += 1
        state.update_fps()

        # JPEG encode and stream
        _, buf = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _pipeline_generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "modes": list_modes(),
            "current_mode": state.current_mode,
        }
    )


@app.get("/api/status")
async def api_status():
    tw_stats = [{"label": tw.label, "in_count": tw.in_count, "out_count": tw.out_count} for tw in state.tripwires]
    z_stats = [{"label": z.label, "total_intrusions": z.total_intrusions, "active_count": len(z.active_intrusions)} for z in state.zones]

    class_counts = {}
    for t in state.tracker.tracks.values():
        class_counts[t.cls_name] = class_counts.get(t.cls_name, 0) + 1

    return JSONResponse({
        "mode": state.current_mode,
        "fps": state.fps,
        "track_count": len(state.tracker.tracks),
        "inference_ms": state.engine.last_inference_ms,
        "model": state.engine.model_name,
        "model_info": state.engine.get_model_info(),
        "frame_size": {"width": state.frame_width, "height": state.frame_height},
        "total_frames": state.total_frames,
        "tripwire_stats": tw_stats,
        "zone_stats": z_stats,
        "track_count_by_class": class_counts,
        "recent_events": state.incident_mgr.recent(limit=30),
        "config": {
            "show_trails": state.show_trails,
            "show_heatmap": state.show_heatmap,
            "show_ids": state.show_ids,
            "show_confidence": state.show_confidence,
            "paused": state.paused,
        },
        "demo_mode": state.demo_mode,
        "source": str(state.video_source),
    })


@app.post("/api/mode")
async def api_set_mode(body: ModeRequest):
    mode_name = body.mode.lower().strip()
    available_keys = list(MODE_PRESETS.keys())
    if mode_name not in available_keys:
        # Match against prefix or name
        matched = None
        for k in available_keys:
            if k in mode_name or mode_name in k:
                matched = k
                break
        if not matched:
            raise HTTPException(status_code=400, detail=f"Unknown mode '{mode_name}'. Available: {available_keys}")
        mode_name = matched

    try:
        state.apply_mode(mode_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"status": "ok", "mode": state.current_mode})


@app.post("/api/model")
async def api_set_model(body: ModelRequest):
    model_name = body.model.strip()
    try:
        state.engine.load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model '{model_name}': {e}")
    return JSONResponse({"status": "ok", "model": state.engine.model_name})


@app.post("/api/source")
async def api_set_source(body: SourceRequest):
    source = body.source.strip()
    opened = state.open_source(source)
    try:
        state.apply_mode(state.current_mode)
    except Exception:
        pass
    return JSONResponse({
        "status": "ok" if opened else "failed",
        "source": str(state.video_source),
        "demo_mode": state.demo_mode,
    })


@app.post("/api/config")
async def api_update_config(body: ConfigRequest):
    if body.confidence is not None:
        p = get_mode(state.current_mode)
        if p:
            p["confidence_threshold"] = max(0.05, min(1.0, body.confidence))
    if body.show_trails is not None:
        state.show_trails = body.show_trails
    if body.show_heatmap is not None:
        state.show_heatmap = body.show_heatmap
    if body.show_ids is not None:
        state.show_ids = body.show_ids
    if body.show_confidence is not None:
        state.show_confidence = body.show_confidence
    if body.paused is not None:
        state.paused = body.paused

    return JSONResponse({
        "status": "ok",
        "config": {
            "show_trails": state.show_trails,
            "show_heatmap": state.show_heatmap,
            "show_ids": state.show_ids,
            "show_confidence": state.show_confidence,
            "paused": state.paused,
        },
    })


@app.get("/api/events")
async def api_events():
    return JSONResponse({"events": state.incident_mgr.recent(limit=50)})


@app.post("/api/reset_counters")
async def api_reset_counters():
    for tw in state.tripwires:
        tw.reset_counts()
    for z in state.zones:
        z.reset_stats()
    state.incident_mgr.clear()
    state.heatmap.reset(width=state.frame_width, height=state.frame_height)
    return JSONResponse({"status": "ok"})


@app.get("/api/modes")
async def api_list_modes():
    return JSONResponse({
        "modes": list_modes(),
        "current": state.current_mode,
    })


@app.get("/api/snapshot")
async def api_snapshot():
    target_frame = state.last_frame if state.last_frame is not None else state.simulator.get_frame()
    _, buf = cv2.imencode(".jpg", target_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="aegisvision_snapshot_{int(time.time())}.jpg"'
        },
    )


@app.get("/api/zones")
async def api_get_zones():
    """Return all active geofence zones."""
    return JSONResponse({
        "zones": [
            {
                "id": z.zone_id,
                "label": z.label,
                "points": z.points.tolist(),
                "total_intrusions": z.total_intrusions,
                "active_intrusions": len(z.active_intrusions),
            }
            for z in state.zones
        ]
    })


@app.post("/api/zones")
async def api_add_zone(body: ZoneAddRequest):
    """Add a new custom polygon geofence zone."""
    if len(body.points) < 3:
        raise HTTPException(status_code=400, detail="Polygon must have at least 3 points.")
    zone_id = f"custom_zone_{int(time.time() * 1000) % 100000}"
    new_zone = GeofenceZone(
        zone_id=zone_id,
        polygon_points=body.points,
        label=body.label,
        color=tuple(body.color or [0, 0, 255]),
    )
    state.zones.append(new_zone)
    return JSONResponse({"status": "ok", "zone_id": zone_id, "label": new_zone.label})


@app.delete("/api/zones/{zone_id}")
async def api_delete_zone(zone_id: str):
    """Remove a geofence zone by ID."""
    state.zones = [z for z in state.zones if z.zone_id != zone_id]
    return JSONResponse({"status": "ok", "deleted": zone_id})


@app.get("/api/tripwires")
async def api_get_tripwires():
    """Return all active tripwire lines."""
    return JSONResponse({
        "tripwires": [
            {
                "id": tw.line_id,
                "label": tw.label,
                "pt1": list(tw.pt1),
                "pt2": list(tw.pt2),
                "in_count": tw.in_count,
                "out_count": tw.out_count,
            }
            for tw in state.tripwires
        ]
    })


@app.post("/api/tripwires")
async def api_add_tripwire(body: TripwireAddRequest):
    """Add a new custom tripwire line."""
    line_id = f"custom_tw_{int(time.time() * 1000) % 100000}"
    new_tw = TripwireLine(
        line_id=line_id,
        pt1=tuple(body.pt1),
        pt2=tuple(body.pt2),
        label=body.label,
    )
    state.tripwires.append(new_tw)
    return JSONResponse({"status": "ok", "line_id": line_id, "label": new_tw.label})


@app.delete("/api/tripwires/{line_id}")
async def api_delete_tripwire(line_id: str):
    """Remove a tripwire line by ID."""
    state.tripwires = [tw for tw in state.tripwires if tw.line_id != line_id]
    return JSONResponse({"status": "ok", "deleted": line_id})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
