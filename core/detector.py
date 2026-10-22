"""
YOLO Detection Engine — wraps Ultralytics YOLOv8/v11 for inference.
Supports dynamic model switching, class filtering, and device selection.
"""
import os
import time
import threading
from ultralytics import YOLO

class YOLOEngine:
    AVAILABLE_MODELS = {
        "yolov8n": {"file": "yolov8n.pt", "label": "YOLOv8 Nano (Fastest)", "params": "3.2M"},
        "yolov8s": {"file": "yolov8s.pt", "label": "YOLOv8 Small", "params": "11.2M"},
        "yolov8m": {"file": "yolov8m.pt", "label": "YOLOv8 Medium", "params": "25.9M"},
        "yolov8l": {"file": "yolov8l.pt", "label": "YOLOv8 Large", "params": "43.7M"},
        "yolov8x": {"file": "yolov8x.pt", "label": "YOLOv8 XLarge (Most Accurate)", "params": "68.2M"},
    }

    def __init__(self, model_key: str = "yolov8n", device: str = "cpu"):
        self.model_key = model_key
        self.device = device
        self.model = None
        self.class_names = {}
        self.lock = threading.Lock()
        self._load_model(model_key)
        
        # Performance metrics
        self.last_inference_ms = 0
        self.avg_inference_ms = 0
        self._inference_times = []

    def _load_model(self, model_key: str):
        info = self.AVAILABLE_MODELS.get(model_key, self.AVAILABLE_MODELS["yolov8n"])
        model_path = info["file"]
        print(f"[AegisVision] Loading {info['label']} ({info['params']} parameters)...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.model_key = model_key
        print(f"[AegisVision] Model loaded successfully on device: {self.device}")

    @property
    def model_name(self):
        return self.model_key

    def load_model(self, model_name: str):
        self.switch_model(model_name)

    def switch_model(self, model_key: str):
        if model_key == self.model_key:
            return
        with self.lock:
            self._load_model(model_key)

    def detect(self, frame, conf_threshold: float = 0.4, target_classes: list = None, imgsz: int = 640, classes: list = None, confidence: float = None):
        """
        Run inference on a single frame.
        Supports both (conf_threshold, target_classes) and (classes, confidence) parameter names.
        Returns list of detection dicts:
          [{'bbox': [x1,y1,x2,y2], 'cls_id': int, 'cls_name': str, 'conf': float}, ...]
        """
        if confidence is not None:
            conf_threshold = confidence
        if classes is not None:
            target_classes = classes

        t0 = time.perf_counter()

        
        with self.lock:
            results = self.model.predict(
                source=frame,
                conf=conf_threshold,
                device=self.device,
                imgsz=imgsz,
                verbose=False,
                classes=target_classes
            )
        
        t1 = time.perf_counter()
        self.last_inference_ms = round((t1 - t0) * 1000, 1)
        self._inference_times.append(self.last_inference_ms)
        if len(self._inference_times) > 60:
            self._inference_times.pop(0)
        self.avg_inference_ms = round(sum(self._inference_times) / len(self._inference_times), 1)

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    detections.append({
                        'bbox': xyxy,
                        'cls_id': cls_id,
                        'cls_name': cls_name,
                        'conf': conf
                    })
        return detections

    def get_model_info(self):
        info = self.AVAILABLE_MODELS.get(self.model_key, {})
        return {
            "key": self.model_key,
            "label": info.get("label", self.model_key),
            "params": info.get("params", "?"),
            "device": self.device,
            "avg_ms": self.avg_inference_ms,
            "last_ms": self.last_inference_ms,
            "num_classes": len(self.class_names),
        }
