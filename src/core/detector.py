# src/core/detector.py
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt', confidence=0.5, logger=None):
        self.logger = logger
        if self.logger:
            self.logger.info(f"Cargando modelo YOLO: {model_name}")
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.class_names = self.model.names

    def predict_image(self, frame):
        return self.model(frame, conf=self.confidence)

    def track_frame(self, frame, tracker="bytetrack.yaml", classes=None):
        return self.model.track(
            frame, 
            conf=self.confidence, 
            persist=True, 
            tracker=tracker, 
            classes=classes, 
            verbose=False
        )