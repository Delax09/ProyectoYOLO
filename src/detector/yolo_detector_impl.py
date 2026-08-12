"""YOLO detector implementation."""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from ultralytics import YOLO
from src.core.base_detector import BaseDetector, Detection
from src.utils.logging_utils import setup_logger


class YOLODetectorImpl(BaseDetector):
    """YOLO v8 detector implementation."""
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence: float = 0.5, device: str = 'cpu'):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: YOLO model name (e.g., 'yolov8n.pt', 'yolov8s.pt')
            confidence: Confidence threshold (0-1)
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        self.device = device
        self.logger = setup_logger(
            'yolo_detector', 
            log_file='logs/yolo_detector.log'
        )
        super().__init__(model_name, confidence)
        
    def _load_model(self):
        """Load YOLO model."""
        try:
            self.logger.info(f"Loading YOLO model: {self.model_name} on device: {self.device}")
            self.model = YOLO(self.model_name)
            
            # Get class names
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            else:
                self.class_names = {}
                
            self.logger.info(f"Model loaded successfully. Classes: {len(self.class_names)}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of Detection objects
        """
        try:
            results = self.model(image, conf=self.confidence)
            detections = []
            
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        class_name = self.class_names.get(
                            cls_id, f"class_{cls_id}"
                        ) if isinstance(self.class_names, dict) else self.class_names[cls_id]
                        
                        bbox = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        detection = Detection(
                            class_id=cls_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=tuple(bbox)
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
    
    def detect_with_tracking(self, image: np.ndarray) -> Tuple[List[Detection], any]:
        """
        Run detection with tracking.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Tuple of (detections list, raw results)
        """
        try:
            # Run inference with tracking
            results = self.model.track(image, conf=self.confidence, persist=True)
            detections = []
            
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        class_name = self.class_names.get(
                            cls_id, f"class_{cls_id}"
                        ) if isinstance(self.class_names, dict) else self.class_names[cls_id]
                        
                        bbox = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        # Get track ID if available
                        track_id = None
                        if box.id is not None:
                            track_id = int(box.id[0])
                        
                        detection = Detection(
                            class_id=cls_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=tuple(bbox),
                            track_id=track_id
                        )
                        detections.append(detection)
            
            return detections, results
            
        except Exception as e:
            self.logger.error(f"Tracking detection failed: {e}")
            return [], None
