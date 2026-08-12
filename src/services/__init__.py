"""Detection service orchestrating detection and alert logic."""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from src.core.config_manager import ConfigManager, DetectorConfig, AlertConfig as AlertConfigDataclass, StorageConfig
from src.core.base_detector import BaseDetector, Detection
from src.detector.yolo_detector_impl import YOLODetectorImpl
from src.alerts import AlertManager, AlertConfig as AlertManagerConfig, Alert
from src.storage import create_storage, BaseStorage
from src.utils.logging_utils import setup_logger


class DetectionService:
    """High-level detection service orchestrating all components."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize detection service.
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager
        self.logger = setup_logger('detection_service', log_file='logs/detection_service.log')
        
        # Initialize components
        self.detector = self._init_detector()
        self.alert_manager: Optional[AlertManager] = None
        self.storage = self._init_storage()
        
        self.logger.info("Detection service initialized")
    
    def _init_detector(self) -> BaseDetector:
        """Initialize detector from config."""
        detector_config = self.config_manager.get_detector_config()
        self.logger.info(f"Initializing detector: {detector_config.model_name}")
        
        detector = YOLODetectorImpl(
            model_name=detector_config.model_name,
            confidence=detector_config.confidence,
            device=detector_config.device,
        )
        return detector
    
    def _init_storage(self) -> BaseStorage:
        """Initialize storage from config."""
        storage_config = self.config_manager.get_storage_config()
        self.logger.info(f"Initializing storage: {storage_config.type}")
        
        storage = create_storage(
            storage_type=storage_config.type,
            output_dir=storage_config.output_dir
        )
        return storage
    
    def init_alert_system(self, polygon_coords: List[Tuple[int, int]], 
                         frame_shape: Tuple[int, int]):
        """
        Initialize alert system with restricted zone.
        
        Args:
            polygon_coords: List of (x, y) coordinates defining the zone
            frame_shape: Tuple (height, width) of video frames
        """
        alert_config_data = self.config_manager.get_alert_config()
        
        alert_config = AlertManagerConfig(
            dwell_threshold_sec=alert_config_data.dwell_threshold_sec,
            alert_cooldown_sec=alert_config_data.alert_cooldown_sec,
            polygon_coords=polygon_coords
        )
        
        self.alert_manager = AlertManager(alert_config, frame_shape)
        self.logger.info("Alert system initialized")
    
    def detect_image(self, image_path: str) -> List[Detection]:
        """
        Detect objects in a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detections
        """
        self.logger.info(f"Processing image: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Failed to read image: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        detections = self.detector.detect(image)
        self.logger.info(f"Found {len(detections)} objects")
        
        return detections
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Detection], List[Alert]]:
        """
        Process a single video frame.
        
        Args:
            frame: Video frame (numpy array)
            
        Returns:
            Tuple of (detections, alerts)
        """
        # Run detection with tracking
        detections, _ = self.detector.detect_with_tracking(frame)
        
        # Evaluate alerts if alert system is initialized
        alerts = []
        if self.alert_manager:
            alerts = self.alert_manager.evaluate_detections(detections)
        
        return detections, alerts
    
    def annotate_frame(self, frame: np.ndarray, detections: List[Detection],
                      alerts: List[Alert] = None) -> np.ndarray:
        """
        Annotate frame with detections and alerts.
        
        Args:
            frame: Video frame
            detections: List of detections
            alerts: List of alerts
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(
                annotated, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                (0, 255, 0), 
                2
            )
            
            # Draw label
            label = f"{detection.class_name} {detection.confidence:.2f}"
            if detection.track_id is not None:
                label += f" (ID: {detection.track_id})"
            
            cv2.putText(
                annotated,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Draw alerts
        if alerts:
            for alert in alerts:
                x1, y1, x2, y2 = alert.bbox
                # Draw red rectangle for alert
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    3
                )
                
                # Draw alert text
                alert_text = f"ALERT: {alert.alert_type} ({alert.dwell_time:.1f}s)"
                cv2.putText(
                    annotated,
                    alert_text,
                    (int(x1), int(y1) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
        
        # Draw zone if alert manager exists and has mask
        if self.alert_manager and self.alert_manager.mask is not None:
            mask_contours = cv2.findContours(
                self.alert_manager.mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )[0]
            cv2.drawContours(annotated, mask_contours, -1, (255, 0, 0), 2)
        
        return annotated
    
    def save_results(self, detections: List[Detection], alerts: List[Alert],
                    filename: str, metadata: Optional[Dict] = None):
        """
        Save detection and alert results.
        
        Args:
            detections: List of detections
            alerts: List of alerts
            filename: Output filename (without extension)
            metadata: Additional metadata to save
        """
        storage_config = self.config_manager.get_storage_config()
        
        if storage_config.save_detections and detections:
            self.storage.save_detections(detections, filename, metadata)
        
        if storage_config.save_alerts and alerts:
            alert_dicts = [alert.to_dict() for alert in alerts]
            self.storage.save_alerts(alert_dicts, filename, metadata)
    
    def get_detector_info(self) -> Dict:
        """Get detector model information."""
        return self.detector.get_model_info()
