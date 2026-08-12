"""Refactored alert management system."""
import cv2
import numpy as np
import time
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from src.core.base_detector import Detection
from src.utils.logging_utils import setup_logger


@dataclass
class AlertConfig:
    """Alert configuration."""
    dwell_threshold_sec: float = 3.0
    alert_cooldown_sec: float = 5.0
    polygon_coords: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class Alert:
    """Alert event."""
    alert_id: str
    track_id: int
    timestamp: datetime
    alert_type: str  # 'dwell', 'intrusion', 'exit'
    bbox: tuple
    dwell_time: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'track_id': self.track_id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'bbox': [float(x) for x in self.bbox],
            'dwell_time': self.dwell_time,
        }


class AlertManager:
    """Manages detection-based alerts."""
    
    def __init__(self, config: AlertConfig, frame_shape: Tuple[int, int]):
        """
        Initialize alert manager.
        
        Args:
            config: AlertConfig object
            frame_shape: Tuple (height, width) of frame
        """
        self.config = config
        self.frame_shape = frame_shape
        self.logger = setup_logger('alert_manager', log_file='logs/alerts.log')
        
        # Build zone mask if polygon provided
        self.mask = self._build_mask()
        
        # Tracking state
        self.track_timers: Dict[int, float] = {}  # track_id -> entry_time
        self.last_alert_time: Dict[int, float] = {}  # track_id -> last_alert_time
        self.alert_history: List[Alert] = []
        self.alert_counter = 0
    
    def _build_mask(self) -> Optional[np.ndarray]:
        """Build binary mask for restricted zone."""
        if not self.config.polygon_coords:
            return None
        
        mask = np.zeros(self.frame_shape[:2], dtype=np.uint8)
        pts = np.array(self.config.polygon_coords, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask
    
    def is_point_in_zone(self, x: int, y: int) -> bool:
        """Check if point is in restricted zone."""
        if self.mask is None:
            return False
        
        h, w = self.mask.shape
        if 0 <= x < w and 0 <= y < h:
            return self.mask[int(y), int(x)] == 255
        return False
    
    def evaluate_detections(self, detections: List[Detection], 
                          current_time: Optional[float] = None) -> List[Alert]:
        """
        Evaluate detections and generate alerts.
        
        Args:
            detections: List of Detection objects
            current_time: Current timestamp (unix time)
            
        Returns:
            List of Alert objects
        """
        if current_time is None:
            current_time = time.time()
        
        active_alerts = []
        current_ids_in_zone: Set[int] = set()
        
        # Process each detection
        for detection in detections:
            if detection.track_id is None:
                continue
            
            # Calculate foot point (center bottom of bbox)
            x1, y1, x2, y2 = detection.bbox
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            
            # Check if in zone
            if self.is_point_in_zone(foot_x, foot_y):
                current_ids_in_zone.add(detection.track_id)
                
                # Register entry time if new
                if detection.track_id not in self.track_timers:
                    self.track_timers[detection.track_id] = current_time
                    self.logger.info(f"Track {detection.track_id} entered zone")
                
                # Evaluate dwell time
                dwell_time = current_time - self.track_timers[detection.track_id]
                
                if dwell_time >= self.config.dwell_threshold_sec:
                    # Check cooldown
                    last_alert = self.last_alert_time.get(detection.track_id, 0)
                    if current_time - last_alert >= self.config.alert_cooldown_sec:
                        alert = self._create_alert(
                            detection.track_id,
                            detection.bbox,
                            dwell_time,
                            'dwell',
                            current_time
                        )
                        active_alerts.append(alert)
                        self.last_alert_time[detection.track_id] = current_time
                        self.logger.warning(
                            f"ALERT: Track {detection.track_id} dwelling for {dwell_time:.1f}s"
                        )
        
        # Clean up IDs that left the zone
        ids_to_remove = [tid for tid in self.track_timers if tid not in current_ids_in_zone]
        for tid in ids_to_remove:
            del self.track_timers[tid]
            self.logger.info(f"Track {tid} exited zone")
        
        return active_alerts
    
    def _create_alert(self, track_id: int, bbox: tuple, dwell_time: float,
                     alert_type: str, timestamp_unix: float) -> Alert:
        """Create alert object."""
        self.alert_counter += 1
        alert = Alert(
            alert_id=f"ALT_{self.alert_counter:06d}",
            track_id=track_id,
            timestamp=datetime.fromtimestamp(timestamp_unix),
            alert_type=alert_type,
            bbox=bbox,
            dwell_time=dwell_time,
        )
        self.alert_history.append(alert)
        return alert
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history."""
        if limit:
            return self.alert_history[-limit:]
        return self.alert_history
    
    def reset(self):
        """Reset alert manager state."""
        self.track_timers.clear()
        self.last_alert_time.clear()
        self.alert_history.clear()
        self.alert_counter = 0
