"""Base detector abstract class for detector implementations."""
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    """Detection result from a detector."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    track_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'bbox': [float(x) for x in self.bbox],
            'track_id': self.track_id,
        }


class BaseDetector(ABC):
    """Abstract base class for object detectors."""
    
    def __init__(self, model_name: str, confidence: float = 0.5, **kwargs):
        """
        Initialize detector.
        
        Args:
            model_name: Name or path of the model
            confidence: Confidence threshold
            **kwargs: Additional arguments
        """
        self.model_name = model_name
        self.confidence = confidence
        self.class_names: Dict[int, str] = {}
        self._load_model()
        
    @abstractmethod
    def _load_model(self):
        """Load the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def detect_with_tracking(self, image: np.ndarray) -> tuple:
        """
        Run detection with tracking.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Tuple of (detections, raw_results)
        """
        pass
    
    def get_class_names(self) -> Dict[int, str]:
        """Get class names mapping."""
        return self.class_names
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'confidence': self.confidence,
            'num_classes': len(self.class_names),
            'class_names': list(self.class_names.values()),
        }
