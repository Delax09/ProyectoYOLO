"""Storage backend for detection results."""
import json
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.core.base_detector import Detection
from src.utils.logging_utils import setup_logger


class BaseStorage(ABC):
    """Abstract base class for storage backends."""
    
    def __init__(self, output_dir: str = 'output'):
        """Initialize storage."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger('storage', log_file='logs/storage.log')
    
    @abstractmethod
    def save_detections(self, detections: List[Detection], 
                       filename: str, metadata: Optional[Dict] = None) -> Path:
        """Save detections to storage."""
        pass
    
    @abstractmethod
    def save_alerts(self, alerts: List[Dict], 
                   filename: str, metadata: Optional[Dict] = None) -> Path:
        """Save alerts to storage."""
        pass


class JSONStorage(BaseStorage):
    """JSON file storage backend."""
    
    def save_detections(self, detections: List[Detection], 
                       filename: str, metadata: Optional[Dict] = None) -> Path:
        """Save detections as JSON."""
        output_path = self.output_dir / f"{filename}_detections.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'detections': [d.to_dict() for d in detections],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(detections)} detections to {output_path}")
        return output_path
    
    def save_alerts(self, alerts: List[Dict], 
                   filename: str, metadata: Optional[Dict] = None) -> Path:
        """Save alerts as JSON."""
        output_path = self.output_dir / f"{filename}_alerts.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'alerts': alerts,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(alerts)} alerts to {output_path}")
        return output_path


class CSVStorage(BaseStorage):
    """CSV file storage backend."""
    
    def save_detections(self, detections: List[Detection], 
                       filename: str, metadata: Optional[Dict] = None) -> Path:
        """Save detections as CSV."""
        output_path = self.output_dir / f"{filename}_detections.csv"
        
        if not detections:
            self.logger.warning(f"No detections to save to {output_path}")
            return output_path
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=['class_id', 'class_name', 'confidence', 'bbox', 'track_id']
            )
            writer.writeheader()
            
            for detection in detections:
                writer.writerow({
                    'class_id': detection.class_id,
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'bbox': str(detection.bbox),
                    'track_id': detection.track_id,
                })
        
        self.logger.info(f"Saved {len(detections)} detections to {output_path}")
        return output_path
    
    def save_alerts(self, alerts: List[Dict], 
                   filename: str, metadata: Optional[Dict] = None) -> Path:
        """Save alerts as CSV."""
        output_path = self.output_dir / f"{filename}_alerts.csv"
        
        if not alerts:
            self.logger.warning(f"No alerts to save to {output_path}")
            return output_path
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=['track_id', 'bbox', 'dwell_time']
            )
            writer.writeheader()
            writer.writerows(alerts)
        
        self.logger.info(f"Saved {len(alerts)} alerts to {output_path}")
        return output_path


def create_storage(storage_type: str = 'json', output_dir: str = 'output') -> BaseStorage:
    """Factory function to create storage backend."""
    if storage_type.lower() == 'json':
        return JSONStorage(output_dir)
    elif storage_type.lower() == 'csv':
        return CSVStorage(output_dir)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
