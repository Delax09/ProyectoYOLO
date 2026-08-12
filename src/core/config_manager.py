"""Configuration manager for centralized project configuration."""
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class DetectorConfig:
    """Configuration for detector models."""
    model_name: str
    confidence: float = 0.5
    device: str = 'cpu'  # 'cpu', 'cuda', 'mps'
    iou_threshold: float = 0.45
    
    
@dataclass
class AlertConfig:
    """Configuration for alert system."""
    enabled: bool = True
    dwell_threshold_sec: float = 3.0
    alert_cooldown_sec: float = 5.0
    

@dataclass
class StorageConfig:
    """Configuration for result storage."""
    type: str = 'json'  # 'json', 'csv', 'database'
    output_dir: str = 'output'
    save_detections: bool = True
    save_alerts: bool = True
    

@dataclass
class LoggerConfig:
    """Configuration for logging."""
    level: str = 'INFO'
    log_dir: str = 'logs'
    file_name: str = 'app.log'


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or Path('config/app.yaml')
        self.config: Dict[str, Any] = {}
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = self._get_defaults()
            
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'detector': asdict(DetectorConfig(model_name='yolov8n.pt')),
            'alerts': asdict(AlertConfig()),
            'storage': asdict(StorageConfig()),
            'logger': asdict(LoggerConfig()),
        }
    
    def get_detector_config(self) -> DetectorConfig:
        """Get detector configuration."""
        cfg = self.config.get('detector', {})
        return DetectorConfig(**cfg)
    
    def get_alert_config(self) -> AlertConfig:
        """Get alert configuration."""
        cfg = self.config.get('alerts', {})
        return AlertConfig(**cfg)
    
    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration."""
        cfg = self.config.get('storage', {})
        return StorageConfig(**cfg)
    
    def get_logger_config(self) -> LoggerConfig:
        """Get logger configuration."""
        cfg = self.config.get('logger', {})
        return LoggerConfig(**cfg)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value or default
    
    def save_config(self, output_path: Optional[Path] = None):
        """Save current configuration to file."""
        save_path = output_path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """Get or create global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager
