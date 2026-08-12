"""Unit tests for the scalable architecture."""
import pytest
import numpy as np
from pathlib import Path
from src.core.base_detector import BaseDetector, Detection
from src.core.config_manager import ConfigManager, DetectorConfig, AlertConfig
from src.alerts import AlertManager, AlertConfig as AlertManagerConfig
from src.storage import JSONStorage, CSVStorage


# ============================================================================
# Tests para Detection
# ============================================================================

class TestDetection:
    """Tests for Detection data class."""
    
    def test_detection_creation(self):
        """Test creating a detection object."""
        detection = Detection(
            class_id=0,
            class_name='person',
            confidence=0.95,
            bbox=(10, 20, 100, 200),
            track_id=1
        )
        
        assert detection.class_id == 0
        assert detection.class_name == 'person'
        assert detection.confidence == 0.95
        assert detection.track_id == 1
    
    def test_detection_to_dict(self):
        """Test converting detection to dictionary."""
        detection = Detection(
            class_id=0,
            class_name='person',
            confidence=0.95,
            bbox=(10, 20, 100, 200),
            track_id=1
        )
        
        d = detection.to_dict()
        
        assert d['class_id'] == 0
        assert d['class_name'] == 'person'
        assert d['confidence'] == 0.95
        assert d['track_id'] == 1


# ============================================================================
# Tests para ConfigManager
# ============================================================================

class TestConfigManager:
    """Tests for configuration management."""
    
    def test_get_detector_config(self):
        """Test getting detector configuration."""
        config = ConfigManager()
        detector_cfg = config.get_detector_config()
        
        assert isinstance(detector_cfg, DetectorConfig)
        assert detector_cfg.model_name is not None
        assert 0 <= detector_cfg.confidence <= 1
    
    def test_get_alert_config(self):
        """Test getting alert configuration."""
        config = ConfigManager()
        alert_cfg = config.get_alert_config()
        
        assert isinstance(alert_cfg, AlertConfig)
        assert alert_cfg.dwell_threshold_sec >= 0
        assert alert_cfg.alert_cooldown_sec >= 0
    
    def test_config_defaults(self):
        """Test configuration defaults."""
        config = ConfigManager()
        
        detector_cfg = config.get_detector_config()
        assert detector_cfg.confidence == 0.5


# ============================================================================
# Tests para AlertManager
# ============================================================================

class TestAlertManager:
    """Tests for alert management system."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager instance."""
        config = AlertManagerConfig(
            dwell_threshold_sec=1.0,
            alert_cooldown_sec=0.5,
            polygon_coords=[(0, 0), (100, 0), (100, 100), (0, 100)]
        )
        return AlertManager(config, frame_shape=(200, 200))
    
    def test_point_in_zone(self, alert_manager):
        """Test point in zone detection."""
        # Point inside zone
        assert alert_manager.is_point_in_zone(50, 50)
        
        # Point outside zone
        assert not alert_manager.is_point_in_zone(150, 150)
    
    def test_evaluate_detections_no_tracking(self, alert_manager):
        """Test alert evaluation without tracking."""
        detections = [
            Detection(
                class_id=0,
                class_name='person',
                confidence=0.95,
                bbox=(20, 20, 80, 80),
                track_id=None  # No tracking
            )
        ]
        
        alerts = alert_manager.evaluate_detections(detections)
        
        # Should be empty since no track_id
        assert len(alerts) == 0
    
    def test_alert_creation(self, alert_manager):
        """Test alert creation."""
        detections = [
            Detection(
                class_id=0,
                class_name='person',
                confidence=0.95,
                bbox=(20, 20, 80, 80),
                track_id=1
            )
        ]
        
        # First evaluation - enters zone
        alerts = alert_manager.evaluate_detections(detections, current_time=0)
        assert len(alerts) == 0  # Too early, threshold not met
        
        # Second evaluation - after dwell threshold
        alerts = alert_manager.evaluate_detections(detections, current_time=2.0)
        assert len(alerts) > 0
        assert alerts[0].track_id == 1


# ============================================================================
# Tests para Storage
# ============================================================================

class TestStorage:
    """Tests for storage backends."""
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detections."""
        return [
            Detection(
                class_id=0,
                class_name='person',
                confidence=0.95,
                bbox=(10, 20, 100, 200),
            ),
            Detection(
                class_id=1,
                class_name='car',
                confidence=0.87,
                bbox=(50, 50, 150, 150),
            )
        ]
    
    def test_json_storage_detections(self, sample_detections, tmp_path):
        """Test JSON storage for detections."""
        storage = JSONStorage(output_dir=str(tmp_path))
        
        result_path = storage.save_detections(
            sample_detections,
            'test',
            metadata={'source': 'test'}
        )
        
        assert result_path.exists()
        assert result_path.suffix == '.json'
    
    def test_csv_storage_detections(self, sample_detections, tmp_path):
        """Test CSV storage for detections."""
        storage = CSVStorage(output_dir=str(tmp_path))
        
        result_path = storage.save_detections(
            sample_detections,
            'test',
            metadata={'source': 'test'}
        )
        
        assert result_path.exists()
        assert result_path.suffix == '.csv'
    
    def test_empty_detections(self, tmp_path):
        """Test storing empty detections."""
        storage = JSONStorage(output_dir=str(tmp_path))
        
        result_path = storage.save_detections([], 'empty')
        
        # Should still create file
        assert result_path.exists()


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the architecture."""
    
    def test_detection_to_alert_flow(self):
        """Test flow from detection to alert."""
        # Create alert manager
        config = AlertManagerConfig(
            dwell_threshold_sec=1.0,
            polygon_coords=[(0, 0), (100, 0), (100, 100), (0, 100)]
        )
        alert_manager = AlertManager(config, frame_shape=(200, 200))
        
        # Create detection in zone
        detection = Detection(
            class_id=0,
            class_name='person',
            confidence=0.95,
            bbox=(20, 20, 80, 80),
            track_id=1
        )
        
        # Evaluate at different times
        alerts_t0 = alert_manager.evaluate_detections([detection], current_time=0)
        alerts_t2 = alert_manager.evaluate_detections([detection], current_time=2.0)
        
        # Should generate alert after dwell threshold
        assert len(alerts_t0) == 0
        assert len(alerts_t2) > 0
    
    def test_config_to_service_integration(self, tmp_path):
        """Test configuration to service integration."""
        # Create config file
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
detector:
  model_name: 'yolov8n.pt'
  confidence: 0.5
  device: 'cpu'

alerts:
  enabled: true
  dwell_threshold_sec: 3.0

storage:
  type: 'json'
  output_dir: 'output'
""")
        
        # Load configuration
        config = ConfigManager(config_path)
        
        # Verify configuration loaded correctly
        detector_cfg = config.get_detector_config()
        assert detector_cfg.model_name == 'yolov8n.pt'
        assert detector_cfg.confidence == 0.5


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
