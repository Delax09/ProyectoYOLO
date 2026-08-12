"""
Ejemplos de uso de la nueva arquitectura escalable.

Este archivo muestra diferentes formas de usar el sistema
en distintos escenarios.
"""

from pathlib import Path
from src.core.config_manager import get_config_manager
from src.services import DetectionService
import json


# ============================================================================
# EJEMPLO 1: Detección simple en una imagen
# ============================================================================

def example_simple_image_detection():
    """Ejemplo simple: detectar objetos en una imagen."""
    
    # Cargar configuración
    config = get_config_manager(Path('config/app.yaml'))
    
    # Crear servicio de detección
    service = DetectionService(config)
    
    # Procesar imagen
    detections = service.detect_image('examples/ImagenPersonas.jpg')
    
    # Guardar resultados
    service.save_results(detections, [], 'imagen_test')
    
    print(f"Encontrados {len(detections)} objetos")


# ============================================================================
# EJEMPLO 2: Procesamiento de video con alertas
# ============================================================================

def example_video_with_alerts():
    """Ejemplo: procesar video y generar alertas."""
    import cv2
    
    config = get_config_manager(Path('config/app.yaml'))
    service = DetectionService(config)
    
    # Cargar zonas restringidas (soporta ambos formatos: 'zones' y 'zonas')
    with open('config/zones.json', 'r', encoding='utf-8') as f:
        zones_data = json.load(f)
        zones_list = zones_data.get('zones', zones_data.get('zonas', []))
        if zones_list:
            zone = zones_list[0]
            zones = zone.get('coordinates', zone.get('puntos', []))
        else:
            zones = []
    
    # Inicializar alertas
    cap = cv2.VideoCapture('examples/VideoPersonas.mp4')
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    service.init_alert_system(zones, (height, width))
    
    # Procesar frames
    frame_count = 0
    detections_list = []
    alerts_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        detections, alerts = service.process_frame(frame)
        
        detections_list.extend(detections)
        alerts_list.extend(alerts)
        
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(detections)} objetos, {len(alerts)} alertas")
    
    cap.release()
    
    # Guardar resultados
    service.save_results(detections_list, alerts_list, 'video_test')
    
    print(f"\nProceso completado:")
    print(f"  - Frames procesados: {frame_count}")
    print(f"  - Total detecciones: {len(detections_list)}")
    print(f"  - Total alertas: {len(alerts_list)}")


# ============================================================================
# EJEMPLO 3: Usar un detector personalizado
# ============================================================================

def example_custom_detector():
    """Ejemplo: usar un detector personalizado."""
    from src.core.base_detector import BaseDetector, Detection
    import numpy as np
    
    class CustomDetector(BaseDetector):
        """Detector personalizado de ejemplo."""
        
        def _load_model(self):
            """Cargar tu modelo aquí."""
            self.class_names = {0: 'person', 1: 'car', 2: 'bicycle'}
            print(f"Modelo personalizado cargado: {self.model_name}")
        
        def detect(self, image: np.ndarray) -> list:
            """Implementar lógica de detección."""
            # Aquí iría tu lógica de detección
            detections = []
            # Ejemplo: detectar un objeto dummy
            detections.append(Detection(
                class_id=0,
                class_name='person',
                confidence=0.95,
                bbox=(100, 100, 200, 300)
            ))
            return detections
        
        def detect_with_tracking(self, image: np.ndarray) -> tuple:
            """Implementar lógica de detección con tracking."""
            detections = self.detect(image)
            for i, det in enumerate(detections):
                det.track_id = i
            return detections, None
    
    # Usar el detector
    detector = CustomDetector(model_name='custom_model.pt')
    print(f"Información del modelo: {detector.get_model_info()}")


# ============================================================================
# EJEMPLO 4: Procesar múltiples imágenes
# ============================================================================

def example_batch_image_processing():
    """Ejemplo: procesar múltiples imágenes."""
    from pathlib import Path
    
    config = get_config_manager(Path('config/app.yaml'))
    service = DetectionService(config)
    
    # Procesar todas las imágenes en una carpeta
    image_dir = Path('examples')
    all_detections = []
    
    for image_path in image_dir.glob('*.jpg'):
        try:
            detections = service.detect_image(str(image_path))
            all_detections.extend(detections)
            print(f"{image_path.name}: {len(detections)} objetos")
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
    
    # Guardar resultados combinados
    service.save_results(all_detections, [], 'batch_results')
    print(f"\nTotal de objetos detectados: {len(all_detections)}")


# ============================================================================
# EJEMPLO 5: Acceder a configuración
# ============================================================================

def example_configuration_access():
    """Ejemplo: acceder a la configuración."""
    
    config = get_config_manager(Path('config/app.yaml'))
    
    # Acceder a configuraciones específicas
    detector_cfg = config.get_detector_config()
    alert_cfg = config.get_alert_config()
    storage_cfg = config.get_storage_config()
    
    print("Configuración del Detector:")
    print(f"  - Modelo: {detector_cfg.model_name}")
    print(f"  - Confianza: {detector_cfg.confidence}")
    print(f"  - Dispositivo: {detector_cfg.device}")
    
    print("\nConfiguración de Alertas:")
    print(f"  - Umbral de permanencia: {alert_cfg.dwell_threshold_sec}s")
    print(f"  - Cooldown: {alert_cfg.alert_cooldown_sec}s")
    
    print("\nConfiguración de Almacenamiento:")
    print(f"  - Tipo: {storage_cfg.type}")
    print(f"  - Directorio: {storage_cfg.output_dir}")


# ============================================================================
# EJEMPLO 6: Cámara en vivo con anotaciones
# ============================================================================

def example_live_camera_with_annotations():
    """Ejemplo: cámara en vivo con anotaciones."""
    import cv2
    
    config = get_config_manager(Path('config/app.yaml'))
    service = DetectionService(config)
    
    cap = cv2.VideoCapture(0)  # Cámara predeterminada
    
    print("Presiona 'q' para salir...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar frame
        detections, alerts = service.process_frame(frame)
        
        # Anotar
        annotated = service.annotate_frame(frame, detections, alerts)
        
        # Mostrar
        cv2.imshow('Detección en Vivo', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# EJEMPLO 7: Contar objetos de una clase específica
# ============================================================================

def example_count_specific_class():
    """Ejemplo: contar objetos de una clase específica."""
    
    config = get_config_manager(Path('config/app.yaml'))
    service = DetectionService(config)
    
    # Detectar
    detections = service.detect_image('examples/ImagenPersonas.jpg')
    
    # Filtrar por clase
    persons = [d for d in detections if d.class_name == 'person']
    cars = [d for d in detections if d.class_name == 'car']
    
    print(f"Objetos detectados:")
    print(f"  - Personas: {len(persons)}")
    print(f"  - Autos: {len(cars)}")
    print(f"  - Total: {len(detections)}")
    
    # Detalles de confianza
    for person in persons:
        print(f"    Persona con confianza {person.confidence:.2f}")


if __name__ == '__main__':
    print("Ejemplos de uso de la arquitectura escalable\n")
    
    # Descomentar el ejemplo que desee ejecutar:
    
    # example_simple_image_detection()
    # example_video_with_alerts()
    # example_custom_detector()
    # example_batch_image_processing()
    # example_configuration_access()
    # example_live_camera_with_annotations()
    example_count_specific_class()
