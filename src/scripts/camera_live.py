"""Live camera detection with alert system."""
import argparse
import cv2
import json
from pathlib import Path
from typing import List, Tuple
from src.core.config_manager import get_config_manager
from src.services import DetectionService
from src.utils.logging_utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detección en vivo con cámara y alertas'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='ID de la cámara (default: 0)'
    )
    parser.add_argument(
        '--config',
        default='config/app.yaml',
        help='Archivo de configuración'
    )
    parser.add_argument(
        '--zones',
        default='config/zones.json',
        help='Archivo JSON con zonas restringidas'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Guardar detecciones y alertas'
    )
    return parser.parse_args()


def load_zones(zones_path: str) -> List[Tuple[int, int]]:
    """Load restricted zones from JSON file. Soporta ambos formatos."""
    zones_file = Path(zones_path)
    if zones_file.exists():
        with open(zones_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Soportar ambos formatos: 'zones' y 'zonas'
            zones = data.get('zones', data.get('zonas', []))
            if zones:
                # Soportar ambos: 'coordinates' y 'puntos'
                zone = zones[0]
                coords = zone.get('coordinates', zone.get('puntos', []))
                return coords
    return []


def main():
    args = parse_args()
    logger = setup_logger('camera_live', log_file='logs/camera_live.log')
    logger.info("=== Iniciando cámara en vivo ===")
    
    try:
        # Cargar configuración
        config_manager = get_config_manager(Path(args.config))
        detection_service = DetectionService(config_manager)
        
        # Abrir cámara
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            logger.error(f"No se pudo abrir la cámara: {args.camera}")
            raise SystemExit(1)
        
        # Obtener propiedades
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Cámara: {width}x{height} @ {fps} FPS")
        
        # Inicializar sistema de alertas
        zones = load_zones(args.zones)
        if zones:
            detection_service.init_alert_system(zones, (height, width))
            logger.info(f"Sistema de alertas activo con {len(zones)} puntos")
        
        # Contadores
        frame_count = 0
        total_detections = 0
        total_alerts = 0
        all_alerts = []
        all_detections = []
        
        print("Presiona 'q' para salir...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Error al capturar frame")
                break
            
            frame_count += 1
            
            # Procesar frame
            detections, alerts = detection_service.process_frame(frame)
            total_detections += len(detections)
            total_alerts += len(alerts)
            
            # Guardar historial si se solicitó
            if args.save_results:
                all_detections.extend(detections)
                all_alerts.extend(alerts)
            
            # Anotar frame
            annotated = detection_service.annotate_frame(frame, detections, alerts)
            
            # Agregar información
            info_text = f"Frame: {frame_count} | Objetos: {len(detections)} | Alertas: {len(alerts)}"
            cv2.putText(
                annotated,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Mostrar
            cv2.imshow('Detección en Vivo', annotated)
            
            # Control
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Salida solicitada por usuario")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Sesión completada: {frame_count} frames")
        logger.info(f"Total de detecciones: {total_detections}")
        logger.info(f"Total de alertas: {total_alerts}")
        
        # Guardar resultados si se solicitó
        if args.save_results and (all_detections or all_alerts):
            detection_service.save_results(
                all_detections,
                all_alerts,
                'camera_session',
                metadata={'frames': frame_count}
            )
            logger.info("Resultados guardados")
        
        print(f"\n✓ Sesión finalizada!")
        print(f"  - Frames procesados: {frame_count}")
        print(f"  - Total de objetos: {total_detections}")
        print(f"  - Total de alertas: {total_alerts}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
