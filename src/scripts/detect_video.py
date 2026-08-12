"""Video detection with alert system using the new architecture."""
import argparse
import cv2
import json
from pathlib import Path
from typing import List, Tuple
from src.core.config_manager import get_config_manager
from src.services import DetectionService
from src.alerts import Alert
from src.utils.logging_utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detección en video con sistema de alertas'
    )
    parser.add_argument(
        '--source', '-s',
        required=True,
        help='Fuente de video (archivo, carpeta con imágenes, o cámara)'
    )
    parser.add_argument(
        '--config', '-c',
        default='config/app.yaml',
        help='Archivo de configuración'
    )
    parser.add_argument(
        '--zones',
        default='config/zones.json',
        help='Archivo JSON con zonas restringidas'
    )
    parser.add_argument(
        '--output', '-o',
        default='output/video_result.mp4',
        help='Archivo de video de salida'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Máximo número de frames a procesar'
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
    logger = setup_logger('detect_video', log_file='logs/detect_video.log')
    logger.info("=== Iniciando detección en video ===")
    logger.info(f"Fuente: {args.source}")
    
    try:
        # Cargar configuración
        config_manager = get_config_manager(Path(args.config))
        detection_service = DetectionService(config_manager)
        
        # Abrir video
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            logger.error(f"No se pudo abrir la fuente: {args.source}")
            raise SystemExit(1)
        
        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {width}x{height} @ {fps} FPS")
        
        # Inicializar sistema de alertas
        zones = load_zones(args.zones)
        if zones:
            detection_service.init_alert_system(zones, (height, width))
            logger.info(f"Sistema de alertas inicializado con {len(zones)} puntos")
        else:
            logger.warning("No se cargaron zonas restringidas")
        
        # Preparar escritor de video
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Procesar frames
        frame_count = 0
        total_detections = 0
        total_alerts = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if args.max_frames and frame_count > args.max_frames:
                break
            
            # Procesar frame
            detections, alerts = detection_service.process_frame(frame)
            total_detections += len(detections)
            total_alerts += len(alerts)
            
            # Anotar frame
            annotated = detection_service.annotate_frame(frame, detections, alerts)
            
            # Agregar información en la esquina
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
            
            # Escribir frame
            out.write(annotated)
            
            if frame_count % 30 == 0:
                logger.info(f"Procesados {frame_count} frames")
        
        cap.release()
        out.release()
        
        logger.info(f"Procesamiento completado: {frame_count} frames")
        logger.info(f"Total de detecciones: {total_detections}")
        logger.info(f"Total de alertas: {total_alerts}")
        logger.info(f"Video guardado: {output_path}")
        
        print(f"\n✓ Procesamiento completado!")
        print(f"  - Frames procesados: {frame_count}")
        print(f"  - Total de objetos: {total_detections}")
        print(f"  - Total de alertas: {total_alerts}")
        print(f"  - Salida: {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
