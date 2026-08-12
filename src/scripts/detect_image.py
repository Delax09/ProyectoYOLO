"""Detect objects in images using the new service-oriented architecture."""
import argparse
import cv2
from pathlib import Path
from src.core.config_manager import get_config_manager
from src.services import DetectionService
from src.utils.logging_utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detección de objetos en imágenes con YOLOv8 (Arquitectura Escalable)'
    )
    parser.add_argument(
        '--image', '-i', 
        required=True, 
        help='Ruta a la imagen de entrada'
    )
    parser.add_argument(
        '--config', '-c',
        default='config/app.yaml',
        help='Ruta al archivo de configuración (default: config/app.yaml)'
    )
    parser.add_argument(
        '--output', '-o',
        default='output/detection_result',
        help='Ruta de salida (sin extensión)'
    )
    parser.add_argument(
        '--target-class', '-t',
        default='person',
        help='Clase a contar/filtrar (default: person)'
    )
    parser.add_argument(
        '--show', 
        action='store_true',
        help='Mostrar imagen anotada'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger('detect_image', log_file='logs/detect_image.log')
    logger.info("=== Iniciando detección de imágenes ===")
    logger.info(f"Imagen: {args.image}")
    logger.info(f"Configuración: {args.config}")

    try:
        # Cargar configuración
        config_manager = get_config_manager(Path(args.config))
        
        # Inicializar servicio de detección
        detection_service = DetectionService(config_manager)
        logger.info(f"Modelo: {detection_service.get_detector_info()}")
        
        # Procesar imagen
        detections = detection_service.detect_image(args.image)
        logger.info(f"Detecciones encontradas: {len(detections)}")
        
        # Filtrar por clase si es necesario
        filtered_detections = [
            d for d in detections 
            if d.class_name == args.target_class
        ]
        
        logger.info(f"Instancias de '{args.target_class}': {len(filtered_detections)}")
        
        # Anotar imagen
        img = cv2.imread(args.image)
        annotated = detection_service.annotate_frame(img, detections)
        
        # Agregar texto de conteo
        cv2.putText(
            annotated,
            f"{args.target_class}: {len(filtered_detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )
        
        # Guardar resultado
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        image_output = output_path.parent / f"{output_path.name}_annotated.jpg"
        cv2.imwrite(str(image_output), annotated)
        logger.info(f"Imagen anotada guardada: {image_output}")
        
        # Guardar resultados en JSON/CSV
        detection_service.save_results(
            detections,
            [],
            output_path.name,
            metadata={'image_path': args.image}
        )
        
        # Mostrar si se solicitó
        if args.show:
            cv2.imshow('Detecciones', annotated)
            print(f"Presiona cualquier tecla para salir...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"\n✓ Detección completada!")
        print(f"  - Total de objetos: {len(detections)}")
        print(f"  - {args.target_class}: {len(filtered_detections)}")
        print(f"  - Salida: {image_output}")
        
    except Exception as e:
        logger.error(f"Error durante la detección: {e}", exc_info=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
