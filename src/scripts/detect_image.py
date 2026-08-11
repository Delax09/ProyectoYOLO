import argparse
import cv2
from ultralytics import YOLO
from src.utils.logging_utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Detectar personas en una imagen con YOLOv8')
    parser.add_argument('--image', '-i', required=True, help='Ruta a la imagen de entrada')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='Ruta al archivo de modelo YOLO (default: yolov8n.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.5, help='Umbral de confianza (0-1)')
    parser.add_argument('--output', '-o', default='output.jpg', help='Ruta de la imagen anotada de salida')
    parser.add_argument('--target-class', '-t', default='person', help='Nombre de la clase a contar (default: person)')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger('detect_image', log_file='logs/detect_image.log')
    logger.info(f"Cargando modelo: {args.model}")

    try:
        model = YOLO(args.model)
    except Exception as exc:
        logger.error(f"No se pudo cargar el modelo: {exc}")
        raise

    class_names = model.names
    logger.info(f"Clases cargadas: {len(class_names)}")

    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"No se pudo leer la imagen: {args.image}")
        raise SystemExit(1)

    logger.info(f"Ejecutando inferencia sobre: {args.image}")
    results = model(img, conf=args.conf)

    if results is None or len(results) == 0:
        logger.info("No se obtuvieron resultados de inferencia")
        annotated = img
        count = 0
    else:
        r = results[0]
        boxes = r.boxes
        count = 0
        for box in boxes:
            cls_id = int(box.cls[0])
            name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else class_names[cls_id]
            if name == args.target_class:
                count += 1

        annotated = r.plot()

    logger.info(f"Número de '{args.target_class}': {count}")

    cv2.putText(annotated,
                f"{args.target_class}: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2)

    cv2.imwrite(args.output, annotated)
    logger.info(f"Imagen anotada guardada en: {args.output}")
    print(count)


if __name__ == '__main__':
    main()
