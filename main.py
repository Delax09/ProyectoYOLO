import sys
from pathlib import Path

# Añadimos la raíz del proyecto para que Python encuentre la carpeta src/ sin problemas
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import time
import json
import cv2
import numpy as np

from src.core.detector import YOLODetector
from src.analytics.alert_manager import AlertManager
from src.utils.visualizer import Visualizer
from src.io.report_writer import ReportWriter
from src.utils.logging_utils import setup_logger

def load_zone_config(config_path="config/zonas.json"):
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            first_zone = data["zonas"][0]
            return [tuple(p) for p in first_zone["puntos"]], first_zone["nombre"]
    return [(150, 150), (450, 150), (550, 400), (50, 400)], "Zona Predeterminada"

def process_video(source, output_dir, detector, zone_points, zone_name):
    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    alert_manager = AlertManager(zone_points, frame_shape=(height, width))
    
    writer = None
    if output_dir:
        out_path = Path(output_dir) / f"out_{Path(str(source)).name}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.track_frame(frame)
            if not results or len(results) == 0:
                continue

            detections = results[0].boxes
            curr_time = time.time()
            active_alerts = alert_manager.evaluate_detections(detections, curr_time)
            alert_ids = [a["track_id"] for a in active_alerts]

            # Renderizado
            annotated_frame = frame.copy()
            Visualizer.draw_detections(annotated_frame, detections, alert_ids)
            Visualizer.draw_zone(annotated_frame, zone_points, is_alert=len(active_alerts) > 0)

            cv2.imshow("YOLO System - Video", annotated_frame)
            if writer:
                writer.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

def process_image(source, output_dir, detector, zone_points, zone_name):
    frame = cv2.imread(str(source))
    if frame is None:
        detector.logger.error(f"Error: No se pudo abrir la imagen {source}")
        return

    height, width = frame.shape[:2]
    alert_manager = AlertManager(zone_points, frame_shape=(height, width))
    
    results = detector.predict_image(frame)
    if not results or len(results) == 0:
        detector.logger.info("Inferencia completada sin resultados")
        return

    detections = results[0].boxes
    annotated_frame = frame.copy()
    
    reporte_data = {
        "archivo_origen": str(source),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dimensiones": {"alto": height, "ancho": width},
        "zona_evaluada": zone_name,
        "resumen": {
            "total_detecciones": len(detections),
            "objetos_en_zona": 0
        },
        "detecciones": []
    }

    if detections is not None and len(detections) > 0:
        coords = detections.xyxy.cpu().numpy()
        confs = detections.conf.cpu().numpy()
        clss = detections.cls.cpu().numpy()

        for i, (box, conf, cls) in enumerate(zip(coords, confs, clss)):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int(y2)
            
            class_name = detector.class_names.get(int(cls), "Desconocido")
            in_zone = alert_manager.is_point_in_zone(cx, cy)

            if in_zone:
                reporte_data["resumen"]["objetos_en_zona"] += 1
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            reporte_data["detecciones"].append({
                "id_int_frame": i + 1,
                "clase": class_name,
                "confianza": float(round(conf, 2)),
                "bbox": [x1, y1, x2, y2],
                "punto_contacto": [cx, cy],
                "en_zona_restringida": bool(in_zone)
            })

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated_frame, (cx, cy), 5, color, -1)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Dibujar polígono estático
    pts = np.array(zone_points, np.int32).reshape((-1, 1, 2))
    zone_color = (0, 0, 255) if reporte_data["resumen"]["objetos_en_zona"] > 0 else (255, 0, 0)
    cv2.polylines(annotated_frame, [pts], isClosed=True, color=zone_color, thickness=2)
    cv2.putText(annotated_frame, f"Total: {len(detections)} | En Zona: {reporte_data['resumen']['objetos_en_zona']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if output_dir:
        out_path = Path(output_dir) / f"out_{Path(str(source)).name}"
        ReportWriter.save_image_report(str(out_path), annotated_frame, reporte_data)
        detector.logger.info(f"Imagen y Reporte JSON guardados en: {output_dir}")
    else:
        cv2.imshow("YOLO System - Imagen", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Sistema de Monitoreo YOLOv8")
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--source', type=str, default='examples/VideoPersonas.mp4')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='output')

    args = parser.parse_args()
    logger = setup_logger('main_runner')
    
    detector = YOLODetector(model_name=args.model, confidence=args.conf, logger=logger)
    zone_pts, zone_name = load_zone_config()

    src_path = Path(args.source)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    image_extensions = {'.jpg', '.jpeg', '.png'}

    # Lógica de enrutamiento (Webcam vs Video vs Imagen)
    if args.source.isdigit():
        process_video(int(args.source), args.output, detector, zone_pts, zone_name)
    elif src_path.exists() and src_path.is_file():
        ext = src_path.suffix.lower()
        if ext in video_extensions:
            process_video(args.source, args.output, detector, zone_pts, zone_name)
        elif ext in image_extensions:
            process_image(args.source, args.output, detector, zone_pts, zone_name)
        else:
            logger.error(f"Extensión no soportada: {ext}")
    else:
        logger.error(f"No se encontró la fuente: {args.source}")

if __name__ == '__main__':
    main()