# main.py
import argparse
import time
import json
import cv2
from pathlib import Path
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

            cv2.imshow("YOLO System", annotated_frame)
            if writer:
                writer.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
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
    if args.source.isdigit() or (src_path.exists() and src_path.is_file() and src_path.suffix.lower() in {'.mp4', '.avi', '.mov'}):
        process_video(args.source if not args.source.isdigit() else int(args.source), args.output, detector, zone_pts, zone_name)

if __name__ == '__main__':
    main()