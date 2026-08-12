import cv2
import argparse
import time
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.utils.logging_utils import setup_logger

class AlertManager:
    def __init__(self, polygon_coords, frame_shape, dwell_threshold_sec=3.0):
        """
        polygon_coords: Lista de puntos [(x1, y1), (x2, y2), ...] definiendo la zona[cite: 5].
        frame_shape: Tupla (height, width) de la imagen[cite: 5].
        dwell_threshold_sec: Tiempo en segundos para activar alerta por permanencia[cite: 5].
        """
        self.dwell_threshold = dwell_threshold_sec
        self.track_timers = {}  # {track_id: timestamp_entrada}
        
        # Crear máscara binaria de la zona restringida para evaluación O(1)[cite: 5]
        self.mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        pts = np.array(polygon_coords, dtype=np.int32)
        cv2.fillPoly(self.mask, [pts], 255)

    def is_point_in_zone(self, x, y):
        h, w = self.mask.shape
        if 0 <= x < w and 0 <= y < h:
            return self.mask[int(y), int(x)] == 255
        return False

    def evaluate_detections(self, detections, current_time):
        active_alerts = []
        current_ids_in_zone = set()

        if detections is None or detections.id is None:
            return active_alerts

        boxes = detections.xyxy.cpu().numpy()
        track_ids = detections.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # Punto de contacto del objeto con el suelo (centro inferior de la Bounding Box)[cite: 5]
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            if self.is_point_in_zone(foot_x, foot_y):
                current_ids_in_zone.add(track_id)
                
                # Registrar tiempo de ingreso si es nuevo[cite: 5]
                if track_id not in self.track_timers:
                    self.track_timers[track_id] = current_time
                
                # Evaluar si supera el umbral de permanencia[cite: 5]
                time_in_zone = current_time - self.track_timers[track_id]
                if time_in_zone >= self.dwell_threshold:
                    active_alerts.append({
                        "track_id": track_id,
                        "bbox": [x1, y1, x2, y2],
                        "dwell_time": time_in_zone
                    })

        # Limpiar IDs que ya salieron de la zona[cite: 5]
        ids_to_remove = [tid for tid in self.track_timers if tid not in current_ids_in_zone]
        for tid in ids_to_remove:
            del self.track_timers[tid]

        return active_alerts

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt', confidence=0.5, logger=None):
        if logger is None:
            logger = setup_logger('yolo_detector', log_file='logs/yolo_detector.log')
        self.logger = logger
        self.logger.info(f"Cargando modelo {model_name}...")
        self.model = YOLO(model_name)
        self.logger.info(type(self.model))
        self.confidence = confidence
        self.class_names = self.model.names
        self.logger.info(f"Modelo cargado con {len(self.class_names)} clases")

    def scan_available_cameras(self, max_index=5):
        available = []
        self.logger.info("Escaneando dispositivos de cámara disponibles...")
        for idx in range(max_index + 1):
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret and frame is not None and frame.size > 0:
                    available.append(idx)
                test_cap.release()
        self.logger.info(f"Cámaras detectadas: {available}")
        return available

    def detect_in_image(self, source_path, output_path=None):
        self.logger.info(f"Abriendo fuente de imagen: {source_path}")
        frame = cv2.imread(str(source_path))
        
        if frame is None:
            self.logger.error(f"Error: No se pudo abrir la imagen {source_path}")
            return

        try:
            # Puedes filtrar por personas añadiendo classes=[0] si lo deseas
            results = self.model(frame, conf=self.confidence)
        except Exception as exc:
            self.logger.error(f"Error en la inferencia YOLO: {exc}")
            return

        if results is None or len(results) == 0:
            self.logger.info("Inferencia completada sin resultados")
            return

        height, width = frame.shape[:2]

        # --- CARGAR ZONA DESDE JSON O DEFECTO ---
        config_path = Path("config/zonas.json")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                primera_zona = config_data["zonas"][0]
                zone_points = [tuple(pt) for pt in primera_zona["puntos"]]
                zona_nombre = primera_zona["nombre"]
        else:
            zone_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
            zona_nombre = "Zona Predeterminada"

        alert_manager = AlertManager(zone_points, frame_shape=(height, width))

        annotated_frame = frame.copy()
        detections = results[0].boxes
        
        # Estructura del reporte que se guardará en JSON
        reporte_data = {
            "archivo_origen": str(source_path),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dimensiones": {"alto": height, "ancho": width},
            "zona_evaluada": zona_nombre,
            "resumen": {
                "total_detecciones": len(detections),
                "objetos_en_zona": 0,
                "personas_en_zona_restringida": 0,
                "personas_fuera_zona_restringida": 0
            },
            "detecciones": []
        }

        if detections is not None and len(detections) > 0:
            coords = detections.xyxy.cpu().numpy()
            confs = detections.conf.cpu().numpy()
            clss = detections.cls.cpu().numpy()

            for i, (box, conf, cls) in enumerate(zip(coords, confs, clss)):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int(y2)  # Punto de contacto en el suelo
                
                class_name = self.class_names.get(int(cls), "Desconocido")
                in_zone = alert_manager.is_point_in_zone(cx, cy)

                if in_zone:
                    reporte_data["resumen"]["objetos_en_zona"] += 1
                    if class_name == "person":
                        reporte_data["resumen"]["personas_en_zona_restringida"] += 1
                    color = (0, 0, 255) # Rojo si está en la zona
                else:
                    if class_name == "person":
                        reporte_data["resumen"]["personas_fuera_zona_restringida"] += 1
                    color = (0, 255, 0) # Verde si está fuera

                # Agregar detalle de cada detección al reporte JSON
                reporte_data["detecciones"].append({
                    "id_int_frame": i + 1,
                    "clase": class_name,
                    "confianza": float(round(conf, 2)),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "punto_contacto": [int(cx), int(cy)],
                    "en_zona_restringida": bool(in_zone)
                })

                # Dibujar en la imagen
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated_frame, (cx, cy), 5, color, -1)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Dibujar polígono
        pts = np.array(zone_points, np.int32).reshape((-1, 1, 2))
        zone_color = (0, 0, 255) if reporte_data["resumen"]["objetos_en_zona"] > 0 else (255, 0, 0)
        cv2.polylines(annotated_frame, [pts], isClosed=True, color=zone_color, thickness=2)

        # Leyenda en la imagen
        cv2.putText(annotated_frame, f"Total: {len(detections)} | En Zona: {reporte_data['resumen']['objetos_en_zona']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- GUARDAR IMAGEN Y ARVO JSON DE REPORTE ---
        if output_path:
            out_img_path = Path(output_path)
            # Guardar la imagen con las marcas
            cv2.imwrite(str(out_img_path), annotated_frame)
            
            # Guardar el archivo JSON con el mismo nombre pero extensión .json
            json_report_path = out_img_path.with_suffix('.json')
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(reporte_data, f, indent=4, ensure_ascii=False)

            self.logger.info(f"Imagen guardada en: {out_img_path}")
            self.logger.info(f"Reporte JSON guardado en: {json_report_path}")
        else:
            cv2.imshow('YOLOv8 - Análisis Espacial', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def detect_in_video(self, source=0, output_path=None, max_frames=None):
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.logger.info(f"Abriendo fuente de video: {source}")

        def warmup_capture(cap, backend_name):
            for attempt in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.logger.info(f"Warmup exitoso con backend {backend_name} en intento {attempt + 1}")
                    return True
                self.logger.warning(f"Warmup intento {attempt + 1} fallido con backend {backend_name}")
                time.sleep(0.2)
            return False

        def open_capture(src):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None] if isinstance(src, int) else [None]
            for backend in backends:
                if backend is None:
                    cap = cv2.VideoCapture(src)
                    name = "default"
                else:
                    cap = cv2.VideoCapture(src, backend)
                    name = "DirectShow" if backend == cv2.CAP_DSHOW else "MediaFoundation"

                if not cap.isOpened():
                    self.logger.warning(f"No se pudo abrir con backend {name}")
                    cap.release()
                    continue

                self.logger.info(f"Fuente abierta con backend: {name}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    self.logger.debug(f"El backend {name} no soporta CAP_PROP_BUFFERSIZE")
                if not warmup_capture(cap, name):
                    self.logger.warning(f"Warmup fallido para backend {name}, cerrando y probando otro")
                    cap.release()
                    continue
                return cap, name
            return None, None

        cap, backend = open_capture(source)
        if cap is None or not cap.isOpened():
            available = self.scan_available_cameras(5) if isinstance(source, int) else []
            self.logger.error(f"Error: No se pudo abrir {source}. Dispositivos detectados: {available}.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 24.0
            self.logger.warning(f"FPS no válido, usando valor por defecto {fps}")

        self.logger.info(f"Video: {width}x{height} @ {fps} FPS")

        # --- INICIALIZAR GESTOR DE ALERTAS ESPACIALES ---
        # Coordenadas de ejemplo (ajustar según el video)
        # --- CARGAR ZONAS DESDE JSON ---
        config_path = Path("config/zonas.json")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                # Tomamos la primera zona del archivo
                primera_zona = config_data["zonas"][0]
                zone_points = [tuple(pt) for pt in primera_zona["puntos"]]
                zona_nombre = primera_zona["nombre"]
                self.logger.info(f"Zona cargada exitosamente: {zona_nombre}")
        else:
            self.logger.warning("No se encontró config/zonas.json. Usando zona por defecto.")
            zone_points = [(150, 150), (450, 150), (550, 400), (50, 400)]

        # Inicializamos el gestor de alertas espaciales
        alert_manager = AlertManager(zone_points, frame_shape=(height, width), dwell_threshold_sec=3.0)

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            self.logger.info(f"Guardando en: {output_path}")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if max_frames and frame_count >= max_frames:
                    break

                try:
                    results = self.model.track(frame, 
                                                conf=self.confidence, 
                                                persist=True, 
                                                tracker="bytetrack.yaml", 
                                                classes = [0], #Solo personas 
                                                verbose=False)
                except Exception as exc:
                    self.logger.error(f"Error en la inferencia YOLO: {exc}")
                    break

                if results is None or len(results) == 0:
                    frame_count += 1
                    continue

                annotated_frame = frame.copy()
                detections = results[0].boxes
                
                # --- EVALUACIÓN ESPACIAL Y DE TIEMPO ---
                current_time = time.time()
                active_alerts = alert_manager.evaluate_detections(detections, current_time)
                alert_track_ids = [alert["track_id"] for alert in active_alerts]

                # Contadores de personas por zona
                personas_en_zona = 0
                personas_fuera_zona = 0
                track_ids_list = []
                if detections is not None and len(detections) > 0:
                    coords = detections.xyxy.cpu().numpy()
                    track_ids = detections.id.int().cpu().numpy() if detections.id is not None else [None] * len(coords)
                    
                    for box, track_id in zip(coords, track_ids):
                        if track_id is not None: track_ids_list.append(int(track_id))
                        
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = int((x1 + x2) / 2), int(y2)
                        
                        # Verificar si el punto de contacto está en la zona
                        in_zone = alert_manager.is_point_in_zone(cx, cy)
                        if in_zone:
                            personas_en_zona += 1
                        else:
                            personas_fuera_zona += 1
                        
                        # Colores: Rojo si hay alerta de permanencia, Verde si está normal
                        color = (0, 0, 255) if track_id in alert_track_ids else (0, 255, 0)

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(annotated_frame, (cx, cy), 5, color, -1)
                        
                        label = f"ID: {track_id}" if track_id is not None else "Obj"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Dibujar el polígono de la zona
                pts = np.array(zone_points, np.int32).reshape((-1, 1, 2))
                zone_color = (0, 0, 255) if len(active_alerts) > 0 else (255, 0, 0)
                cv2.polylines(annotated_frame, [pts], isClosed=True, color=zone_color, thickness=2)

                # Imprimir alertas en pantalla
                y_offset = 60
                for alert in active_alerts:
                    alert_text = f"ALERTA ID {alert['track_id']} - Tiempo: {alert['dwell_time']:.1f}s"
                    cv2.putText(annotated_frame, alert_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 25

                if frame_count % 30 == 0:
                    self.logger.info(f"Frame {frame_count}: {len(detections)} det. IDs: {track_ids_list}. Alertas activas: {len(active_alerts)}")

                cv2.putText(annotated_frame, f"En zona: {personas_en_zona} | Fuera: {personas_fuera_zona} | Alertas: {len(active_alerts)}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('YOLOv8 - Control de Zonas y Permanencia', annotated_frame)

                if writer:
                    writer.write(annotated_frame)

                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.logger.info(f"Total de frames procesados: {frame_count}")

def process_examples_folder(detector, input_folder='examples', output_folder='output'):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

    if not input_path.exists():
        detector.logger.error(f"La carpeta '{input_folder}' no existe.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    detector.logger.info(f"Procesando recursos en '{input_folder}', guardando resultados en '{output_folder}'")

    for file_path in input_path.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lower()
            out_file = output_path / f"out_{file_path.name}"

            if ext in image_extensions:
                detector.logger.info(f"--- Procesando Imagen: {file_path.name} ---")
                detector.detect_in_image(source_path=file_path, output_path=out_file)
                
            elif ext in video_extensions:
                detector.logger.info(f"--- Procesando Video: {file_path.name} ---")
                detector.detect_in_video(source=str(file_path), output_path=str(out_file))

def main():
    parser = argparse.ArgumentParser(description='Detección YOLO en tiempo real y por lotes con Tracking')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Modelo YOLO a usar (default: yolov8n.pt)')
    parser.add_argument('--source', type=str, default='examples', help='Fuente: 0=webcam, ruta del video, o carpeta')
    parser.add_argument('--conf', type=float, default=0.5, help='Umbral de confianza (0-1, default: 0.5)')
    parser.add_argument('--output', type=str, default='output', help='Ruta de salida')
    parser.add_argument('--max-frames', type=int, default=None, help='Máximo de frames a procesar en videos')
    parser.add_argument('--scan-devices', action='store_true', help='Escanear cámaras disponibles y salir')

    args = parser.parse_args()

    logger = setup_logger('yolo_detector', log_file='logs/yolo_detector.log')
    detector = YOLODetector(model_name=args.model, confidence=args.conf, logger=logger)
    
    if args.scan_devices:
        detector.scan_available_cameras(10)
        return

    source_path = Path(args.source)
    
    if args.source.isdigit():
        detector.detect_in_video(source=int(args.source), output_path=args.output, max_frames=args.max_frames)
    elif source_path.is_dir():
        process_examples_folder(detector, input_folder=args.source, output_folder=args.output)
    elif source_path.is_file():
        # 1. Crear el directorio de salida si no existe
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Generar el nombre final del archivo con su extensión
        out_file = out_dir / f"out_{source_path.name}"
        
        # 3. Procesar pasando la ruta correcta
        if source_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            detector.detect_in_image(args.source, output_path=str(out_file))
        else:
            detector.detect_in_video(source=str(source_path), output_path=str(out_file), max_frames=args.max_frames)

if __name__ == '__main__':
    main()