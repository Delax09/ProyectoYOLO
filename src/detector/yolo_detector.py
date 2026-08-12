import cv2
import argparse
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.utils.logging_utils import setup_logger

"""
Para los Bounding Boxes, YOLO devuelve las coordenadas
y necesitamos calcular los centroides de cada caja con la siguiente formula: 

Calculos de Centroides 

c_x = (x1 + x2) / 2
c_y = (y1 + y2) / 2
"""

class ZoneAnalyzer:
    def __init__(self, zone_polygon):
        self.zone = np.array(zone_polygon, np.int32)
        self.zone = self.zone.reshape((-1, 1, 2))

    def is_point_in_zone(self, point):
        result = cv2.pointPolygonTest(self.zone, point, measureDist=False)
        return result >= 0

    def draw_zone(self, frame, is_alert=False):
        color = (0, 0, 255) if is_alert else (255, 0, 0) # Rojo (Alerta) o Azul (Normal)
        cv2.polylines(frame, [self.zone], isClosed=True, color=color, thickness=2)
        
        # Efecto transparente
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.zone], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

def resolve_output_path(output_path, source_path=None, default_ext='.jpg'):
    """Resuelve la ruta de salida para archivos o carpetas."""
    output = Path(output_path) if output_path is not None else Path('output')

    if output.suffix:
        return output

    if source_path is not None:
        source_name = Path(source_path).name
        resolved = output / f"out_{source_name}"
        if not resolved.suffix:
            resolved = resolved.with_suffix(default_ext)
        return resolved

    return output / f"output{default_ext}"


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
        """Método para procesar imágenes estáticas"""
        self.logger.info(f"Abriendo fuente de imagen: {source_path}")
        frame = cv2.imread(str(source_path))
        
        if frame is None:
            self.logger.error(f"Error: No se pudo abrir la imagen {source_path}")
            return

        try:
            results = self.model(frame, conf=self.confidence)
        except Exception as exc:
            self.logger.error(f"Error en la inferencia YOLO: {exc}")
            return

        if results is None or len(results) == 0:
            self.logger.info("Inferencia completada sin resultados")
            return

        annotated_frame = results[0].plot()
        detections = results[0].boxes
        self.logger.info(f"Imagen procesada: {len(detections)} detecciones encontradas")

        cv2.putText(annotated_frame,
                    f"Detecciones: {len(detections)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        final_output = resolve_output_path(output_path, source_path=source_path, default_ext='.jpg')
        if final_output is not None:
            final_output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(final_output), annotated_frame)
            self.logger.info(f"Guardando imagen procesada en: {final_output}")
        else:
            cv2.imshow('YOLOv8 - Imagen', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def detect_in_video(self, source=0, output_path=None, max_frames=None):
        """Método para procesar video con Object Tracking (ByteTrack)"""
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
            self.logger.error(
                f"Error: No se pudo abrir {source}. Dispositivos detectados: {available}. Verifica que la webcam esté disponible y que ninguna otra aplicación la use."
            )
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 24.0
            self.logger.warning(f"FPS no válido, usando valor por defecto {fps}")

        self.logger.info(f"Video: {width}x{height} @ {fps} FPS")

        writer = None
        final_output = resolve_output_path(output_path, source_path=source, default_ext='.mp4')
        if final_output is not None:
            final_output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(final_output), fourcc, fps, (width, height))
            self.logger.info(f"Guardando en: {final_output}")

        frame_count = 0

        #Definimos los puntos de nuestra zona (puedes ajustarlos a tu cámara)
        #Formato: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        zone_points = [(150, 150), (450, 150), (550, 400), (50, 400)]
        zone_analyzer = ZoneAnalyzer(zone_points)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Fin del video o no se pudo leer el frame")
                    break

                if frame is None:
                    self.logger.warning("Frame leído es None")
                    continue

                if max_frames and frame_count >= max_frames:
                    self.logger.info(f"Máximo de frames alcanzado: {max_frames}")
                    break

                try:
                    # Inferencia usando Tracking
                    results = self.model.track(
                        frame, 
                        conf=self.confidence, 
                        persist=True, 
                        tracker="bytetrack.yaml",
                        verbose=False
                    )
                except Exception as exc:
                    self.logger.error(f"Error en la inferencia YOLO: {exc}")
                    break

                if results is None or len(results) == 0:
                    self.logger.info("Inferencia completada sin resultados")
                    frame_count += 1
                    continue

                # --- INICIO DE NUEVA LÓGICA DE BOUNDING BOXES Y ZONAS ---
                
                annotated_frame = frame.copy() # Copiamos el frame original para dibujar manualmente
                detections = results[0].boxes
                
                track_ids = detections.id.int().cpu().tolist() if detections.id is not None else []
                objects_in_zone = 0

                if detections is not None and len(detections) > 0:
                    coords = detections.xyxy.cpu().numpy() # Extraemos coordenadas x1, y1, x2, y2
                    
                    # Iteramos sobre cada Bounding Box detectada
                    for i, box in enumerate(coords):
                        x1, y1, x2, y2 = map(int, box)
                        track_id = track_ids[i] if i < len(track_ids) else None
                        
                        # Definimos el punto de interés (el centro inferior de la caja)
                        cx = int((x1 + x2) / 2)
                        cy = int(y2)
                        
                        # Verificamos si el objeto tocó la zona
                        in_zone = zone_analyzer.is_point_in_zone((cx, cy))
                        
                        if in_zone:
                            objects_in_zone += 1
                            color = (0, 0, 255) # Bounding Box Roja si invade
                        else:
                            color = (0, 255, 0) # Bounding Box Verde si está fuera

                        # Dibujamos nuestra propia Bounding Box y el punto de referencia
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(annotated_frame, (cx, cy), 5, color, -1)
                        
                        # Dibujamos el ID
                        label = f"ID: {track_id}" if track_id is not None else "Obj"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Dibujamos el polígono en el frame (cambia de color si hay objetos dentro)
                zone_analyzer.draw_zone(annotated_frame, is_alert=(objects_in_zone > 0))

                # --- FIN DE NUEVA LÓGICA ---

                # Logs de consola
                if frame_count % 30 == 0:
                    self.logger.info(f"Frame {frame_count}: {len(detections)} detecciones. IDs: {track_ids}. En Zona: {objects_in_zone}")

                # Textos generales en pantalla
                cv2.putText(annotated_frame, f"Rastreados: {len(track_ids)} | En Zona: {objects_in_zone}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('YOLOv8 - Tracking y Zonas', annotated_frame)

                if writer:
                    writer.write(annotated_frame)

                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("Saliendo por pulsar 'q'")
                    break

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.logger.info(f"Total de frames procesados: {frame_count}")


def process_examples_folder(detector, input_folder='examples', output_folder='output'):
    """Recorre la carpeta y procesa todos los archivos de imagen y video"""
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
            else:
                detector.logger.warning(f"Formato no soportado, omitiendo: {file_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Detección YOLO en tiempo real y por lotes con Tracking')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Modelo YOLO a usar (default: yolov8n.pt)')
    parser.add_argument('--source', type=str, default='examples',
                        help='Fuente: 0=webcam, ruta del video/imagen, o "examples" para procesar carpeta (default: examples)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Umbral de confianza (0-1, default: 0.5)')
    parser.add_argument('--output', type=str, default='output',
                        help='Ruta de salida (default: carpeta "output")')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Máximo de frames a procesar en videos (default: todos)')
    parser.add_argument('--scan-devices', action='store_true',
                        help='Escanear dispositivos de cámara disponibles y salir')

    args = parser.parse_args()

    logger = setup_logger('yolo_detector', log_file='logs/yolo_detector.log')
    logger.info('\n Iniciando detector YOLO')
    
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
        if source_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            detector.detect_in_image(args.source, output_path=args.output)
        else:
            detector.detect_in_video(source=args.source, output_path=args.output, max_frames=args.max_frames)
    else:
        logger.error(f"Fuente '{args.source}' no reconocida o no existe.")

if __name__ == '__main__':
    main()