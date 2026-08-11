import cv2
import argparse
import time
import os
from pathlib import Path
from ultralytics import YOLO
from src.utils.logging_utils import setup_logger

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
        """Nuevo método para procesar imágenes estáticas"""
        self.logger.info(f"Abriendo fuente de imagen: {source_path}")
        frame = cv2.imread(source_path)
        
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

        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            self.logger.info(f"Guardando imagen procesada en: {output_path}")
        else:
            cv2.imshow('YOLOv8 - Imagen', annotated_frame)
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
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            self.logger.info(f"Guardando en: {output_path}")

        frame_count = 0

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
                    results = self.model(frame, conf=self.confidence)
                except Exception as exc:
                    self.logger.error(f"Error en la inferencia YOLO: {exc}")
                    break

                if results is None or len(results) == 0:
                    self.logger.info("Inferencia completada sin resultados")
                    frame_count += 1
                    continue

                annotated_frame = results[0].plot()
                detections = results[0].boxes
                self.logger.info(f"Frame {frame_count}: {len(detections)} detecciones")

                cv2.putText(annotated_frame,
                            f"Detecciones: {len(detections)}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                cv2.imshow('YOLOv8 - Detección en Tiempo Real', annotated_frame)

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
    
    # Extensiones soportadas
    image_extensions = {'.jpg', '.jpeg', '.png'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

    if not input_path.exists():
        detector.logger.error(f"La carpeta '{input_folder}' no existe.")
        return

    # Crear carpeta de salida si no existe
    output_path.mkdir(parents=True, exist_ok=True)
    detector.logger.info(f"Procesando recursos en '{input_folder}', guardando resultados en '{output_folder}'")

    for file_path in input_path.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lower()
            out_file = output_path / f"out_{file_path.name}"

            if ext in image_extensions:
                detector.logger.info(f"--- Procesando Imagen: {file_path.name} ---")
                detector.detect_in_image(str(file_path), output_path=str(out_file))
                
            elif ext in video_extensions:
                detector.logger.info(f"--- Procesando Video: {file_path.name} ---")
                detector.detect_in_video(source=str(file_path), output_path=str(out_file))
            else:
                detector.logger.warning(f"Formato no soportado, omitiendo: {file_path.name}")

def main():
    parser = argparse.ArgumentParser(description='Detección YOLO en tiempo real y por lotes')
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

    # Logica de ruteo según la fuente (carpeta, imagen, video o webcam)
    source_path = Path(args.source)
    
    if args.source.isdigit():
        # Es una webcam
        detector.detect_in_video(source=int(args.source), output_path=args.output, max_frames=args.max_frames)
    elif source_path.is_dir():
        # Es una carpeta
        process_examples_folder(detector, input_folder=args.source, output_folder=args.output)
    elif source_path.is_file():
        # Es un archivo único
        if source_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            detector.detect_in_image(args.source, output_path=args.output)
        else:
            detector.detect_in_video(source=args.source, output_path=args.output, max_frames=args.max_frames)
    else:
        logger.error(f"Fuente '{args.source}' no reconocida o no existe.")

if __name__ == '__main__':
    main()