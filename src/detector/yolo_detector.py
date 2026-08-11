"""
Detector de objetos en tiempo real con YOLOv8
Detección en webcam o archivos de video
"""

import cv2
from ultralytics import YOLO
import argparse
import time
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


def main():
    parser = argparse.ArgumentParser(description='Detección YOLO en tiempo real')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Modelo YOLO a usar (default: yolov8n.pt)')
    parser.add_argument('--source', type=str, default='0',
                        help='Fuente: 0=webcam, ruta del video, o URL (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Umbral de confianza (0-1, default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Ruta para guardar video procesado (opcional)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Máximo de frames a procesar (default: todos)')
    parser.add_argument('--scan-devices', action='store_true',
                        help='Escanear dispositivos de cámara disponibles y salir')

    args = parser.parse_args()
    source = 0 if args.source == '0' else args.source

    logger = setup_logger('yolo_detector', log_file='logs/yolo_detector.log')
    logger.info('\n Iniciando detector YOLO')
    detector = YOLODetector(model_name=args.model, confidence=args.conf, logger=logger)
    if args.scan_devices:
        detector.scan_available_cameras(10)
        return

    detector.detect_in_video(source=source, output_path=args.output,
                            max_frames=args.max_frames)


if __name__ == '__main__':
    main()
