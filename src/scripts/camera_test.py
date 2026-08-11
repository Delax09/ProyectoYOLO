import cv2
import argparse
import time


def test_camera(index: int, timeout: float = 2.0) -> bool:
    print(f"Probando cámara {index}...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"  No se pudo abrir la cámara {index}")
        return False

    start = time.time()
    while time.time() - start < timeout:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue
        print(f"  Cámara {index} OK - frame leído: shape={frame.shape}, dtype={frame.dtype}")
        cap.release()
        return True

    print(f"  Cámara {index} abrió pero no devolvió frames válidos en {timeout} segundos")
    cap.release()
    return False


def scan_cameras(max_index: int = 5):
    found = []
    print(f"Escaneando cámaras del 0 al {max_index}...")
    for idx in range(max_index + 1):
        if test_camera(idx):
            found.append(idx)
    print(f"Cámaras válidas detectadas: {found}")
    return found


def main():
    parser = argparse.ArgumentParser(description="Prueba sencilla de cámaras OpenCV")
    parser.add_argument("--max-index", type=int, default=5,
                        help="Índice máximo de cámara a escanear")
    parser.add_argument("--index", type=int, default=None,
                        help="Probar sólo una cámara específica")
    args = parser.parse_args()

    if args.index is not None:
        test_camera(args.index)
    else:
        scan_cameras(args.max_index)


if __name__ == "__main__":
    main()
