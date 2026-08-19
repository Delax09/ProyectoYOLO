import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara con DSHOW índice 0.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Reconocimiento YOLO en Tiempo Real", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Reconocimiento YOLO en Tiempo Real", 640, 480)

for _ in range(10):
    cap.read()

while cap.isOpened():
    exito, fotograma = cap.read()
    if exito:
        cv2.imwrite("frame_test.jpg", fotograma)
        print("min:", fotograma.min(), "max:", fotograma.max())

    if exito:
        resultados = model(fotograma)
        fotograma_anotado = resultados[0].plot()
        cv2.imshow("Reconocimiento YOLO en Tiempo Real", fotograma_anotado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("No se pudo leer el fotograma")
        break

cap.release()
cv2.destroyAllWindows()