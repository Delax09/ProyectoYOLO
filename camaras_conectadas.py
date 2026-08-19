import cv2

for backend, nombre in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")]:
    print(f"--- Backend: {nombre} ---")
    for i in range(4):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            exito, frame = cap.read()
            if exito:
                print(f"  Índice {i}: OK, shape={frame.shape}, min/max={frame.min()}/{frame.max()}")
            else:
                print(f"  Índice {i}: abre pero no lee frame")
        else:
            print(f"  Índice {i}: no abre")
        cap.release()