from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data= 'data.yaml', #Ruta al archivo .yaml
    epochs = 100, #Epocas para entrenar
    imgsz = 640, #Resolucion de la imagen
    batch = 8, #Imagenes procesadas a la vez (bajar si me quedo sin memoria)
    name= 'maiz_yolov8s',
)