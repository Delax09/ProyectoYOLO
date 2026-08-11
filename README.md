# Proyecto de Visión por Computadora con YOLOv8

Detección de objetos con YOLOv8. Este repositorio está organizado para facilitar desarrollo, pruebas y despliegue.

ejecucion en la consonal 
- python -m src.detector.yolo_detector --source examples
- python -m src.detector.yolo_detector --source examples/VideoPersonas.mp4

**Estructura del proyecto (propuesta aplicada)**

- `configs/` : configuraciones YAML/JSON (ej. `data_example.yaml`).
- `data/` : datasets y anotaciones (no versionar datos pesados).
- `models/` : pesos y checkpoints (ej. `yolov8n.pt`, modelos entrenados).
- `src/` : código fuente del proyecto.
  - `src/detector/` : lógica del detector (`yolo_detector.py`).
  - `src/scripts/` : scripts ejecutables (`detect_image.py`, `camera_test.py`, `train_custom_model.py`).
  - `src/utils/` : utilidades (`logging_utils.py`, helpers).
- `examples/` : imágenes y resultados de ejemplo (`ImagenPersonas.jpg`, `salida.jpg`).
- `logs/` : archivos de log generados por los scripts.
- `tests/` : pruebas unitarias e integración.
- `requirements.txt`, `README.md`, `.gitignore`.

## 📋 Requisitos

- Python 3.8+ (se probó con 3.14 en este entorno).
- Recomendado: GPU + drivers/CUDA compatibles para acelerar inferencia/entrenamiento.

## 🚀 Instalación Rápida

1) Crear y activar entorno virtual (Windows):

```powershell
python -m venv venv
venv\Scripts\activate
```

2) Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Si necesita instalar manualmente:

```powershell
pip install ultralytics opencv-python numpy torch torchvision
```

## 🧭 Comandos de uso (nuevas rutas)

Los scripts ahora están en `src/scripts/`. Puede ejecutarlos de dos maneras:

- Directo (desde la raíz del repo):

```powershell
python src\scripts\detect_image.py --image examples\ImagenPersonas.jpg --output examples\salida.jpg
```

- Módulo (recomendado para imports de paquete):

```powershell
python -m src.scripts.detect_image --image examples\ImagenPersonas.jpg --output examples\salida.jpg
```

Ejemplos útiles:

- Detección en imagen (cuenta `person` por defecto):

```powershell
python -m src.scripts.detect_image --image examples\ImagenPersonas.jpg --output examples\salida.jpg
```

- Probar cámara:

```powershell
python -m src.scripts.camera_test --max-index 3
```

- Entrenar modelo personalizado:

```powershell
python -m src.scripts.train_custom_model --data configs\data_example.yaml --epochs 50
```

## 🔧 Parámetros principales

- `--model`: ruta al archivo del modelo (ej. `models/yolov8n.pt`).
- `--conf`: umbral de confianza (0-1). Default: `0.5`.
- `--target-class`: en `detect_image.py` permite seleccionar la clase a contar (default: `person`).

## 📄 Logs

Los scripts generan logs en `logs/` automáticamente. Ejemplos:

- `logs/detect_image.log`
- `logs/yolo_detector.log`
- `logs/train_custom_model.log`

## 📁 Ejemplos y pruebas rápidas

Para probar rápidamente con la imagen incluida:

```powershell
python -m src.scripts.detect_image --image examples\ImagenPersonas.jpg --output examples\salida.jpg
# El comando imprimirá el conteo y guardará la imagen anotada en examples\salida.jpg
```

## 📝 Notas

- Mantenga `models/` y `data/` fuera del control de versiones si contienen archivos grandes; use `.gitignore`.
- Para producción, considere empaquetar `src/` como módulo y añadir un `pyproject.toml`.

Si quieres, puedo:
- Añadir un `pyproject.toml` o `setup.cfg` para empaquetar el proyecto.
- Crear scripts de CI para tests y linter.
