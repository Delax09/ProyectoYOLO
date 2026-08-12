# Guía de Migración - Código Antiguo a Nueva Arquitectura

## 📋 Resumen de Cambios

Este documento guía la migración del código monolítico antiguo a la nueva arquitectura modular y escalable.

## 🔄 Comparación Antes/Después

### Antes (Monolítico)

```python
# Código antiguo - Todo mezclado
from ultralytics import YOLO
import cv2
import json

# 1. Cargar modelo
model = YOLO('yolov8n.pt')

# 2. Procesar imagen
img = cv2.imread('image.jpg')
results = model(img, conf=0.5)

# 3. Lógica de alertas (mezclada)
def check_zone(box, polygon):
    # Lógica de alertas aquí
    pass

# 4. Guardar resultados (ad-hoc)
for result in results:
    # Guardar en JSON manualmente
    pass

# 5. Logging ad-hoc
print("Procesando...")
```

### Después (Modular)

```python
# Código nuevo - Arquitectura limpia
from src.core.config_manager import get_config_manager
from src.services import DetectionService
from pathlib import Path

# 1. Inicializar con configuración centralizada
config = get_config_manager(Path('config/app.yaml'))
service = DetectionService(config)

# 2. Procesar imagen (solo una línea)
detections = service.detect_image('image.jpg')

# 3. Sistema de alertas desacoplado
service.init_alert_system(zones, frame_shape)
detections, alerts = service.process_frame(frame)

# 4. Almacenamiento automático
service.save_results(detections, alerts, 'resultado')

# 5. Logging automático y estructurado
# Manejado internamente por los servicios
```

## 🔀 Pasos de Migración

### Paso 1: Instalar Dependencias Nuevas

```powershell
pip install pyyaml
```

### Paso 2: Crear Archivo de Configuración

Crear `config/app.yaml`:

```yaml
detector:
  model_name: 'yolov8n.pt'
  confidence: 0.5
  device: 'cpu'

alerts:
  enabled: true
  dwell_threshold_sec: 3.0

storage:
  type: 'json'
  output_dir: 'output'
```

### Paso 3: Reemplazar Inicialización de Modelo

**Antes:**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
class_names = model.names
confidence = 0.5
```

**Después:**
```python
from src.core.config_manager import get_config_manager
from src.services import DetectionService
from pathlib import Path

config = get_config_manager(Path('config/app.yaml'))
service = DetectionService(config)
```

### Paso 4: Reemplazar Lógica de Detección

**Antes:**
```python
img = cv2.imread('image.jpg')
results = model(img, conf=0.5)

if results:
    r = results[0]
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        name = class_names[cls_id]
        # Procesar...
```

**Después:**
```python
detections = service.detect_image('image.jpg')

for detection in detections:
    print(f"Clase: {detection.class_name}")
    print(f"Confianza: {detection.confidence}")
    print(f"BBox: {detection.bbox}")
```

### Paso 5: Reemplazar Sistema de Alertas

**Antes:**
```python
def evaluate_alerts(boxes, polygon):
    alerts = []
    for box in boxes:
        x1, y1, x2, y2 = box
        foot_x = (x1 + x2) / 2
        foot_y = y2
        
        if is_in_polygon(foot_x, foot_y, polygon):
            # Lógica de alerta
            alerts.append({...})
    
    return alerts
```

**Después:**
```python
# Inicializar sistema de alertas
service.init_alert_system(polygon_coords, frame_shape)

# Evaluar alertas automáticamente
detections, alerts = service.process_frame(frame)

# Las alertas ya están procesadas
for alert in alerts:
    print(f"ALERTA: {alert.alert_id} en track {alert.track_id}")
```

### Paso 6: Reemplazar Guardado de Resultados

**Antes:**
```python
# Guardar JSON manualmente
output_data = {
    'detections': [
        {
            'class': class_names[box.cls],
            'confidence': float(box.conf),
            'bbox': box.xyxy.tolist()
        }
        for box in boxes
    ]
}

with open('output.json', 'w') as f:
    json.dump(output_data, f)
```

**Después:**
```python
# Una sola línea - Todo automático
service.save_results(detections, alerts, 'output')
```

### Paso 7: Reemplazar Anotación de Imágenes

**Antes:**
```python
annotated = results[0].plot()
cv2.putText(annotated, f"Personas: {count}", ...)
cv2.imwrite('output.jpg', annotated)
```

**Después:**
```python
img = cv2.imread('image.jpg')
annotated = service.annotate_frame(img, detections, alerts)
cv2.imwrite('output.jpg', annotated)
```

## 📝 Migración Completa de Script

### Script Antiguo

```python
# detect_old.py
import cv2
from ultralytics import YOLO
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Cargar modelo
    logger.info("Cargando modelo...")
    model = YOLO('yolov8n.pt')
    
    # Cargar imagen
    logger.info("Leyendo imagen...")
    img = cv2.imread('image.jpg')
    
    # Detección
    logger.info("Ejecutando detección...")
    results = model(img, conf=0.5)
    
    # Procesar
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                'class': str(cls_id),
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
    
    # Guardar
    logger.info(f"Encontrados {len(detections)} objetos")
    with open('output.json', 'w') as f:
        json.dump(detections, f)
    
    logger.info("Listo!")

if __name__ == '__main__':
    main()
```

### Script Nuevo (Migrado)

```python
# detect_new.py
from pathlib import Path
from src.core.config_manager import get_config_manager
from src.services import DetectionService

def main():
    # Inicializar con configuración
    config = get_config_manager(Path('config/app.yaml'))
    service = DetectionService(config)
    
    # Detección y guardado en 2 líneas
    detections = service.detect_image('image.jpg')
    service.save_results(detections, [], 'output')

if __name__ == '__main__':
    main()
```

## 🔌 Migración de Componentes Específicos

### Migrar Logger

**Antes:**
```python
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO
)
```

**Después:**
```python
from src.utils.logging_utils import setup_logger

logger = setup_logger('mi_modulo', log_file='logs/mi_modulo.log')
```

### Migrar Tracking

**Antes:**
```python
results = model(frame)
# No hay tracking en modelo base
```

**Después:**
```python
detections, alerts = service.process_frame(frame)
# Tracking automático + alertas
for detection in detections:
    track_id = detection.track_id  # Ya disponible
```

### Migrar Procesamiento de Video

**Antes:**
```python
cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model(frame)
    # Procesar...
```

**Después:**
```python
from src.scripts.detect_video import main

# Usar script directamente
# python -m src.scripts.detect_video --source video.mp4
```

## ✅ Checklist de Migración

- [ ] Crear `config/app.yaml` con configuración
- [ ] Instalar dependencias (`pip install pyyaml`)
- [ ] Reemplazar inicialización de modelo
- [ ] Reemplazar código de detección
- [ ] Reemplazar código de alertas
- [ ] Reemplazar código de guardado
- [ ] Reemplazar código de logging
- [ ] Probar scripts individuales
- [ ] Verificar resultados en `output/`
- [ ] Eliminar código antiguo
- [ ] Actualizar documentación

## 🆘 Problemas Comunes

### Error: `ModuleNotFoundError: No module named 'yaml'`

```powershell
pip install pyyaml
```

### Archivo de configuración no encontrado

Asegúrate de que `config/app.yaml` existe en la raíz del proyecto.

### Detecciones vacías

- Verifica que la imagen sea válida
- Ajusta el umbral de confianza en `config/app.yaml`
- Prueba con una imagen de prueba conocida

### Alertas no se generan

- Verifica que `config/zones.json` esté presente
- Ajusta `dwell_threshold_sec` en `config/app.yaml`
- Verifica que los polígonos de zona sean válidos

## 📚 Recursos Adicionales

- [docs/ARCHITECTURE_SCALABLE.md](../docs/ARCHITECTURE_SCALABLE.md) - Arquitectura completa
- [examples_usage.py](../examples_usage.py) - Ejemplos prácticos
- [README.md](../README.md) - Documentación general
