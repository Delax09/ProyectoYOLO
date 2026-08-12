# Arquitectura Escalable - Guía de Implementación

## 📐 Estructura Refactorizada

```
src/
├── core/                      # Núcleo (abstracciones)
│   ├── __init__.py
│   ├── config_manager.py      # Gestión centralizada de configuración
│   └── base_detector.py       # Clase abstracta para detectores
│
├── detector/                  # Implementaciones de detectores
│   ├── __init__.py
│   ├── yolo_detector_impl.py  # Implementación YOLO v8
│   └── (otros detectores)     # Futura: TensorFlow, ONNX, etc.
│
├── alerts/                    # Sistema de alertas modular
│   ├── __init__.py
│   └── (futuro: nuevos tipos)
│
├── storage/                   # Persistencia de datos
│   ├── __init__.py
│   └── (futuro: DB, API, Cloud)
│
├── services/                  # Orquestación de lógica
│   └── __init__.py            # DetectionService principal
│
├── scripts/                   # Scripts ejecutables
│   ├── detect_image.py        # Detección en imágenes
│   ├── detect_video.py        # Detección en videos
│   ├── camera_live.py         # Cámara en vivo
│   └── train_custom_model.py  # Entrenamiento
│
└── utils/                     # Utilidades
    └── logging_utils.py       # Sistema de logging
```

## 🔑 Principios de Diseño

### 1. **Separación de Responsabilidades**
- **Detectores**: Solo infieren objetos
- **Alertas**: Solo evalúan condiciones en detecciones
- **Storage**: Solo persiste datos
- **Services**: Orquestan todo

### 2. **Abstracciones Polimórficas**
```python
# Agregar nuevo detector es simple
class CustomDetector(BaseDetector):
    def _load_model(self):
        # Cargar tu modelo
        pass
    
    def detect(self, image):
        # Tu lógica de inferencia
        pass
```

### 3. **Configuración Centralizada**
```yaml
# config/app.yaml
detector:
  model_name: 'yolov8n.pt'
  confidence: 0.5
  device: 'cuda'

alerts:
  dwell_threshold_sec: 3.0
  alert_cooldown_sec: 5.0

storage:
  type: 'json'
```

## 🚀 Casos de Escalabilidad

### Caso 1: Múltiples Modelos
```python
detector = DetectorFactory.create(
    type='yolo',
    model_name='yolov8l.pt'
)

detector = DetectorFactory.create(
    type='tensorflow',
    model_path='models/custom.pb'
)
```

### Caso 2: Base de Datos
```python
storage = create_storage(
    storage_type='postgresql',
    connection_string='postgresql://...'
)
```

### Caso 3: Múltiples Cámaras
```python
detections = []
for camera_id in range(4):
    frame = capture(camera_id)
    det, alerts = service.process_frame(frame)
    detections.append(det)
```

### Caso 4: API REST
```python
from fastapi import FastAPI
from src.services import DetectionService

app = FastAPI()

@app.post("/detect")
async def detect_endpoint(file: UploadFile):
    service = get_detection_service()
    detections = service.detect_image(file.path)
    return {"detections": detections}
```

## 📝 Migrando Código Antiguo

### Antes (Monolítico)
```python
model = YOLO('yolov8n.pt')
results = model(img)
# Lógica de alertas mezclada
# Lógica de almacenamiento mezclada
```

### Después (Modular)
```python
config = get_config_manager()
service = DetectionService(config)

detections = service.detect_image('image.jpg')
service.save_results(detections, [], 'image')
```

## 🔌 Extensibilidad Futura

### Agregar Detector Personalizado
1. Crear clase que extienda `BaseDetector`
2. Implementar `_load_model()` y `detect()`
3. Registrar en factory

### Agregar Storage Backend
1. Crear clase que extienda `BaseStorage`
2. Implementar `save_detections()` y `save_alerts()`
3. Registrar en factory

### Agregar Tipo de Alerta
1. Extender `AlertManager`
2. Agregar nueva lógica de evaluación
3. Configurar en `config/app.yaml`

## 📊 Ventajas de la Refactorización

| Aspecto | Antes | Después |
|--------|-------|---------|
| **Testabilidad** | Difícil (acoplado) | Fácil (inyección) |
| **Reutilización** | Limitada | Máxima |
| **Escalabilidad** | Monolítica | Modular |
| **Mantenimiento** | Complejo | Simple |
| **Configuración** | Hardcodeada | Centralizada |
| **Múltiples modelos** | No soportado | Soportado |

## 🛠️ Siguientes Pasos

1. **Base de Datos**: Implementar SQLAlchemy storage
2. **API REST**: Crear FastAPI wrapper
3. **Docker**: Containerizar para deploy
4. **Caché**: Redis para resultados frecuentes
5. **Monitoring**: Prometheus/Grafana para métricas
6. **Queue**: Celery para procesamiento async
7. **Tests**: Suite de pruebas unitarias
8. **CI/CD**: GitHub Actions para automatización
