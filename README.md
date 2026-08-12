# Proyecto de Visión por Computadora con YOLOv8 - Arquitectura Escalable
Consola: 
```
python -m src.scripts.detect_video 
  --source examples/VideoPersonas.mp4 
  --zones config/zonas.json 
  --output output/video_with_alerts.mp4


python -m src.scripts.detect_image ^
  --image examples/ImagenPersonas.jpg ^
  --config config/app.yaml ^
  --output output/imagen_result ^
  --show
```


Sistema modular y escalable de detección de objetos con YOLOv8, alertas en tiempo real y persistencia de datos. Diseñado para facilitar desarrollo, pruebas y despliegue en entornos de producción.

## ✨ Características Principales

- ✅ **Arquitectura Modular**: Separación clara de responsabilidades
- ✅ **Configuración Centralizada**: Archivo YAML único para toda la aplicación
- ✅ **Múltiples Detectores**: Fácil extensión para nuevos modelos
- ✅ **Sistema de Alertas**: Alertas de dwell time en zonas restringidas
- ✅ **Persistencia Flexible**: JSON, CSV (bases de datos en futuro)
- ✅ **Logging Estructurado**: Trazabilidad completa de operaciones
- ✅ **Escalable**: Diseñado para crecimiento y mantenibilidad

# Proyecto de Visión por Computadora con YOLOv8 - Arquitectura Escalable

Sistema modular y escalable de detección de objetos con YOLOv8, alertas en tiempo real y persistencia de datos. Diseñado para facilitar desarrollo, pruebas y despliegue en entornos de producción.

## ✨ Características Principales

- ✅ **Arquitectura Modular**: Separación clara de responsabilidades
- ✅ **Configuración Centralizada**: Archivo YAML único para toda la aplicación
- ✅ **Múltiples Detectores**: Fácil extensión para nuevos modelos
- ✅ **Sistema de Alertas**: Alertas de dwell time en zonas restringidas
- ✅ **Persistencia Flexible**: JSON, CSV (bases de datos en futuro)
- ✅ **Logging Estructurado**: Trazabilidad completa de operaciones
- ✅ **Escalable**: Diseñado para crecimiento y mantenibilidad

## 📁 Estructura del Proyecto (Refactorizado)

```
src/
├── core/                          # Abstracciones y configuración
│   ├── config_manager.py          # Gestión centralizada
│   └── base_detector.py           # Clase base para detectores
├── detector/
│   └── yolo_detector_impl.py      # Implementación YOLO
├── alerts/                        # Sistema de alertas
├── storage/                       # Persistencia de datos
├── services/                      # Orquestación de lógica
└── scripts/
    ├── detect_image.py            # Detección en imágenes
    ├── detect_video.py            # Detección en videos
    ├── camera_live.py             # Cámara en vivo
    └── train_custom_model.py      # Entrenamiento

config/
├── app.yaml                       # Configuración principal
└── zones.json                     # Zonas restringidas

logs/                              # Archivos de logging
output/                            # Resultados de detección
```

## 📋 Requisitos

- Python 3.8+ (se probó con Python 3.14)
- GPU + CUDA (recomendado pero opcional)

## 🚀 Instalación Rápida

### 1. Crear entorno virtual

```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Instalar dependencias

```powershell
pip install -r requirements.txt
pip install pyyaml  # Requerido para configuración
```

## 🧭 Comandos de Uso (Nueva Arquitectura)

### Detección en Imágenes

```powershell
python -m src.scripts.detect_image ^
  --image examples/ImagenPersonas.jpg ^
  --config config/app.yaml ^
  --output output/imagen_result ^
  --show
```

**Opciones:**
- `--image, -i`: Ruta a la imagen (requerido)
- `--config, -c`: Archivo de configuración (default: config/app.yaml)
- `--output, -o`: Ruta de salida sin extensión
- `--target-class, -t`: Clase a filtrar (default: person)
- `--show`: Mostrar imagen anotada

### Detección en Video

```powershell
python -m src.scripts.detect_video ^
  --source examples/VideoPersonas.mp4 ^
  --config config/app.yaml ^
  --zones config/zones.json ^
  --output output/video_result.mp4
```

**Opciones:**
- `--source, -s`: Archivo de video (requerido)
- `--config, -c`: Archivo de configuración
- `--zones`: Archivo JSON con zonas restringidas
- `--output, -o`: Archivo de video de salida
- `--max-frames`: Máximo número de frames a procesar

### Cámara en Vivo (Con Alertas)

```powershell
python -m src.scripts.camera_live ^
  --camera 0 ^
  --config config/app.yaml ^
  --zones config/zones.json ^
  --save-results
```

**Opciones:**
- `--camera, -c`: ID de la cámara (default: 0)
- `--config`: Archivo de configuración
- `--zones`: Archivo JSON con zonas
- `--save-results`: Guardar detecciones y alertas

### Entrenar Modelo Personalizado

```powershell
python -m src.scripts.train_custom_model ^
  --data config/data_example.yaml ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch-size 16
```

## ⚙️ Configuración

### config/app.yaml

```yaml
detector:
  model_name: 'yolov8n.pt'    # Modelo a usar
  confidence: 0.5             # Umbral de confianza
  device: 'cpu'               # 'cpu', 'cuda', 'mps'

alerts:
  enabled: true               # Activar alertas
  dwell_threshold_sec: 3.0    # Segundos para alerta
  alert_cooldown_sec: 5.0     # Cooldown entre alertas

storage:
  type: 'json'                # 'json', 'csv'
  output_dir: 'output'        # Directorio de salida
  save_detections: true
  save_alerts: true

logger:
  level: 'INFO'               # DEBUG, INFO, WARNING, ERROR
  log_dir: 'logs'
```

### config/zones.json

Define polígonos de zonas restringidas:

```json
{
  "zones": [
    {
      "id": "zona_entrada",
      "name": "Zona de Entrada",
      "enabled": true,
      "dwell_threshold_sec": 3.0,
      "coordinates": [
        [100, 100], [500, 100],
        [500, 400], [100, 400]
      ]
    }
  ]
}
```

## 📊 Resultados

Los resultados se guardan en `output/` en formato JSON/CSV:

### Detecciones (JSON)
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.95,
      "bbox": [100, 50, 200, 300],
      "track_id": 1
    }
  ]
}
```

### Alertas (JSON)
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "alerts": [
    {
      "alert_id": "ALT_000001",
      "track_id": 1,
      "alert_type": "dwell",
      "dwell_time": 5.2
    }
  ]
}
```

## 🔌 Extensibilidad

### Agregar un Nuevo Detector

```python
from src.core.base_detector import BaseDetector

class MiDetector(BaseDetector):
    def _load_model(self):
        # Cargar tu modelo
        pass
    
    def detect(self, image):
        # Tu lógica
        pass
    
    def detect_with_tracking(self, image):
        # Tu lógica de tracking
        pass
```

### Usar el Nuevo Detector

Actualizar `src/services/__init__.py` para usar tu detector:

```python
def _init_detector(self) -> BaseDetector:
    detector_config = self.config_manager.get_detector_config()
    return MiDetector(
        model_name=detector_config.model_name,
        confidence=detector_config.confidence
    )
```

## 📈 Casos de Escalado

- **Múltiples Cámaras**: Procesar varias fuentes simultáneamente
- **Base de Datos**: Cambiar storage de JSON a PostgreSQL
- **API REST**: Crear endpoint con FastAPI
- **Async Processing**: Usar Celery para procesamiento asíncrono
- **Monitoring**: Integrar Prometheus/Grafana
- **Docker**: Containerizar para deploy

Ver [docs/ARCHITECTURE_SCALABLE.md](docs/ARCHITECTURE_SCALABLE.md) para más detalles.

## 📝 Logging

Todos los scripts generan logs automáticamente en `logs/`:

- `app.log` - Logs generales
- `detection_service.log` - Servicios de detección
- `alerts.log` - Sistema de alertas
- `storage.log` - Persistencia de datos

Nivel de logging configurable en `config/app.yaml`.

## 🛠️ Desarrollo

### Instalar dependencias de desarrollo

```powershell
pip install pytest pytest-cov black flake8
```

### Ejecutar tests

```powershell
pytest tests/ -v --cov=src
```

### Formato de código

```powershell
black src/
flake8 src/
```

## 📚 Documentación Adicional

- [Arquitectura Escalable](docs/ARCHITECTURE_SCALABLE.md) - Guía completa de diseño
- [API Reference](docs/API.md) - Referencia de funciones (próximamente)
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Solución de problemas (próximamente)

## 📄 Logs Disponibles

Ejemplo de logs generados:
- `logs/app.log` - Aplicación general
- `logs/detection_service.log` - Servicios
- `logs/alerts.log` - Sistema de alertas

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
