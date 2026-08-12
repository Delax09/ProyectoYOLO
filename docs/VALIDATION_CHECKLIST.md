# Validación de Refactorización - Checklist de Testing

Este documento proporciona un plan paso a paso para validar que la refactorización fue exitosa.

## ✅ Checklist de Validación

### Fase 1: Instalación y Setup

- [ ] **Instalar dependencias nuevas**
  ```powershell
  pip install pyyaml
  ```
  
- [ ] **Verificar archivos de configuración**
  - [ ] `config/app.yaml` existe
  - [ ] `config/zones.json` existe
  - [ ] Archivos son válidos YAML/JSON

- [ ] **Verificar estructura de carpetas**
  ```powershell
  Get-ChildItem -Recurse src/ | Measure-Object
  ```

### Fase 2: Verificar Módulos Base

- [ ] **Configuración**
  ```powershell
  python -c "from src.core.config_manager import get_config_manager; print('✓ Config OK')"
  ```

- [ ] **Detectores**
  ```powershell
  python -c "from src.core.base_detector import BaseDetector, Detection; print('✓ Base detector OK')"
  ```

- [ ] **Alertas**
  ```powershell
  python -c "from src.alerts import AlertManager; print('✓ Alert manager OK')"
  ```

- [ ] **Storage**
  ```powershell
  python -c "from src.storage import create_storage; print('✓ Storage OK')"
  ```

- [ ] **Servicios**
  ```powershell
  python -c "from src.services import DetectionService; print('✓ Services OK')"
  ```

### Fase 3: Pruebas de Funcionalidad Básica

#### Test 1: Detección Simple

```powershell
python -m src.scripts.detect_image `
  --image examples/ImagenPersonas.jpg `
  --config config/app.yaml `
  --output output/test1 `
  --show
```

**Validar:**
- ✓ Imagen se procesa sin errores
- ✓ Se generan archivos en `output/`
- ✓ Se crea archivo JSON con detecciones
- ✓ Log se crea en `logs/detect_image.log`

#### Test 2: Procesamiento de Video

```powershell
python -m src.scripts.detect_video `
  --source examples/VideoPersonas.mp4 `
  --config config/app.yaml `
  --zones config/zones.json `
  --output output/test_video.mp4 `
  --max-frames 100
```

**Validar:**
- ✓ Video se procesa sin errores
- ✓ Se genera video de salida
- ✓ Sistema de alertas se inicializa
- ✓ Se generan resultados JSON

#### Test 3: Cámara en Vivo

```powershell
python -m src.scripts.camera_live `
  --camera 0 `
  --config config/app.yaml `
  --zones config/zones.json
```

**Validar:**
- ✓ Cámara se abre correctamente
- ✓ Detecciones se muestran en vivo
- ✓ Presionar 'q' cierra la aplicación
- ✓ No hay excepciones

### Fase 4: Pruebas de Configuración

#### Test 4: Cambiar Parámetros

1. Modificar `config/app.yaml`:
   ```yaml
   detector:
     model_name: 'yolov8m.pt'  # Cambiar de n a m
     confidence: 0.6            # Aumentar confianza
   ```

2. Ejecutar:
   ```powershell
   python -m src.scripts.detect_image --image examples/ImagenPersonas.jpg
   ```

**Validar:**
- ✓ Los cambios se aplican
- ✓ El modelo se recarga
- ✓ Resultados cambian según configuración

#### Test 5: Diferentes Tipos de Storage

1. Cambiar en `config/app.yaml`:
   ```yaml
   storage:
     type: 'csv'
   ```

2. Ejecutar detección:
   ```powershell
   python -m src.scripts.detect_image --image examples/ImagenPersonas.jpg
   ```

**Validar:**
- ✓ Se generan archivos CSV
- ✓ CSV contiene datos correctos

### Fase 6: Pruebas de Alertas

#### Test 6: Sistema de Alertas con Zonas

1. Verificar `config/zones.json` tiene coordenadas válidas
2. Ejecutar:
   ```powershell
   python -m src.scripts.detect_video `
     --source examples/VideoPersonas.mp4 `
     --zones config/zones.json `
     --max-frames 300
   ```

**Validar:**
- ✓ Se generan alertas si hay detecciones en zona
- ✓ Archivo `*_alerts.json` contiene alertas
- ✓ `dwell_time` se calcula correctamente

### Fase 7: Pruebas de Logging

#### Test 7: Logs Generados

```powershell
Get-ChildItem logs/ -Recurse
```

**Validar:**
- ✓ Existen archivos en `logs/`
- ✓ Logs contienen información relevante
- ✓ No hay errores o warnings no esperados

Ver contenido:
```powershell
Get-Content logs/detection_service.log -Tail 20
```

### Fase 8: Pruebas de Casos de Uso

#### Test 8: Usar Ejemplos de Código

```powershell
python examples_usage.py
```

Esto ejecutará varios ejemplos. **Validar:**
- ✓ Ejemplos de configuración funcionan
- ✓ Detección simple funciona
- ✓ No hay excepciones

#### Test 9: Pruebas Unitarias

```powershell
pip install pytest pytest-cov
pytest tests/test_architecture.py -v
```

**Validar:**
- ✓ Todas las pruebas pasan
- ✓ Cobertura es > 70%

### Fase 9: Verificación de Retrocompatibilidad

#### Test 10: Scripts Antiguos (si existen)

Intentar ejecutar cualquier script antiguo:

```powershell
# Si existen scripts antiguos
python src/detector/yolo_detector.py --source examples
```

**Validar:**
- ✓ Script antiguo aún funciona (si se conservó)
- O ✓ Nuevo script reemplaza la funcionalidad

## 📊 Resumen de Cambios

### Nuevo en la Arquitectura

| Componente | Ubicación | Propósito |
|-----------|-----------|----------|
| ConfigManager | `src/core/config_manager.py` | Configuración centralizada |
| BaseDetector | `src/core/base_detector.py` | Abstracción para detectores |
| Detection | `src/core/base_detector.py` | Objeto de detección tipado |
| YOLODetectorImpl | `src/detector/yolo_detector_impl.py` | Implementación YOLO |
| AlertManager | `src/alerts/` | Sistema de alertas modular |
| BaseStorage | `src/storage/` | Abstracción de almacenamiento |
| DetectionService | `src/services/` | Orquestador principal |

### Scripts Actualizados

| Script | Cambios |
|--------|---------|
| `detect_image.py` | Ahora usa DetectionService |
| `detect_video.py` | Nuevo - Detección en video |
| `camera_live.py` | Nuevo - Cámara en vivo |

### Archivos de Configuración Nuevos

- `config/app.yaml` - Configuración centralizada
- `config/zones.json` - Ejemplo de zonas restringidas
- `docs/ARCHITECTURE_SCALABLE.md` - Documentación de arquitectura
- `docs/MIGRATION_GUIDE.md` - Guía de migración

## 🔍 Comando de Validación Rápida

Ejecutar esta secuencia para validar todo:

```powershell
# 1. Verificar importes
python -c "from src.core.config_manager import ConfigManager; from src.services import DetectionService; print('✓ Imports OK')"

# 2. Prueba simple
python -m src.scripts.detect_image `
  --image examples/ImagenPersonas.jpg `
  --output output/validation_test

# 3. Verificar resultados
Test-Path output/validation_test_detections.json
Get-ChildItem logs/
```

## ⚠️ Problemas Conocidos y Soluciones

### Problema: `ModuleNotFoundError: No module named 'yaml'`
**Solución:**
```powershell
pip install pyyaml
```

### Problema: `FileNotFoundError: config/app.yaml`
**Solución:**
```powershell
# Copiar archivo de ejemplo
Copy-Item config/app.yaml config/app.yaml.bak
# Recrear desde documentación en ARCHITECTURE_SCALABLE.md
```

### Problema: Detecciones vacías
**Solución:**
- Verificar que la imagen sea válida
- Reducir threshold de confianza en `config/app.yaml`
- Probar con imagen de ejemplo conocida

### Problema: Cámara no funciona
**Solución:**
- Verificar que cámara está disponible: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
- Cambiar ID de cámara (probar 0, 1, 2, etc.)

## 📈 Métricas de Éxito

Después de la refactorización:

- ✅ Código modular y reutilizable
- ✅ Configuración centralizada
- ✅ Sistemas desacoplados (detector, alertas, storage)
- ✅ Fácil de extender con nuevos componentes
- ✅ Logging estructurado en todos los componentes
- ✅ Resultados persistentes en `output/`
- ✅ Compatibilidad hacia adelante para escalado

## 🎯 Próximos Pasos (Futura Escalabilidad)

- [ ] API REST con FastAPI
- [ ] Base de datos con SQLAlchemy
- [ ] Procesamiento asíncrono con Celery
- [ ] Caché con Redis
- [ ] Monitoring con Prometheus/Grafana
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Pruebas de carga
