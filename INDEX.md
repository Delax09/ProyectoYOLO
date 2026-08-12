# 📑 Índice de Refactorización - ProyectoYOLO

Guía rápida para encontrar archivos y documentación relacionados con la refactorización.

## 🚀 Empezar Aquí

1. **Primeras 5 minutos**: Ejecuta `python quick_start.py`
2. **Primeros 30 minutos**: Lee `REFACTORING_SUMMARY.md`
3. **Primeras 2 horas**: Revisa `docs/ARCHITECTURE_SCALABLE.md`

## 📂 Estructura de Archivos Nuevos

### Núcleo de la Arquitectura (`src/core/`)

| Archivo | Propósito |
|---------|-----------|
| `config_manager.py` | Gestión centralizada de configuración YAML |
| `base_detector.py` | Clase abstracta para detectores + dataclass Detection |

📖 **Lee**: `docs/ARCHITECTURE_SCALABLE.md` → Sección "Abstracciones Polimórficas"

### Implementación de Detectores (`src/detector/`)

| Archivo | Propósito |
|---------|-----------|
| `yolo_detector_impl.py` | Implementación de YOLO v8 basada en BaseDetector |

📖 **Lee**: `docs/ARCHITECTURE_SCALABLE.md` → Sección "Caso 1: Múltiples Modelos"

### Sistema de Alertas (`src/alerts/`)

| Archivo | Propósito |
|---------|-----------|
| `__init__.py` | AlertManager, Alert dataclass, lógica de alertas |

📖 **Lee**: `examples_usage.py` → `example_video_with_alerts()`

### Almacenamiento Flexible (`src/storage/`)

| Archivo | Propósito |
|---------|-----------|
| `__init__.py` | BaseStorage, JSONStorage, CSVStorage |

📖 **Lee**: `docs/ARCHITECTURE_SCALABLE.md` → Sección "Caso 2: Base de Datos"

### Servicios de Orquestación (`src/services/`)

| Archivo | Propósito |
|---------|-----------|
| `__init__.py` | DetectionService - orquestador principal |

📖 **Lee**: `examples_usage.py` → `example_simple_image_detection()`

### Scripts Ejecutables (`src/scripts/`)

| Script | Propósito | Nuevo? |
|--------|-----------|--------|
| `detect_image.py` | Detección en imágenes | ✏️ Actualizado |
| `detect_video.py` | Detección en videos | ✨ NUEVO |
| `camera_live.py` | Cámara en vivo con alertas | ✨ NUEVO |
| `train_custom_model.py` | Entrenar modelos personalizados | - |

📖 **Lee**: `README.md` → Sección "Comandos de Uso"

### Configuración (`config/`)

| Archivo | Propósito |
|---------|-----------|
| `app.yaml` | Configuración centralizada de toda la aplicación |
| `zones_example.json` | Ejemplo de zonas restringidas para alertas |

📖 **Lee**: `README.md` → Sección "Configuración"

### Tests (`tests/`)

| Archivo | Propósito |
|---------|-----------|
| `test_architecture.py` | Suite de pruebas unitarias de la arquitectura |

📖 **Lee**: `docs/VALIDATION_CHECKLIST.md` → "Fase 9: Verificación"

## 📚 Documentación

### Documentos Principales

| Documento | Contenido | Lectura |
|-----------|----------|---------|
| `REFACTORING_SUMMARY.md` | Resumen ejecutivo de cambios | ⏱️ 10 min |
| `README.md` | Documentación general actualizada | ⏱️ 15 min |
| `quick_start.py` | Guía interactiva de inicio | ⏱️ 5 min |
| `examples_usage.py` | 7 ejemplos prácticos ejecutables | ⏱️ 20 min |

### Documentación Técnica

| Documento | Contenido | Lectura |
|-----------|----------|---------|
| `docs/ARCHITECTURE_SCALABLE.md` | Guía completa de arquitectura | ⏱️ 45 min |
| `docs/MIGRATION_GUIDE.md` | Cómo migrar código antiguo | ⏱️ 30 min |
| `docs/VALIDATION_CHECKLIST.md` | Plan de testing y validación | ⏱️ 20 min |

### Este Documento

| Documento | Contenido | Lectura |
|-----------|----------|---------|
| `INDEX.md` (este archivo) | Guía para encontrar archivos | ⏱️ 5 min |

## 🎯 Guías por Caso de Uso

### "Quiero entender la arquitectura"
1. Lee: `REFACTORING_SUMMARY.md`
2. Lee: `docs/ARCHITECTURE_SCALABLE.md`
3. Explora: `src/core/` y `src/services/`

### "Quiero detectar objetos en imágenes"
1. Ejecuta: `quick_start.py`
2. Corre: `python -m src.scripts.detect_image --image examples/ImagenPersonas.jpg`
3. Lee: `examples_usage.py` → `example_simple_image_detection()`

### "Quiero procesar video con alertas"
1. Lee: `README.md` → "Detección en Video"
2. Corre: `python -m src.scripts.detect_video --source examples/VideoPersonas.mp4 --zones config/zones.json`
3. Revisa: `output/` para resultados

### "Quiero usar cámara en vivo"
1. Corre: `python -m src.scripts.camera_live --camera 0`
2. Lee: `examples_usage.py` → `example_live_camera_with_annotations()`

### "Quiero agregar mi propio detector"
1. Lee: `docs/ARCHITECTURE_SCALABLE.md` → "Extensibilidad Futura"
2. Revisa: `src/detector/yolo_detector_impl.py` como ejemplo
3. Crea: Clase que extienda `BaseDetector`

### "Quiero cambiar a base de datos"
1. Lee: `docs/ARCHITECTURE_SCALABLE.md` → "Caso 2: Base de Datos"
2. Lee: `src/storage/__init__.py` para ver patrón
3. Implementa: Clase que extienda `BaseStorage`

### "Quiero migrar código antiguo"
1. Lee: `docs/MIGRATION_GUIDE.md` completo
2. Sigue: Ejemplos antes/después en el documento
3. Usa: Checklist de migración al final

### "Quiero testear la arquitectura"
1. Ejecuta: `pytest tests/test_architecture.py -v`
2. Lee: `docs/VALIDATION_CHECKLIST.md`
3. Sigue: Plan de testing paso a paso

## 🔍 Búsqueda Rápida

### Por Componente
- **Configuración**: `src/core/config_manager.py`
- **Detección**: `src/core/base_detector.py` + `src/detector/yolo_detector_impl.py`
- **Alertas**: `src/alerts/`
- **Almacenamiento**: `src/storage/`
- **Orquestación**: `src/services/`

### Por Concepto
- **Abstracciones**: `src/core/base_detector.py` + `src/storage/__init__.py`
- **Configuración YAML**: `config/app.yaml`
- **Logging**: `src/utils/logging_utils.py`
- **Detección**: `src/detector/yolo_detector_impl.py`
- **Alertas**: `src/alerts/__init__.py`

### Por Arquitectura
- **Inyección de Dependencias**: Ver `DetectionService.__init__`
- **Factory Pattern**: `create_storage()` en `src/storage/__init__.py`
- **Strategy Pattern**: Detectores en `src/detector/`
- **Observer Pattern**: Alertas en `src/alerts/`

## 📊 Tamaño de Cambios

### Archivos Creados Nuevos
- ✨ 11 archivos core/services
- ✨ 2 scripts nuevos (detect_video.py, camera_live.py)
- ✨ 3 documentos técnicos
- ✨ 1 script de quick start
- ✨ 1 suite de tests

### Archivos Actualizados
- ✏️ `detect_image.py` (refactorizado)
- ✏️ `requirements.txt` (agregado pyyaml)
- ✏️ `README.md` (documentación)

### Líneas de Código
- ➕ ~3000 líneas de código nuevo
- ➕ ~2000 líneas de documentación
- ✏️ ~100 líneas modificadas en código antiguo

## ⚡ Comandos Útiles

```powershell
# Instalar dependencias
pip install -r requirements.txt
pip install pyyaml

# Ejecutar detección en imagen
python -m src.scripts.detect_image --image examples/ImagenPersonas.jpg

# Ejecutar detección en video
python -m src.scripts.detect_video --source examples/VideoPersonas.mp4

# Ejecutar cámara en vivo
python -m src.scripts.camera_live --camera 0

# Ejecutar tests
pytest tests/test_architecture.py -v

# Ejecutar quick start
python quick_start.py

# Ver estructura
tree src/ /F
```

## 🎓 Línea de Aprendizaje Recomendada

```
1. quick_start.py (5 min)
   ↓
2. REFACTORING_SUMMARY.md (10 min)
   ↓
3. README.md - Sección de uso (15 min)
   ↓
4. examples_usage.py - Ejecutar ejemplos (20 min)
   ↓
5. docs/ARCHITECTURE_SCALABLE.md (45 min)
   ↓
6. Explorar código en src/ (30 min)
   ↓
7. docs/MIGRATION_GUIDE.md si aplica (30 min)
   ↓
8. Ejecutar tests (10 min)
```

**Tiempo total**: ~2-3 horas para dominar la arquitectura

## 🚀 Siguientes Pasos

Después de familiarizarte con la nueva arquitectura:

1. **Validar**: Ejecuta `docs/VALIDATION_CHECKLIST.md`
2. **Personalizar**: Edita `config/app.yaml`
3. **Extender**: Agrega tus propios detectores/storage
4. **Escalar**: Implementa API REST, Database, etc.

## 📞 Referencia Rápida

| Necesito... | Archivo | Línea |
|-------------|---------|-------|
| Cambiar parámetros de detección | `config/app.yaml` | Línea 1-6 |
| Entender DetectionService | `src/services/__init__.py` | Línea 1-50 |
| Ver ejemplo de uso | `examples_usage.py` | Inicio |
| Validar instalación | `docs/VALIDATION_CHECKLIST.md` | Fase 1-2 |
| Migrar código antiguo | `docs/MIGRATION_GUIDE.md` | Sección "Pasos" |

---

✅ **Todo está documentado y listo para usar**  
🚀 **La arquitectura es escalable y extensible**  
📚 **Hay documentación para cada componente**
