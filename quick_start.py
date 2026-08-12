#!/usr/bin/env python
"""
Quick Start Guide - ProyectoYOLO Refactored Architecture

Este script te guía por los primeros pasos para usar la nueva arquitectura.
Ejecutar: python quick_start.py
"""

import os
import sys
from pathlib import Path


def print_section(title):
    """Print a formatted section title."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def check_environment():
    """Check if environment is properly set up."""
    print_section("🔍 Verificando Ambiente")
    
    checks = {
        "Python 3.8+": sys.version_info >= (3, 8),
        "config/app.yaml": Path("config/app.yaml").exists(),
        "src/ exists": Path("src").is_dir(),
        "examples/ exists": Path("examples").is_dir(),
    }
    
    all_ok = True
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}")
        if not result:
            all_ok = False
    
    return all_ok


def install_dependencies():
    """Guide user to install dependencies."""
    print_section("📦 Instalando Dependencias")
    
    print("Ejecuta el siguiente comando:")
    print("\n  pip install -r requirements.txt")
    print("  pip install pyyaml\n")
    
    input("Presiona Enter cuando hayas instalado las dependencias...")


def first_detection():
    """Run first detection example."""
    print_section("🚀 Primera Detección")
    
    print("Ejecutando detección en imagen de ejemplo...\n")
    
    cmd = (
        "python -m src.scripts.detect_image "
        "--image examples/ImagenPersonas.jpg "
        "--output output/quick_start_result "
        "--show"
    )
    
    print(f"Comando:\n  {cmd}\n")
    print("Si la imagen se abre, ¡la instalación fue exitosa!\n")
    
    input("Presiona Enter para ejecutar...")
    
    os.system(cmd)


def explore_configuration():
    """Guide user through configuration."""
    print_section("⚙️ Explorando Configuración")
    
    print("El archivo de configuración principal es: config/app.yaml\n")
    print("Parámetros principales:\n")
    
    params = {
        "detector.model_name": "Modelo YOLO a usar (yolov8n.pt, yolov8s.pt, etc.)",
        "detector.confidence": "Umbral de confianza (0.0-1.0)",
        "detector.device": "Dispositivo (cpu, cuda, mps)",
        "alerts.dwell_threshold_sec": "Segundos de permanencia para alerta",
        "storage.type": "Formato de salida (json, csv)",
    }
    
    for param, desc in params.items():
        print(f"  • {param}")
        print(f"    → {desc}\n")
    
    print("Para editar: Abre config/app.yaml con tu editor favorito\n")
    input("Presiona Enter para continuar...")


def explore_scripts():
    """Guide user through available scripts."""
    print_section("📜 Scripts Disponibles")
    
    scripts = {
        "detect_image.py": "Detectar objetos en una imagen",
        "detect_video.py": "Detectar objetos en un video",
        "camera_live.py": "Cámara en vivo con detección",
        "train_custom_model.py": "Entrenar modelo personalizado",
    }
    
    print("Scripts disponibles en src/scripts/:\n")
    
    for script, desc in scripts.items():
        print(f"  • {script}")
        print(f"    {desc}\n")


def show_examples():
    """Show example usage patterns."""
    print_section("💡 Ejemplos de Uso")
    
    examples = [
        ("Detección en imagen", 
         "python -m src.scripts.detect_image --image examples/ImagenPersonas.jpg"),
        
        ("Detección en video",
         "python -m src.scripts.detect_video --source examples/VideoPersonas.mp4 --zones config/zones.json"),
        
        ("Cámara en vivo",
         "python -m src.scripts.camera_live --camera 0"),
        
        ("Con configuración personalizada",
         "python -m src.scripts.detect_image --image examples/ImagenPersonas.jpg --config config/app.yaml"),
    ]
    
    for title, cmd in examples:
        print(f"📌 {title}")
        print(f"   {cmd}\n")


def show_next_steps():
    """Show recommended next steps."""
    print_section("🎯 Próximos Pasos")
    
    steps = [
        ("1. Explorar ejemplos", "Abre examples_usage.py y ejecuta los ejemplos"),
        ("2. Validar instalación", "Ejecuta: pytest tests/test_architecture.py -v"),
        ("3. Leer documentación", "Ver docs/ARCHITECTURE_SCALABLE.md"),
        ("4. Personalizar", "Edita config/app.yaml según tus necesidades"),
        ("5. Agregar modelos", "Copia tus modelos YOLO a models/"),
    ]
    
    for step, desc in steps:
        print(f"  {step}")
        print(f"  → {desc}\n")


def show_documentation():
    """Show documentation reference."""
    print_section("📚 Documentación Disponible")
    
    docs = {
        "README.md": "Documentación general del proyecto",
        "REFACTORING_SUMMARY.md": "Resumen de cambios realizados",
        "docs/ARCHITECTURE_SCALABLE.md": "Guía completa de arquitectura",
        "docs/MIGRATION_GUIDE.md": "Migración de código antiguo",
        "docs/VALIDATION_CHECKLIST.md": "Plan de validación",
        "examples_usage.py": "7 ejemplos prácticos",
    }
    
    for doc, desc in docs.items():
        print(f"  📄 {doc}")
        print(f"     {desc}\n")


def main():
    """Main quick start guide."""
    print("\n" + "🎯 " * 20)
    print("    GUÍA DE INICIO RÁPIDO - ProyectoYOLO (Arquitectura Escalable)")
    print("🎯 " * 20)
    
    # Check environment
    if not check_environment():
        print("\n⚠️  Problemas detectados. Por favor revisa la configuración.\n")
        return
    
    # Menu
    while True:
        print_section("📋 MENÚ PRINCIPAL")
        
        menu = {
            "1": ("Instalar dependencias", install_dependencies),
            "2": ("Ejecutar primera detección", first_detection),
            "3": ("Explorar configuración", explore_configuration),
            "4": ("Ver scripts disponibles", explore_scripts),
            "5": ("Ver ejemplos de uso", show_examples),
            "6": ("Ver documentación", show_documentation),
            "7": ("Próximos pasos recomendados", show_next_steps),
            "0": ("Salir", None),
        }
        
        for key, (label, _) in menu.items():
            print(f"  {key}. {label}")
        
        choice = input("\nSelecciona una opción: ").strip()
        
        if choice == "0":
            print("\n✨ ¡Gracias por usar ProyectoYOLO!")
            print("📖 Para más información, revisa docs/\n")
            break
        elif choice in menu:
            label, func = menu[choice]
            if func:
                func()
        else:
            print("\n❌ Opción no válida. Intenta de nuevo.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Hasta luego!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
