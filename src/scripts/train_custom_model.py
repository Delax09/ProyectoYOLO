"""
Entrenamiento de modelo YOLO personalizado
"""

from ultralytics import YOLO
import argparse
from pathlib import Path
from src.utils.logging_utils import setup_logger

def train_model(data_yaml, model_name='yolov8n.pt', epochs=100, imgsz=640,
                batch_size=16, device=0, project='runs/detect', name='custom_model', logger=None):
    if logger is None:
        logger = setup_logger('train_custom_model', log_file='logs/train_custom_model.log')

    logger.info(f"Inicializando entrenamiento...")
    logger.info(f"  Modelo: {model_name}")
    logger.info(f"  Dataset: {data_yaml}")
    logger.info(f"  Épocas: {epochs}")
    logger.info(f"  Tamaño de imagen: {imgsz}")

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        patience=20,
        save=True,
        verbose=True,
        plots=True,
    )

    logger.info("\nEntrenamiento completado!")
    logger.info(f"Resultados guardados en: {project}/{name}")

    logger.info("\nValidando modelo...")
    metrics = model.val()

    return model, results, metrics

def validate_model(model_path, data_yaml):
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento YOLO personalizado')
    parser.add_argument('--data', type=str, required=True,
                        help='Ruta a data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Modelo base (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Tamaño del batch (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Tamaño de imagen (default: 640)')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU ID o cpu (default: 0)')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Carpeta de salida (default: runs/detect)')
    parser.add_argument('--name', type=str, default='custom_model',
                        help='Nombre del experimento (default: custom_model)')

    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Error: {args.data} no encontrado")
        return

    logger = setup_logger('train_custom_model', log_file='logs/train_custom_model.log')
    logger.info('Iniciando entrenamiento YOLO personalizado')
    logger.info(f"Modelo: {args.model}, data: {args.data}, epochs: {args.epochs}, batch: {args.batch}, imgsz: {args.imgsz}, device: {args.device}")

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        logger=logger
    )

if __name__ == '__main__':
    main()
