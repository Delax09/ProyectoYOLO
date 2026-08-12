# src/io/report_writer.py
import json
import time
import cv2
from pathlib import Path

class ReportWriter:
    @staticmethod
    def save_image_report(output_path, annotated_frame, report_data):
        out_img_path = Path(output_path)
        out_img_path.parent.mkdir(parents=True, exist_ok=True)

        # Guardar la imagen
        cv2.imwrite(str(out_img_path), annotated_frame)

        # Guardar el JSON equivalente
        json_path = out_img_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)