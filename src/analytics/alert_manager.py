# src/analytics/alert_manager.py
import cv2
import numpy as np
import time

class AlertManager:
    def __init__(self, polygon_coords, frame_shape, dwell_threshold_sec=3.0):
        self.dwell_threshold = dwell_threshold_sec
        self.track_timers = {}  # {track_id: timestamp_ingreso}
        self.polygon_coords = polygon_coords
        self.counted_ids = set() #Registra ID unico 
        
        # Generar máscara binaria una sola vez durante la inicialización
        self.mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        pts = np.array(polygon_coords, dtype=np.int32)
        cv2.fillPoly(self.mask, [pts], 255)

    def is_point_in_zone(self, x: int, y: int) -> bool:
        h, w = self.mask.shape
        if 0 <= x < w and 0 <= y < h:
            return self.mask[int(y), int(x)] == 255
        return False

    def evaluate_detections(self, detections, current_time: float):
        active_alerts = []
        current_ids_in_zone = set()

        if detections is None or detections.id is None:
            return active_alerts

        boxes = detections.xyxy.cpu().numpy()
        track_ids = detections.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            # Centro inferior de la Bounding Box (contacto con el suelo)
            foot_x, foot_y = int((x1 + x2) / 2), int(y2)

            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)  

            if self.is_point_in_zone(foot_x, foot_y):
                current_ids_in_zone.add(track_id)
                if track_id not in self.track_timers:
                    self.track_timers[track_id] = current_time
                
                time_in_zone = current_time - self.track_timers[track_id]
                if time_in_zone >= self.dwell_threshold:
                    active_alerts.append({
                        "track_id": int(track_id),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "dwell_time": round(time_in_zone, 2)
                    })

        # Limpieza de objetos que salieron del área
        for tid in list(self.track_timers.keys()):
            if tid not in current_ids_in_zone:
                del self.track_timers[tid]

        return active_alerts

    #Muestra el total de personas que cruzaron 
    def get_total_count(self) -> int:
        return len(self.counted_ids)