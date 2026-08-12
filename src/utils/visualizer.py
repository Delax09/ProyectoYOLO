# src/utils/visualizer.py
import cv2
import numpy as np

class Visualizer:
    @staticmethod
    def draw_zone(frame, polygon_coords, is_alert=False):
        pts = np.array(polygon_coords, np.int32).reshape((-1, 1, 2))
        color = (0, 0, 255) if is_alert else (255, 0, 0)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    @staticmethod
    def draw_detections(frame, detections, alert_ids=None):
        if alert_ids is None:
            alert_ids = []

        if detections is not None and len(detections) > 0:
            coords = detections.xyxy.cpu().numpy()
            track_ids = detections.id.int().cpu().numpy() if detections.id is not None else [None] * len(coords)

            for box, track_id in zip(coords, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int(y2)
                
                color = (0, 0, 255) if track_id in alert_ids else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                
                label = f"ID: {track_id}" if track_id is not None else "Obj"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)