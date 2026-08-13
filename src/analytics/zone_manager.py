import cv2
import numpy as np

class ZoneManager:

    def __init__(self, zones, frame_shape):
        self.zones = zones
        self.frame_shape = frame_shape
        self.masks = {}

        self._build_masks()

    def _build_masks(self):
        height, width = self.frame_shape[:2]

        for zone in self.zones:
            mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            points = np.array(
                zone["puntos"],
                dtype=np.int32
            )

            cv2.fillPoly(
                mask,
                [points],
                255
            )

            self.masks[zone["id"]] = mask

    def is_point_in_zone(self, x, y, zone_id):
        mask = self.masks.get(zone_id)

        if mask is None:
            return False

        height, width = mask.shape

        if not (0 <= x < width and 0 <= y < height):
            return False

        return mask[y, x] == 255

    def get_zone(self, x, y):
        for zone in self.zones:

            if self.is_point_in_zone(
                x,
                y,
                zone["id"]
            ):
                return zone

        return None

    def get_zone_id(self, x, y):
        zone = self.get_zone(x, y)

        if zone is None:
            return None

        return zone["id"]

    def get_zones(self):
        return self.zones