import cv2
import json
import numpy as np


class ShoeMeasurement:

    def __init__(self):

        with open("scale.json", "r") as f:
            scale = json.load(f)

        self.mm_per_pixel = scale["mm_per_pixel"]

    def measure(self, contour):

        # --------------------------
        # Convert contour
        # --------------------------

        pts = contour.reshape(-1, 2).astype(np.float32)

        # --------------------------
        # PCA
        # --------------------------

        mean, eigenvectors = cv2.PCACompute(pts, mean=None)

        center = mean[0]

        direction = eigenvectors[0]

        # Normalize

        direction = direction / np.linalg.norm(direction)

        # --------------------------
        # Project contour points
        # --------------------------

        projections = np.dot(
            pts - center,
            direction
        )

        min_proj = np.min(projections)
        max_proj = np.max(projections)

        heel = center + direction * min_proj
        toe = center + direction * max_proj

        heel = heel.astype(int)
        toe = toe.astype(int)

        pixel_length = max_proj - min_proj

        mm_length = pixel_length * self.mm_per_pixel

        return heel, toe, pixel_length, mm_length

    def draw(self, image, heel, toe, length):

        output = image.copy()

        cv2.line(
            output,
            tuple(heel),
            tuple(toe),
            (0,255,255),
            3
        )

        cv2.circle(
            output,
            tuple(heel),
            8,
            (255,0,0),
            -1
        )

        cv2.circle(
            output,
            tuple(toe),
            8,
            (0,0,255),
            -1
        )

        center = (
            (heel + toe)//2
        )

        cv2.putText(
            output,
            f"{length:.2f} mm",
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

        return output