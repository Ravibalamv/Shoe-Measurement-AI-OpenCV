import cv2
import numpy as np
import json


class TongueDetector:

    def __init__(self):

        # HSV range for blue tongue
        self.lower_blue = np.array([90, 60, 40])
        self.upper_blue = np.array([140, 255, 255])

        with open("scale.json", "r") as f:
            self.mm_per_pixel = json.load(f)["mm_per_pixel"]

    def detect(self, shoe_roi):

        hsv = cv2.cvtColor(shoe_roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            self.lower_blue,
            self.upper_blue
        )

        kernel = np.ones((5,5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        h, w = mask.shape

        # -------------------------------------------------
        # Ignore top 25% (toe area)
        # Ignore bottom 20% (lower tongue area)
        # -------------------------------------------------

        start_search = int(h * 0.25)
        end_search = int(h * 0.80)

        tongue_point = None

        for y in range(start_search, end_search):

            xs = np.where(mask[y] > 0)[0]

            # Ignore tiny noisy rows
            if len(xs) < 30:
                continue

            left = xs.min()
            right = xs.max()

            center_x = int((left + right) / 2)

            tongue_point = (center_x, y)

            break

        if tongue_point is None:
            return None, None

        return tongue_point, mask

    def draw(self, image, tongue_point):

        output = image.copy()

        cv2.circle(
            output,
            tongue_point,
            8,
            (255,0,255),
            -1
        )

        cv2.putText(
            output,
            "Tongue",
            (tongue_point[0] + 10, tongue_point[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,255),
            2
        )

        return output