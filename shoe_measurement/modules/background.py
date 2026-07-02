import cv2
import numpy as np


class BackgroundRemover:

    def __init__(self):

        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([90, 255, 255])

        self.kernel = np.ones((5,5), np.uint8)

    def remove(self, shoe_roi):

        hsv = cv2.cvtColor(
            shoe_roi,
            cv2.COLOR_BGR2HSV
        )

        green_mask = cv2.inRange(
            hsv,
            self.lower_green,
            self.upper_green
        )

        shoe_mask = cv2.bitwise_not(green_mask)

        shoe_mask = cv2.morphologyEx(
            shoe_mask,
            cv2.MORPH_OPEN,
            self.kernel
        )

        shoe_mask = cv2.morphologyEx(
            shoe_mask,
            cv2.MORPH_CLOSE,
            self.kernel
        )

        return shoe_mask