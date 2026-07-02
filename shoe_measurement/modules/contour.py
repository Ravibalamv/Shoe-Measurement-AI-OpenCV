import cv2


class ContourFinder:

    def find_largest(self, mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None

        largest = max(
            contours,
            key=cv2.contourArea
        )

        return largest

    def draw(self, image, contour):

        output = image.copy()

        cv2.drawContours(
            output,
            [contour],
            -1,
            (0,255,0),
            3
        )

        return output