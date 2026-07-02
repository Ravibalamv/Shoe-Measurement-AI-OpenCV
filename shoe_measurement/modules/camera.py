import cv2


class IndustrialCamera:

    def __init__(self, camera_index=1, width=1920, height=1080):

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise Exception("Cannot open industrial camera.")

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):

        self.cap.release()