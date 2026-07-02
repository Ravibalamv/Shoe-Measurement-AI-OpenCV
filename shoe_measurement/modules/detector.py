from ultralytics import YOLO


class ShoeDetector:

    def __init__(self, model_path="best.pt"):

        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.5):

        results = self.model.predict(
            frame,
            conf=conf,
            verbose=False
        )

        return results