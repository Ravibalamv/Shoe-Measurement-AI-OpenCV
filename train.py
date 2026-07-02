from ultralytics import YOLO

def main():
    model = YOLO("yolov8s-seg.pt")

    model.train(
        data="data.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        cache=True,
        patience=30,
        project="runs",
        name="shoe_segmentation_v2"
    )

if __name__ == "__main__":
    main()