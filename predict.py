from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/segment/runs/shoe_segmentation-3/weights/best.pt")

# Predict on all new images
model.predict(
    source="new_images",
    conf=0.5,
    imgsz=640,
    save=True,
    show=False
)

print("Prediction completed!")