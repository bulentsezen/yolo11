from ultralytics import YOLO

# Load a model
# model = YOLO("runs/detect/train/weights/best.pt")
model = YOLO("best.pt")

# Perform object detection on an image
results = model("Dataset/SplitData/test/images", save=True)
