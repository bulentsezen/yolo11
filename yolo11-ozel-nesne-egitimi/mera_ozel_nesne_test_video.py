from ultralytics import YOLO

# Load a model
# model = YOLO("runs/detect/train/weights/best.pt")
model = YOLO("best.pt")

# Perform object detection on a video
results = model('kucukbas.mp4', save=True)
