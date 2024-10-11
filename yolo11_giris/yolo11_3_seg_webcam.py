from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')

results = model(0, save=True)
