from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')

results = model('dans.mp4', save=True)
