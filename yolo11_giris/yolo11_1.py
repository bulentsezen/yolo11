from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model('ucus.mp4', save=True)
