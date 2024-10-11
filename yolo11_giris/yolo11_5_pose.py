from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')

results = model('dans.jpg')
results[0].show()