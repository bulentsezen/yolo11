from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')

results = model('araba.jpg')
results[0].show()