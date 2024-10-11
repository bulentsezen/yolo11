from ultralytics import YOLO

model = YOLO('yolo11n-cls.pt')

results = model('ucak.jpg')
results[0].show()