from ultralytics import YOLO

model = YOLO("/home/qcj/workcode/runs/detect/train5/weights/best.pt")
# It'll use the data yaml file in model.pt if you don't set data.
model.val()
# or you can set the data you want to val
model.val(data='custom.yaml')