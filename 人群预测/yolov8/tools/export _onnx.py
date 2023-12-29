from ultralytics import YOLO

# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/home/qcj/workcode/nnvlc/data/yolov8n.pt')  # load a custom trained model

# Export the model
model.export(format='onnx')