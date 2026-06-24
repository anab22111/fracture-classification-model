from ultralytics import YOLO

# load trained model
model = YOLO(r'runs\classify\train-6\weights\last.pt')

# export model to onnx format
model.export(format='onnx')