from ultralytics import YOLO     

model = YOLO('yolov8n-cls.pt')   # load a pretrained model

model.train(data=r'C:\Users\LENOVO\Documents\ANA\Treca_godina\ADOS\fracture_classification\dataset_CLAHE', epochs = 20, imgsz = 244)

