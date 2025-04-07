import cv2
from ultralytics import YOLO
import supervision as sv


model = YOLO('yolov12n.pt')
results = model.train(
    data='D:\yolov12-main\widerface\data.yaml',  # 数据集配置
    epochs=1,                  
    imgsz=640,                   
    batch=32,                    
    name='yolov12-face',
    task='pose'                # 明确指定任务为pose
)