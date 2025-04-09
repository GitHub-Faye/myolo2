from ultralytics import YOLO
import torch

# 加载最佳权重进行微调
model = YOLO('runs/pose/yolov12-face/weights/best.pt')
try:
    model.train(
        data='widerface_litel/data.yaml',    # 数据集配置
        epochs=10,                           # 增加训练轮数，1轮太少
        lr0=0.0002,                         # 适当调整学习率
        patience=20,                         # 早停耐心值
        batch=2,                            # 减小批次大小
        device='cpu',                       # 明确指定设备
        workers=0,                          # 设置workers为0
        optimizer='AdamW',                  # 明确指定优化器
        weight_decay=0.0005,                # 权重衰减
        warmup_epochs=3,                    # 预热轮数
        cos_lr=True,                        # 使用余弦学习率调度
        save_period=10,                     # 每10轮保存一次
        verbose=True                        # 添加详细输出
    )
except Exception as e:
    print(f"训练出错: {str(e)}")

print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")