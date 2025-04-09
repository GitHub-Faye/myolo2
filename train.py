from ultralytics import YOLO
import torch

# 1. 加载原始预训练模型
pretrained_model = YOLO('yolov12n.pt')
pretrained_weights = pretrained_model.model.state_dict()

# 2. 创建自定义人脸检测模型
custom_model = YOLO('yolov12-face.yaml')
custom_state = custom_model.model.state_dict()

# 打印详细的层信息
print("预训练模型层数:", len(pretrained_weights))
print("自定义模型层数:", len(custom_state))

# 比较前缀
pre_keys = list(pretrained_weights.keys())
cus_keys = list(custom_state.keys())
print("预训练模型前10个层:", pre_keys[:10])
print("自定义模型前10个层:", cus_keys[:10])

# 检查层名称格式差异
print("\n层名称比较:")
for i in range(min(5, len(pre_keys), len(cus_keys))):
    print(f"预训练: {pre_keys[i]} | 自定义: {cus_keys[i]}")

# 3. 匹配和加载权重
matched_weights = {}

# 遍历自定义模型的所有层
for k_cus in custom_state.keys():
    # 如果该层在预训练模型中存在，且形状相同
    if k_cus in pretrained_weights and custom_state[k_cus].shape == pretrained_weights[k_cus].shape:
        matched_weights[k_cus] = pretrained_weights[k_cus]

print(f"\n成功匹配 {len(matched_weights)}/{len(custom_state)} 层")

# 4. 加载匹配的权重
if len(matched_weights) > 0:
    custom_model.model.load_state_dict(matched_weights, strict=False)
    print("成功加载预训练权重")
else:
    print("未找到匹配的权重")

# 5. 开始训练
results = custom_model.train(
    data='widerface_litel/data.yaml',  # 您的数据集配置文件
    epochs=1,
    imgsz=640,
    batch=2,
    task='pose',                      # 使用pose任务类型
    name='yolov12-face',
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    patience=50,
    save_period=10,
)