# YOLOv12模型评估指南

## 模型评估概述

训练完成后的模型评估是确保模型性能和质量的关键步骤。YOLOv12提供了全面的评估工具，可以测量模型在检测任务上的精确度、召回率和平均精度(mAP)等关键指标。

## 基本评估命令

### 命令行评估

```bash
yolo val model=runs/train/exp/weights/best.pt data=数据集配置文件.yaml
```

### Python代码评估

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/train/exp/weights/best.pt')

# 对模型进行评估
results = model.val(data='数据集配置文件.yaml')

# 查看结果
print(results.box.map)  # 打印mAP50-95
print(results.box.map50)  # 打印mAP50
print(results.box.mp)  # 打印平均精确率(mean precision)
print(results.box.mr)  # 打印平均召回率(mean recall)
```

## 评估参数选项

评估时可以传递以下参数来自定义评估过程：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model` | 模型路径 | 无 |
| `data` | 数据集配置文件 | 无 |
| `imgsz` | 图像大小 | 640 |
| `batch` | 批处理大小 | 16 |
| `device` | 使用的设备 (如"0"表示第一个GPU) | 自动选择 |
| `save_txt` | 保存结果为.txt文件 | False |
| `save_json` | 保存结果为COCO格式的JSON文件 | False |
| `save_hybrid` | 保存标注和预测标签的混合版本 | False |
| `conf` | 置信度阈值 | 0.001 |
| `iou` | IoU阈值 | 0.6 |
| `half` | 使用FP16半精度 | False |
| `rect` | 使用矩形推理 | True |
| `plots` | 保存评估结果图像 | False |

例如：

```bash
yolo val model=runs/train/exp/weights/best.pt data=coco.yaml imgsz=640 batch=8 device=0 save_json
```

## 评估指标解释

YOLOv12评估后会返回多项关键指标：

1. **mAP50** (mAP@0.5): 在IoU阈值为0.5时的平均精度
2. **mAP50-95** (mAP@0.5:0.95): 在IoU阈值从0.5到0.95(步长0.05)的平均精度
3. **精确率(Precision)**: 正确检测的目标与所有检测目标的比率
4. **召回率(Recall)**: 正确检测的目标与所有真实目标的比率
5. **F1分数**: 精确率和召回率的调和平均值

## 结果可视化

使用`plots=True`选项可以生成评估结果的可视化图表，包括：

- 混淆矩阵
- 精确率-召回率曲线
- F1-置信度曲线

这些图表保存在模型评估目录中，通常为`runs/val/exp/`。

```bash
yolo val model=runs/train/exp/weights/best.pt data=coco.yaml plots=True
```

## 使用自定义数据集评估模型

如果您有自定义数据集，可以按照以下步骤进行评估：

### 1. 准备验证数据集

确保您的验证数据集已按照YOLOv12格式组织。典型的结构如下：

```
dataset/
├── images/
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── labels/
    └── val/
        ├── image1.txt
        ├── image2.txt
        └── ...
```

### 2. 创建数据集配置文件

如果尚未创建，请创建一个YAML配置文件，例如`custom_dataset.yaml`：

```yaml
# 数据集路径
path: ./dataset  # 数据集根目录
train: images/train  # 训练图像相对路径
val: images/val  # 验证图像相对路径
test:  # 测试图像相对路径（可选）

# 类别
names:
  0: 类别1
  1: 类别2
  # ...
```

### 3. 运行评估

使用您的配置文件执行评估：

```bash
yolo val model=runs/train/exp/weights/best.pt data=custom_dataset.yaml
```

### 4. 跨数据集评估

有时，需要在模型未见过的数据集上评估其泛化能力：

```bash
yolo val model=runs/train/exp/weights/best.pt data=new_dataset.yaml
```

## 深入分析验证结果

### 按类别分析性能

验证完成后，您可以查看每个类别的性能：

```python
from ultralytics import YOLO

model = YOLO('runs/train/exp/weights/best.pt')
results = model.val(data='custom_dataset.yaml')

# 按类别输出性能
for i, c in enumerate(results.names.values()):
    print(f"类别: {c}, mAP50: {results.maps[i]:.4f}")
```

### 分析困难样本

查看模型表现不佳的图像，以便进一步改进：

1. 使用`save_txt=True`和`save_conf=True`选项保存预测结果
2. 分析低置信度预测或漏检的样本
3. 考虑添加更多类似样本到训练集中

### 混淆矩阵分析

混淆矩阵可以帮助识别经常混淆的类别：

```bash
yolo val model=runs/train/exp/weights/best.pt data=custom_dataset.yaml plots=True
```

然后查看生成的`confusion_matrix.png`文件。

## 针对特定应用场景的评估

### 低延迟应用评估

对于需要实时性能的应用，评估推理速度也很重要：

```bash
yolo val model=runs/train/exp/weights/best.pt data=custom_dataset.yaml benchmark=True
```

### 小目标检测评估

如果应用涉及小目标检测，可以调整评估参数：

```bash
yolo val model=runs/train/exp/weights/best.pt data=custom_dataset.yaml imgsz=1280
```

增大图像尺寸可以提高小目标检测性能。

### 资源受限环境评估

在资源受限设备上评估模型性能：

```bash
yolo val model=runs/train/exp/weights/best.pt data=custom_dataset.yaml device=cpu half=True
```

## 模型比较

要比较不同模型的性能，可以执行多次评估，然后比较其结果：

```python
model1 = YOLO('runs/train/exp1/weights/best.pt')
model2 = YOLO('runs/train/exp2/weights/best.pt')

results1 = model1.val(data='数据集配置文件.yaml')
results2 = model2.val(data='数据集配置文件.yaml')

print(f"模型1 mAP50-95: {results1.box.map:.4f}")
print(f"模型2 mAP50-95: {results2.box.map:.4f}")
```

## 最佳实践

1. **使用相同的评估数据集**: 确保在比较不同模型时使用相同的验证数据集。
2. **设置适当的置信度阈值**: 默认值是0.001，可以根据需要调整。
3. **使用批量大小**: 根据GPU内存调整batch大小以优化评估速度。
4. **保存评估结果**: 使用`save_json`或`save_txt`选项保存详细结果以便进一步分析。
5. **检查边缘情况**: 仔细查看模型在具有挑战性的图像上的表现。

## 使用验证结果改进模型

评估结果可以帮助识别模型的优缺点：

1. **低mAP值**: 可能需要更多训练数据、更长的训练时间或调整模型架构。
2. **低精确率**: 考虑增加置信度阈值或处理误检问题。
3. **低召回率**: 可能需要更低的置信度阈值或改进模型对困难样本的检测能力。
4. **类别不平衡**: 查看每个类别的性能，可能需要为表现不佳的类别增加训练样本。

## 评估特定形式的模型

YOLOv12支持评估多种导出格式的模型：

```bash
# PyTorch模型
yolo val model=yolo11n.pt data=coco.yaml

# ONNX模型
yolo val model=yolo11n.onnx data=coco.yaml

# TensorRT模型
yolo val model=yolo11n.engine data=coco.yaml
```

## 排查评估问题

如果遇到评估问题：

1. 确保模型和数据集路径正确
2. 检查设备兼容性和内存要求
3. 尝试减小批处理大小或图像尺寸
4. 查看日志文件了解详细错误信息

## 结论

有效的模型评估是机器学习流程中不可或缺的一部分。通过YOLOv12提供的评估工具，可以全面了解模型性能，为进一步优化提供依据。定期评估模型并记录结果，可以系统地改进检测性能并确保模型满足应用需求。 