# YOLOv12训练流程指南

## 目录
- [训练流程概述](#训练流程概述)
- [训练数据准备](#训练数据准备)
- [自定义数据集](#自定义数据集)
- [训练命令和参数](#训练命令和参数)
- [训练流程定制](#训练流程定制)
  - [修改数据增强策略](#修改数据增强策略)
  - [修改训练超参数](#修改训练超参数)
  - [修改优化器和学习率调度](#修改优化器和学习率调度)
  - [修改模型结构](#修改模型结构)
  - [修改损失函数](#修改损失函数)
  - [在自定义模型中使用预训练参数](#在自定义模型中使用预训练参数)
  - [添加自定义回调函数](#添加自定义回调函数)
- [高级训练技巧](#高级训练技巧)
  - [分布式训练](#分布式训练)
  - [混合精度训练](#混合精度训练)
  - [早停和模型保存](#早停和模型保存)

## 训练流程概述

YOLOv12训练流程基于Ultralytics框架实现，整体流程如下：

1. **初始化阶段**：加载配置、设置设备、创建保存目录
2. **数据准备阶段**：构建数据集和数据加载器
3. **模型构建阶段**：创建或加载预训练模型
4. **训练循环阶段**：迭代训练多个epoch，包括前向传播、损失计算、反向传播和参数更新
5. **验证评估阶段**：定期在验证集上评估模型性能
6. **保存阶段**：保存最佳模型和最后模型

YOLOv12的训练由`DetectionTrainer`类管理，它继承自`BaseTrainer`类。核心训练逻辑在`_do_train`方法中实现。

## 训练数据准备

### 数据集格式

YOLOv12使用YAML格式的数据配置文件，例如：

```yaml
path: /path/to/dataset  # 数据集根目录
train: train  # 训练集路径（相对于path）
val: val      # 验证集路径（相对于path）
test: test    # 测试集路径（相对于path）

nc: 80        # 类别数量
names: [...]  # 类别名称列表
```

数据集应当符合以下结构：
```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       └── ...
├── val/
└── test/
```

### 标签格式

每个标签文件对应一张图像，包含以下格式的行：
```
class_id x_center y_center width height
```
所有值都是归一化到0-1范围的浮点数。

### 准备COCO格式数据

如果你有COCO格式的数据集，可以使用内置的转换工具：

```python
from ultralytics.data.converter import convert_coco
convert_coco('path/to/coco', use_segments=True)
```

## 自定义数据集

### 创建自定义数据集

要创建自定义数据集用于YOLOv12训练，需要遵循以下步骤：

#### 1. 收集和标注数据

1. **收集图像**：收集与任务相关的图像数据
2. **标注工具**：使用以下工具之一标注数据：
   - [Roboflow](https://roboflow.com/)：提供完整的数据管理和标注解决方案
   - [LabelImg](https://github.com/tzutalin/labelImg)：用于边界框标注的轻量级工具
   - [CVAT](https://www.cvat.ai/)：功能丰富的标注工具，支持多种标注类型
   - [Labelme](https://github.com/wkentaro/labelme)：支持各种类型标注的工具

3. **导出格式**：将标注结果导出为YOLO格式，每个图像对应一个文本文件，每行包含：
   ```
   class_id x_center y_center width height
   ```

#### 2. 组织数据集结构

按照以下结构组织数据集：
```
custom_dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
├── val/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── test/  # 可选
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

#### 3. 创建数据集配置文件

创建一个`custom_dataset.yaml`文件，内容如下：

```yaml
# 数据集路径和类别信息
path: /absolute/path/to/custom_dataset  # 数据集根目录绝对路径
train: train/images  # 训练图像相对路径
val: val/images      # 验证图像相对路径
test: test/images    # 测试图像相对路径（可选）

# 类别信息
nc: 3  # 类别数量
names: ['class1', 'class2', 'class3']  # 类别名称

# 可选：关键点信息（如果有）
kpt_shape: [17, 3]  # [关键点数量, 关键点维度]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 水平翻转索引
```

### 自定义数据集的常见问题及解决方案

#### 数据集划分

如果需要将收集的数据划分为训练集、验证集和测试集，可以使用以下Python脚本：

```python
import os
import random
import shutil
from pathlib import Path

# 配置
dataset_path = 'path/to/all_images'
output_path = 'path/to/custom_dataset'
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 创建目录结构
for split in ['train', 'val', 'test']:
    for folder in ['images', 'labels']:
        os.makedirs(os.path.join(output_path, split, folder), exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(os.path.join(dataset_path, 'images')) 
               if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

# 计算每个集合的大小
n_train = int(len(image_files) * train_ratio)
n_val = int(len(image_files) * val_ratio)

# 分割数据
train_files = image_files[:n_train]
val_files = image_files[n_train:n_train + n_val]
test_files = image_files[n_train + n_val:]

# 复制文件到各自目录
def copy_files(file_list, split):
    for f in file_list:
        # 复制图像
        src_img = os.path.join(dataset_path, 'images', f)
        dst_img = os.path.join(output_path, split, 'images', f)
        shutil.copy2(src_img, dst_img)
        
        # 复制标签（假设标签文件与图像文件同名，扩展名为.txt）
        label_file = os.path.splitext(f)[0] + '.txt'
        src_label = os.path.join(dataset_path, 'labels', label_file)
        if os.path.exists(src_label):
            dst_label = os.path.join(output_path, split, 'labels', label_file)
            shutil.copy2(src_label, dst_label)

copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print(f"数据集划分完成: {len(train_files)} 训练, {len(val_files)} 验证, {len(test_files)} 测试")
```

#### 数据集增强

为小型数据集增加样本量，可以使用以下技术：

1. **使用Ultralytics内置增强**：YOLOv12默认启用mosaic、随机裁剪等增强
   
2. **使用Roboflow进行自动增强**：Roboflow平台提供多种自动化增强选项

3. **手动实现额外增强**：
   ```python
   from ultralytics import YOLO
   from ultralytics.data.augment import Compose, Mosaic, RandomPerspective, LetterBox
   
   # 自定义增强组合
   custom_augment = Compose([
       Mosaic(prob=0.8),
       RandomPerspective(degrees=10, translate=0.1, scale=0.5, shear=2.0, perspective=0.0, border=(0, 0)),
       LetterBox(new_shape=(640, 640))
   ])
   
   # 在训练时使用
   model = YOLO('yolov12n.pt')
   model.train(data='custom_dataset.yaml', epochs=100)
   ```

#### 类别不平衡处理

当数据集中某些类别样本数量远少于其他类别时：

1. **调整类别权重**：
   ```yaml
   # 在数据集配置中添加类别权重
   class_weights: [1.0, 2.0, 4.0]  # 样本较少的类别给予更高权重
   ```

2. **过采样少数类**：复制少数类样本以平衡数据分布

3. **通过训练参数处理**：
   ```bash
   yolo train model=yolov12n.pt data=custom_dataset.yaml fl_gamma=1.5
   ```
   增加焦点损失gamma值，使模型更关注难分类的样本

#### 小物体检测优化

对于包含大量小物体的数据集：

1. **增加输入分辨率**：
   ```bash
   yolo train model=yolov12n.pt data=custom_dataset.yaml imgsz=1280
   ```

2. **优化锚框**：
   ```python
   from ultralytics.utils.autoancher import AutoAnchor
   
   # 为数据集自动计算最佳锚框
   auto_anchor = AutoAnchor(dataset, model)
   new_anchors = auto_anchor.run()
   print(f"优化后的锚框: {new_anchors}")
   ```

3. **使用更大的模型变体**：
   ```bash
   yolo train model=yolov12l.pt data=custom_dataset.yaml
   ```

### 数据集验证

在训练前验证数据集的正确性：

```python
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset

# 加载数据集配置
data_path = 'custom_dataset.yaml'
model = YOLO('yolov12n.pt')

# 构建数据集以检查格式
dataset = build_yolo_dataset(model.args, img_path='custom_dataset/train/images', 
                            batch=1, data=data_path, mode='train')

# 检查第一个样本
batch = next(iter(build_dataloader(dataset, batch_size=1, workers=1, shuffle=False, rank=-1)))
print(f"批次形状: {batch['img'].shape}")
print(f"标签数: {len(batch['cls'])}")

# 可视化数据集样本
model.plot_training_samples(batch=batch, fname='dataset_samples.jpg')
```

### 使用自定义数据集进行训练

一旦准备好数据集，可以使用以下命令开始训练：

```bash
# 命令行方式
yolo train model=yolov12n.pt data=path/to/custom_dataset.yaml epochs=100

# Python方式
from ultralytics import YOLO

model = YOLO('yolov12n.pt')
results = model.train(
    data='path/to/custom_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='custom_yolov12'
)
```

## 训练命令和参数

### 基本训练命令

```bash
yolo train model=yolov12n.pt data=custom.yaml epochs=100 imgsz=640
```

### 常用训练参数

| 参数 | 说明 | 默认值 |
|------|------|-------|
| model | 模型路径或配置文件 | yolov12n.pt |
| data | 数据集配置文件 | coco8.yaml |
| epochs | 训练轮数 | 100 |
| patience | 早停patience | 100 |
| batch | 批量大小 | 16 |
| imgsz | 输入图像尺寸 | 640 |
| save | 是否保存模型 | True |
| save_period | 每隔多少epoch保存 | -1 |
| cache | 是否缓存图像 | False |
| device | 训练设备 | '' |
| workers | 数据加载线程数 | 8 |
| optimizer | 优化器 | 'auto' |
| lr0 | 初始学习率 | 0.01 |
| lrf | 最终学习率因子 | 0.01 |
| momentum | 动量 | 0.937 |
| weight_decay | 权重衰减 | 0.0005 |
| warmup_epochs | 预热轮数 | 3.0 |
| warmup_momentum | 预热动量 | 0.8 |
| warmup_bias_lr | 预热偏置学习率 | 0.1 |
| box | 边界框损失权重 | 7.5 |
| cls | 分类损失权重 | 0.5 |
| dfl | 分布焦点损失权重 | 1.5 |
| fl_gamma | 焦点损失gamma | 0.0 |
| label_smoothing | 标签平滑 | 0.0 |
| nbs | 标称批大小 | 64 |
| val | 是否进行验证 | True |

### 数据增强参数

| 参数 | 说明 | 默认值 |
|------|------|-------|
| hsv_h | HSV-Hue增强 | 0.015 |
| hsv_s | HSV-Saturation增强 | 0.7 |
| hsv_v | HSV-Value增强 | 0.4 |
| degrees | 旋转增强 | 0.0 |
| translate | 平移增强 | 0.1 |
| scale | 缩放增强 | 0.5 |
| shear | 剪切增强 | 0.0 |
| perspective | 透视增强 | 0.0 |
| flipud | 上下翻转增强 | 0.0 |
| fliplr | 左右翻转增强 | 0.5 |
| mosaic | 马赛克增强 | 1.0 |
| mixup | 混合增强 | 0.0 |
| copy_paste | 复制粘贴增强 | 0.0 |

## 训练流程定制

### 修改数据增强策略

1. **通过命令行参数修改**:
   ```bash
   yolo train model=yolov12n.pt data=custom.yaml mosaic=0.5 fliplr=0.7 scale=0.6
   ```

2. **通过修改代码修改**:
   在`ultralytics/data/augment.py`中修改数据增强转换。

3. **添加自定义增强**:
   ```python
   from ultralytics import YOLO
   from ultralytics.data.augment import Albumentations
   import albumentations as A

   # 自定义Albumentations增强
   custom_transforms = A.Compose([
       A.RandomBrightnessContrast(p=0.5),
       A.GaussianBlur(p=0.3)
   ])

   # 训练时使用自定义增强
   model = YOLO('yolov12n.pt')
   model.train(data='custom.yaml', epochs=100, augment=True)
   ```

### 修改训练超参数

1. **通过命令行参数修改**:
   ```bash
   yolo train model=yolov12n.pt data=custom.yaml epochs=150 lr0=0.005 lrf=0.005 weight_decay=0.001
   ```

2. **通过YAML配置修改**:
   创建一个`custom_train.yaml`配置文件:
   ```yaml
   model: yolov12n.pt
   data: custom.yaml
   epochs: 150
   lr0: 0.005
   lrf: 0.005
   weight_decay: 0.001
   ```
   
   然后使用该配置文件进行训练:
   ```bash
   yolo train cfg=custom_train.yaml
   ```

3. **通过Python代码修改**:
   ```python
   from ultralytics import YOLO

   model = YOLO('yolov12n.pt')
   model.train(
       data='custom.yaml',
       epochs=150,
       lr0=0.005,
       lrf=0.005,
       weight_decay=0.001
   )
   ```

### 修改优化器和学习率调度

1. **更改优化器**:
   ```bash
   yolo train model=yolov12n.pt data=custom.yaml optimizer=AdamW
   ```
   
   支持的优化器包括: `SGD`, `Adam`, `AdamW`, `RMSProp`等。

2. **自定义学习率调度器**:
   YOLOv12默认使用余弦退火学习率调度。你可以在Python代码中自定义学习率调度:
   
   ```python
   from ultralytics import YOLO
   import torch.optim as optim

   model = YOLO('yolov12n.pt')
   
   # 自定义优化器和学习率调度
   def custom_optimizer(model):
       return optim.Adam(model.parameters(), lr=0.001)
   
   def custom_scheduler(optimizer):
       return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
   
   model.train(
       data='custom.yaml',
       epochs=100,
       optimizer=custom_optimizer,
       scheduler=custom_scheduler
   )
   ```

### 修改模型结构

1. **使用不同的预定义模型**:
   ```bash
   # 使用不同大小的YOLOv12模型
   yolo train model=yolov12n.pt data=custom.yaml  # nano模型
   yolo train model=yolov12s.pt data=custom.yaml  # small模型
   yolo train model=yolov12m.pt data=custom.yaml  # medium模型
   yolo train model=yolov12l.pt data=custom.yaml  # large模型
   yolo train model=yolov12x.pt data=custom.yaml  # xlarge模型
   ```

2. **修改现有模型结构**:
   
   你可以修改`ultralytics/cfg/models/v12/yolov12.yaml`创建自定义模型:
   
   ```yaml
   # 修改主干网络或头部
   backbone:
     # [层次, 重复次数, 模块, 参数...]
     - [-1, 1, Conv, [64, 3, 2]]  # 添加或修改层
     
   head:
     - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
     - [[-1, 6], 1, Concat, [1]]  # 修改跳连接
   ```

3. **初始化自定义模型并训练**:
   ```python
   from ultralytics import YOLO
   
   # 从自定义YAML创建模型
   model = YOLO('path/to/custom_model.yaml')
   
   # 训练自定义模型
   model.train(data='custom.yaml', epochs=100)
   ```

### 修改损失函数

YOLOv12使用多个损失函数组件共同优化目标检测性能。默认损失函数包括边界框损失(box_loss)、分类损失(cls_loss)和分布焦点损失(dfl_loss)。以下是几种修改损失函数的方法：

#### 1. 调整损失权重

最简单的方法是通过命令行参数调整不同损失组件的权重：

```bash
# 修改边界框、分类和DFL损失权重
yolo train model=yolov12n.pt data=custom.yaml box=10.0 cls=1.0 dfl=2.0
```

#### 2. 修改焦点损失参数

调整焦点损失的gamma参数来处理简单/困难样本的权重：

```bash
# 增加gamma值，使模型更关注难分类的样本
yolo train model=yolov12n.pt data=custom.yaml fl_gamma=2.0
```

#### 3. 完全自定义损失函数

要深度修改损失函数，需要继承并重写模型的损失计算方法：

```python
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

class CustomDetectionModel(DetectionModel):
    def __init__(self, cfg='yolov12n.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        # 可以在这里添加额外的初始化，例如自定义损失层
        self.custom_loss = nn.SmoothL1Loss(reduction='none')
    
    def _custom_box_loss(self, pred_bboxes, target_bboxes, target_scores, target_scores_sum):
        # 自定义边界框损失实现
        # 这里使用平滑L1损失替代默认的CIoU损失
        bbox_loss = self.custom_loss(pred_bboxes, target_bboxes).sum(-1)
        bbox_loss = (bbox_loss * target_scores).sum() / target_scores_sum
        return bbox_loss
    
    def loss(self, batch):
        # 这是从最新的YOLO模型中获取损失计算逻辑的简化版本
        # 实际实现可能需要根据当前版本的DetectionModel.loss()方法进行调整
        device = batch["img"].device
        loss = torch.zeros(3, device=device)  # box, cls, dfl
        
        # 计算预测结果
        pred_distri, pred_scores = self(batch["img"])
        
        # 处理标签数据...
        # (此处省略了大量标签预处理代码)
        
        # 使用自定义边界框损失
        loss[0] = self._custom_box_loss(pred_bboxes, target_bboxes, target_scores, target_scores_sum)
        
        # 其他损失计算...
        # (此处省略了分类损失和DFL损失计算代码)
        
        return loss.sum() * batch["img"].shape[0], loss.detach()  # 总损失, 各部分损失值

# 使用自定义模型
custom_model = CustomDetectionModel('yolov12n.yaml')
model = YOLO(model=custom_model)
model.train(data='custom.yaml', epochs=50)
```

#### 4. 添加额外的损失组件

添加额外的损失组件，如特征蒸馏损失或正则化损失：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

class DistillationDetectionModel(DetectionModel):
    def __init__(self, cfg='yolov12n.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        # 加载教师模型
        self.teacher = YOLO('yolov12l.pt').model
        for param in self.teacher.parameters():
            param.requires_grad = False  # 冻结教师模型
        self.teacher.eval()
        
        # 蒸馏损失权重
        self.distill_weight = 0.5
    
    def loss(self, batch):
        device = batch["img"].device
        loss = torch.zeros(4, device=device)  # box, cls, dfl, distill
        
        # 计算学生模型预测
        pred_distri, pred_scores = self(batch["img"])
        
        # 计算教师模型预测
        with torch.no_grad():
            teacher_distri, teacher_scores = self.teacher(batch["img"])
        
        # 计算标准YOLO损失（前三个组件）
        # (此处省略了标准损失计算代码)
        
        # 计算蒸馏损失（第四个组件）
        # 使用KL散度作为特征蒸馏损失
        distill_loss = F.kl_div(
            F.log_softmax(pred_scores / 0.1, dim=1),
            F.softmax(teacher_scores / 0.1, dim=1),
            reduction='batchmean'
        ) * (0.1 ** 2)
        
        loss[3] = distill_loss * self.distill_weight
        
        return loss.sum() * batch["img"].shape[0], loss.detach()

# 使用带蒸馏的自定义模型
distill_model = DistillationDetectionModel('yolov12n.yaml')
model = YOLO(model=distill_model)
model.train(data='custom.yaml', epochs=50)
```

#### 5. 使用注意力损失

加入注意力损失来改善特征关注度：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

class AttentionLossModel(DetectionModel):
    def __init__(self, cfg='yolov12n.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        # 添加注意力模块
        self.attention_weight = 0.1
    
    def _attention_loss(self, features, targets):
        """计算特征图与目标区域的注意力损失"""
        loss = 0
        # 假设features是模型中间层特征图，targets是目标信息
        # 此处实现根据目标位置生成注意力图并与特征图比较的损失
        # 实际实现需要根据具体的模型结构和任务需求定制
        return loss * self.attention_weight
    
    def loss(self, batch):
        # 扩展标准损失计算以包含注意力损失
        # (此处省略了标准损失计算代码)
        
        # 添加注意力损失到总损失中
        attention_loss = self._attention_loss(features, targets)
        total_loss = standard_loss + attention_loss
        
        return total_loss, (standard_loss_detached, attention_loss.detach())
```

通过以上方法，你可以根据具体任务需求定制YOLOv12的损失函数，来提高模型在特定场景下的性能。

### 在自定义模型中使用预训练参数

当你修改了YOLOv12的模型结构后，通常希望能利用原始预训练模型的参数来加速训练。以下是几种在自定义模型中使用预训练参数的方法：

#### 1. 部分加载预训练权重

使用Python代码自定义权重加载过程：

```python
from ultralytics import YOLO
import torch

# 加载原始预训练模型
pretrained_model = YOLO('yolov12n.pt')
pretrained_weights = pretrained_model.model.state_dict()

# 创建自定义模型
custom_model = YOLO('path/to/custom_model.yaml')
custom_state = custom_model.model.state_dict()

# 筛选并加载匹配的权重参数
matched_weights = {k: v for k, v in pretrained_weights.items() if k in custom_state and custom_state[k].shape == v.shape}
print(f"加载了 {len(matched_weights)}/{len(custom_state)} 层预训练权重")

# 更新自定义模型参数
custom_model.model.load_state_dict(matched_weights, strict=False)

# 开始训练
custom_model.train(data='custom.yaml', epochs=100)
```

#### 2. 层名称映射加载

处理层名称变更情况下的权重迁移：

```python
from ultralytics import YOLO
import torch

# 加载预训练权重
pretrained_weights = torch.load('yolov12n.pt', map_location='cpu')
if 'model' in pretrained_weights:
    pretrained_weights = pretrained_weights['model'].float().state_dict()
else:
    pretrained_weights = pretrained_weights['state_dict']

# 创建自定义模型
custom_model = YOLO('custom_model.yaml')
custom_state = custom_model.model.state_dict()

# 定义层名称映射（旧层名->新层名）
layer_mapping = {
    'model.0.conv': 'model.0.new_conv',
    'model.1.cv1.conv': 'model.1.custom_cv1.conv',
    # 添加更多映射...
}

# 基于映射加载权重
matched_count = 0
for k_old, v_old in pretrained_weights.items():
    # 检查是否有映射
    k_new = layer_mapping.get(k_old, k_old)
    if k_new in custom_state and custom_state[k_new].shape == v_old.shape:
        custom_state[k_new] = v_old
        matched_count += 1

custom_model.model.load_state_dict(custom_state, strict=False)
print(f"成功加载 {matched_count}/{len(custom_state)} 层参数")

# 开始训练
custom_model.train(data='custom.yaml', epochs=100)
```

#### 3. 模块化权重加载

针对特定模块进行权重迁移：

```python
from ultralytics import YOLO
import torch

# 获取原始模型的主干网络权重
original_model = YOLO('yolov12n.pt')
backbone_weights = {k: v for k, v in original_model.model.state_dict().items() if 'backbone' in k}

# 创建自定义模型
custom_model = YOLO('custom_model.yaml')

# 加载主干网络权重
custom_state = custom_model.model.state_dict()
matched_backbone = {k: v for k, v in backbone_weights.items() if k in custom_state and custom_state[k].shape == v.shape}
custom_model.model.load_state_dict(matched_backbone, strict=False)

print(f"成功加载主干网络参数: {len(matched_backbone)} 层")

# 开始训练，只针对检测头进行训练，冻结主干网络
custom_model.train(data='custom.yaml', epochs=50, freeze=[0, 10])  # 冻结前10层
```

#### 4. 特征提取与迁移

完全不同结构模型间的特征迁移：

```python
from ultralytics import YOLO
import torch
import torch.nn as nn

class CustomYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载原始模型作为特征提取器
        self.feature_extractor = YOLO('yolov12n.pt').model.backbone
        
        # 冻结特征提取器
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 添加自定义检测头
        self.custom_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 5 * (80 + 4), kernel_size=1)  # 假设80个类别
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.custom_head(features[-1])
        return output

# 创建并训练自定义模型
custom_yolo = CustomYOLO()
model = YOLO(model=custom_yolo)
model.train(data='custom.yaml', epochs=100)
```

#### 5. 逐层参数分析与迁移

当模型结构有较大差异时的权重分析和迁移：

```python
from ultralytics import YOLO
import torch
import numpy as np

# 加载原始模型
original_model = YOLO('yolov12n.pt')
original_weights = original_model.model.state_dict()

# 创建自定义模型
custom_model = YOLO('custom_model.yaml')
custom_state = custom_model.model.state_dict()

# 逐层分析参数统计特性
for name, param in original_weights.items():
    if name in custom_state and custom_state[name].shape == param.shape:
        # 直接加载匹配参数
        custom_state[name] = param
    elif name in custom_state and custom_state[name].shape != param.shape:
        # 分析参数统计特性
        orig_mean = param.mean().item()
        orig_std = param.std().item()
        
        # 基于统计特性初始化新参数
        if 'conv.weight' in name:
            # 卷积权重通常需要保持统计特性
            nn.init.normal_(custom_state[name], mean=orig_mean, std=orig_std)
            print(f"参数统计初始化: {name}, 均值={orig_mean:.4f}, 标准差={orig_std:.4f}")

# 加载处理后的参数
custom_model.model.load_state_dict(custom_state)

# 开始训练
custom_model.train(data='custom.yaml', epochs=100)
```

通过以上方法，你可以在修改模型结构后充分利用预训练参数，加速训练收敛并提高最终性能。特别是针对较小数据集或计算资源有限的情况，这种迁移学习方法非常有效。

### 添加自定义回调函数

在训练过程中可以添加自定义回调函数:

```python
from ultralytics import YOLO
from ultralytics.utils.callbacks.base import Callback

class CustomCallback(Callback):
    def on_train_start(self, trainer):
        print("训练开始!")
    
    def on_train_epoch_end(self, trainer):
        print(f"Epoch {trainer.epoch} 完成，当前损失: {trainer.loss:.4f}")
    
    def on_train_end(self, trainer):
        print("训练结束!")

# 初始化模型并添加回调函数
model = YOLO('yolov12n.pt')
model.add_callback("custom", CustomCallback())

# 开始训练
model.train(data='custom.yaml', epochs=10)
```

支持的回调事件包括:
- `on_pretrain_routine_start/end`
- `on_train_start/end`
- `on_train_epoch_start/end`
- `on_train_batch_start/end`
- `on_val_start/end`
- `on_val_batch_start/end`
- `on_fit_epoch_end`
- `on_model_save`
- `on_params_update`
- `teardown`

## 高级训练技巧

### 分布式训练

YOLOv12支持使用PyTorch的分布式数据并行(DDP)进行多GPU训练:

```bash
# 使用所有可用GPU
python -m torch.distributed.run --nproc_per_node=-1 ultralytics/yolo/train.py model=yolov12n.pt data=custom.yaml epochs=100
```

### 混合精度训练

默认启用混合精度训练以提高性能和内存效率:

```bash
# 启用混合精度训练
yolo train model=yolov12n.pt data=custom.yaml amp=True

# 禁用混合精度训练
yolo train model=yolov12n.pt data=custom.yaml amp=False
```

### 早停和模型保存

控制早停和模型保存策略:

```bash
# 设置早停patience
yolo train model=yolov12n.pt data=custom.yaml patience=50

# 自定义保存间隔
yolo train model=yolov12n.pt data=custom.yaml save_period=10  # 每10个epoch保存一次
```

### 恢复训练

从上次训练的检查点恢复训练:

```bash
# 从最后一个检查点恢复
yolo train model=path/to/last.pt data=custom.yaml resume=True
```

### 冻结层训练

在微调过程中冻结某些层:

```bash
# 冻结主干网络前10层
yolo train model=yolov12n.pt data=custom.yaml freeze=10
```

通过这些方法，你可以根据自己的需求定制YOLOv12的训练流程，以获得最佳的模型性能。 