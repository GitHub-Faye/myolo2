# YOLOv12创建与修改指南

## 目录

1. [YOLOv12模型简介](#1-yolov12模型简介)
   - [1.1 核心特点](#11-核心特点)
   - [1.2 模型规格](#12-模型规格)
2. [YOLOv12模型配置文件详解](#2-yolov12模型配置文件详解)
   - [2.1 配置文件基本结构](#21-配置文件基本结构)
   - [2.2 参数详解](#22-参数详解)
   - [2.3 核心模块配置详解](#23-核心模块配置详解)
3. [创建YOLOv12模型](#3-创建yolov12模型)
   - [3.1 从预训练模型加载](#31-从预训练模型加载)
   - [3.2 从配置文件创建模型的详细过程](#32-从配置文件创建模型的详细过程)
   - [3.3 配置文件与模型构建的内部机制](#33-配置文件与模型构建的内部机制)
   - [3.4 自定义数据集训练](#34-自定义数据集训练)
4. [修改YOLOv12模型](#4-修改yolov12模型)
   - [4.1 修改模型架构](#41-修改模型架构)
   - [4.2 修改区域注意力参数](#42-修改区域注意力参数)
   - [4.3 自定义类别数和任务](#43-自定义类别数和任务)
   - [4.4 高级修改](#44-高级修改)
5. [实践示例](#5-实践示例)
   - [5.1 创建轻量化YOLOv12模型](#51-创建轻量化yolov12模型)
   - [5.2 高精度YOLOv12模型](#52-高精度yolov12模型)
   - [5.3 训练自定义数据集的完整示例](#53-训练自定义数据集的完整示例)
6. [添加新模型到YOLOv12项目](#6-添加新模型到yolov12项目)
   - [6.1 模型配置规范](#61-模型配置规范)
   - [6.2 添加新模型的步骤](#62-添加新模型的步骤)
   - [6.3 高级模型定制示例](#63-高级模型定制示例)
   - [6.4 模型发布和共享](#64-模型发布和共享)
   - [6.5 模型配置规范检查清单](#65-模型配置规范检查清单)
7. [添加新任务类型到YOLOv12项目](#7-添加新任务类型到yolov12项目)
   - [7.1 任务类型扩展规范](#71-任务类型扩展规范)
   - [7.2 添加新任务的详细步骤](#72-添加新任务的详细步骤)
   - [7.3 创建任务特定的配置文件](#73-创建任务特定的配置文件)
   - [7.4 准备人脸关键点数据集配置](#74-准备人脸关键点数据集配置)
   - [7.5 测试和使用新任务](#75-测试和使用新任务)
   - [7.6 任务添加检查清单](#76-任务添加检查清单)
8. [总结](#8-总结)
9. [YOLOv12模型加载机制和任务注册流程](#9-yolov12模型加载机制和任务注册流程)
   - [9.1 模型加载机制](#91-模型加载机制)
   - [9.2 任务注册流程](#92-任务注册流程)
   - [9.3 任务组件执行流程](#93-任务组件执行流程)
   - [9.4 添加新任务的最佳实践](#94-添加新任务的最佳实践)
10. [自定义模块与模块注册](#10-自定义模块与模块注册)
   - [10.1 创建自定义模块的基本步骤](#101-创建自定义模块的基本步骤)
   - [10.2 自定义模块类的实现](#102-自定义模块类的实现)
   - [10.3 模块注册与导入](#103-模块注册与导入)
   - [10.4 在YAML配置文件中使用自定义模块](#104-在yaml配置文件中使用自定义模块)
   - [10.5 模块注册示例](#105-模块注册示例)
   - [10.6 高级模块注册技巧](#106-高级模块注册技巧)
   - [10.7 常见问题与解决方案](#107-常见问题与解决方案)
   - [10.8 总结](#108-总结)
11. [自定义任务和模型注册](#11-自定义任务和模型注册)
   - [11.1 YOLOv12任务架构概述](#111-yolov12任务架构概述)
   - [11.2 自定义新任务的步骤](#112-自定义新任务的步骤)
   - [11.3 将自定义任务注册到模型](#113-将自定义任务注册到模型)
   - [11.4 模型训练和使用示例](#114-模型训练和使用示例)
   - [11.5 自定义损失函数](#115-自定义损失函数)
   - [11.6 常见问题与解决方案](#116-常见问题与解决方案)
12. [自定义损失函数与注册](#12-自定义损失函数与注册)
   - [12.1 YOLOv12损失函数架构](#121-yolov12损失函数架构)
   - [12.2 创建自定义损失函数](#122-创建自定义损失函数)
   - [12.3 注册自定义损失函数](#123-注册自定义损失函数)
   - [12.4 使用自定义损失函数的方法](#124-使用自定义损失函数的方法)
   - [12.5 高级损失函数自定义示例](#125-高级损失函数自定义示例)
   - [12.6 实用技巧与最佳实践](#126-实用技巧与最佳实践)
   - [12.7 总结](#127-总结)

## 1. YOLOv12模型简介

YOLOv12是一个注意力机制驱动的实时目标检测模型，由Yunjie Tian、Qixiang Ye和David Doermann开发。该模型引入了创新的区域注意力机制，在保持实时检测速度的同时显著提高了检测精度。

### 1.1 核心特点

- **区域注意力机制**：将特征图分为多个区域进行注意力计算，提高计算效率
- **A2C2f模块**：Area-Attention Cross-stage Channel Fusion，YOLOv12的核心创新模块
- **高效推理**：在T4 GPU上实现毫秒级推理时间
- **多规模模型**：提供从n(nano)到x(xlarge)的多个规模模型，满足不同场景需求

### 1.2 模型规格

| 模型 | 参数量 | FLOPs | 推理时间(T4) | mAP |
|------|--------|-------|-------------|-----|
| YOLOv12n | 2.6M | 6.2G | 1.64ms | 40.6% |
| YOLOv12s | 9.1M | 19.7G | 2.61ms | 48.0% |
| YOLOv12m | 19.7M | 60.4G | 4.86ms | 52.5% |
| YOLOv12l | 26.5M | 83.3G | 6.77ms | 53.7% |
| YOLOv12x | 59.4M | 185.9G | 11.79ms | 55.2% |

## 2. YOLOv12模型配置文件详解

YOLOv12的模型结构和参数由YAML格式的配置文件定义。这些配置文件是构建和修改模型的基础，提供了灵活的模型定制能力。

### 2.1 配置文件基本结构

YOLOv12的核心配置文件位于`ultralytics/cfg/models/v12/yolov12.yaml`，包含以下主要部分：

1. **参数部分**：定义模型的基本参数
2. **缩放配置**：定义不同规模模型的缩放比例
3. **主干网络**：定义特征提取部分
4. **检测头**：定义特征融合和目标检测部分

配置文件示例：

```yaml
# YOLOv12配置文件顶部部分
# YOLOv12 🚀, AGPL-3.0 license
# YOLOv12 object detection model with P3-P5 outputs

# 参数部分
nc: 80  # 类别数量
scales:  # 模型复合缩放常数，例如'model=yolov12n.yaml'将使用'n'缩放
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]  # 2.6M参数, 6.2 GFLOPs
  s: [0.50, 0.50, 1024]  # 9.1M参数, 19.7 GFLOPs
  m: [0.50, 1.00, 512]   # 19.7M参数, 60.4 GFLOPs
  l: [1.00, 1.00, 512]   # 26.5M参数, 83.3 GFLOPs
  x: [1.00, 1.50, 512]   # 59.4M参数, 185.9 GFLOPs
```

### 2.2 参数详解

#### 2.2.1 基本参数

- **nc**：检测类别数量，默认为80（COCO数据集类别数）。
- **scales**：定义不同规模模型的缩放参数，有三项：
  - 第一项：深度缩放比例（影响模块重复次数）
  - 第二项：宽度缩放比例（影响通道数）
  - 第三项：最大通道数限制

#### 2.2.2 主干网络配置

主干网络定义了模型如何从输入图像提取特征：

```yaml
# YOLO12-turbo backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]]        # 0-P1/2 - 第一个卷积层
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]] # 1-P2/4 - 第二个卷积层
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]] # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]       # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]    # 核心区域注意力模块
  - [-1, 1, Conv,  [1024, 3, 2]]      # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]   # 8
```

每一行定义了一个层或模块，格式为：
- **[from, repeats, module, args]**
  - **from**：输入来源。`-1`表示前一层，也可以是具体的层索引。
  - **repeats**：模块重复次数。
  - **module**：模块类型，如`Conv`、`A2C2f`等。
  - **args**：模块参数，根据模块类型不同而变化。

#### 2.2.3 检测头配置

检测头负责特征融合和生成最终检测结果：

```yaml
# YOLO12-turbo head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 上采样
  - [[-1, 6], 1, Concat, [1]]                   # 特征连接
  - [-1, 2, A2C2f, [512, False, -1]]            # 11 - 特征融合

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]            # 14

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]            # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]                 # 20 (P5/32-large)

  - [[14, 17, 20], 1, Detect, [nc]]             # 检测层，使用P3, P4, P5特征
```

特别说明：
- **特征融合**：通过上采样和特征连接（Concat）实现多尺度特征融合。
- **Detect层**：最后一层是检测层，接收多个输入特征层（P3、P4、P5），输出最终的检测框和类别。

### 2.3 核心模块配置详解

#### 2.3.1 A2C2f模块参数

A2C2f是YOLOv12的核心创新模块，配置格式为：
```
[输入层, 重复次数, A2C2f, [通道数, 残差连接, 区域数]]
```

参数解释：
- **通道数**：输出通道数
- **残差连接**：是否使用残差连接（True/False）
- **区域数**：区域注意力中的区域划分数量，影响注意力计算的粒度

例如：`[-1, 4, A2C2f, [512, True, 4]]`表示：
- 输入来自前一层
- 重复4次A2C2f模块
- 输出512通道
- 使用残差连接
- 将特征图划分为4个区域进行注意力计算

#### 2.3.2 C3k2模块参数

C3k2是YOLOv12中使用的改进CSP模块，配置格式为：
```
[输入层, 重复次数, C3k2, [通道数, 残差连接, 通道比例]]
```

例如：`[-1, 2, C3k2, [256, False, 0.25]]`表示：
- 输入来自前一层
- 重复2次C3k2模块
- 输出256通道
- 不使用残差连接
- 内部通道比例为0.25

## 3. 创建YOLOv12模型

### 3.1 从预训练模型加载

使用Ultralytics库可以直接加载预训练的YOLOv12模型：

```python
from ultralytics import YOLO

# 加载预训练的YOLOv12模型
model = YOLO('yolov12n.pt')  # nano版本
# 其他可选规格：yolov12s.pt, yolov12m.pt, yolov12l.pt, yolov12x.pt
```

### 3.2 从配置文件创建模型的详细过程

#### 3.2.1 基本创建步骤

```python
from ultralytics import YOLO

# 从YAML配置文件创建模型
model = YOLO('yolov12.yaml')  # 默认配置
```

这一行代码背后发生了什么：

1. **配置文件解析**：读取YAML文件，解析模型结构和参数
2. **模型实例化**：根据配置创建对应的PyTorch模型
3. **模块构建**：构建backbone和head中定义的各个模块
4. **参数初始化**：初始化模型权重

#### 3.2.2 指定模型规模

可以通过在创建时指定规模参数：

```python
# 创建nano规模的模型
model = YOLO('yolov12n.yaml')

# 或者通过参数指定
model = YOLO('yolov12.yaml', scale='n')
```

内部过程：
1. 读取配置文件中的`scales`部分
2. 根据指定的规模（如'n'）获取对应的缩放参数
3. 对模型深度和宽度进行相应缩放

#### 3.2.3 命令行创建与训练

在命令行中创建和训练模型：

```bash
# 基本命令格式
yolo task=detect mode=train model=yolov12n.yaml data=coco128.yaml epochs=100

# 详细参数示例
yolo task=detect mode=train \
  model=yolov12n.yaml \
  data=coco128.yaml \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  device=0
```

命令行参数说明：
- **task**：任务类型，如detect（目标检测）、segment（分割）等
- **mode**：模式，train（训练）、val（验证）、predict（预测）
- **model**：模型配置文件或预训练权重
- **data**：数据集配置文件
- **epochs**：训练轮数
- **batch**：批次大小
- **imgsz**：输入图像尺寸
- **device**：使用的设备（0表示第一个GPU）

### 3.3 配置文件与模型构建的内部机制

当使用`model = YOLO('yolov12.yaml')`创建模型时，内部执行以下步骤：

1. **YAML解析**：
   ```python
   # 伪代码示例
   cfg = yaml.safe_load(open('yolov12.yaml'))
   ```

2. **模型初始化**：
   ```python
   # 伪代码示例
   model = DetectionModel(cfg)  # 创建检测模型实例
   ```

3. **构建网络层**：
   ```python
   # 伪代码示例
   # 根据配置文件中的backbone和head构建网络
   for layer_cfg in cfg['backbone'] + cfg['head']:
       from_layer = layer_cfg[0]
       repeats = layer_cfg[1]
       module_name = layer_cfg[2]
       args = layer_cfg[3]
       
       # 创建并添加模块
       layer = create_module(module_name, args)
       model.add_module(layer)
   ```

4. **应用缩放**：
   ```python
   # 伪代码示例
   if scale in cfg['scales']:
       depth_scale, width_scale, max_channels = cfg['scales'][scale]
       apply_scaling(model, depth_scale, width_scale, max_channels)
   ```

### 3.4 自定义数据集训练

使用自定义数据集训练YOLOv12模型：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov12n.pt')  # 或从配置文件：'yolov12n.yaml'

# 训练模型
results = model.train(
    data='path/to/your/data.yaml',  # 数据集配置文件
    epochs=100,
    batch=16,
    imgsz=640,
    device=0  # 使用的GPU编号
)
```

数据集配置文件例子 (data.yaml):

```yaml
path: /path/to/dataset  # 数据集根目录
train: images/train     # 训练图像目录
val: images/val         # 验证图像目录

# 类别
nc: 3  # 类别数量
names: ['person', 'car', 'bicycle']  # 类别名称
```

## 4. 修改YOLOv12模型

### 4.1 修改模型架构

YOLOv12模型架构定义在配置文件中（`ultralytics/cfg/models/v12/yolov12.yaml`），可以通过修改此文件来自定义模型架构。

#### 4.1.1 创建自定义配置文件

复制原始配置文件并进行修改：

```python
# 创建自定义配置文件
import shutil
from pathlib import Path

# 复制原始配置文件
original_config = Path('ultralytics/cfg/models/v12/yolov12.yaml')
custom_config = Path('custom_yolov12.yaml')
shutil.copy(original_config, custom_config)

# 使用自定义配置文件
model = YOLO('custom_yolov12.yaml')
```

#### 4.1.2 修改主干网络结构

主干网络（backbone）定义了特征提取部分，可以通过修改层数、通道数、模块类型进行自定义：

```yaml
# 修改主干网络示例
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]] # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]] # 3-P3/8
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]] # 5-P4/16
  # 修改A2C2f的区域数量（第三个参数：4→2）
  - [-1, 4, A2C2f, [512, True, 2]]  # 减少区域数量为2
  - [-1, 1, Conv,  [1024, 3, 2]] # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]] # 8
```

#### 4.1.3 修改检测头结构

检测头（head）处理特征提取后的信息并进行目标检测，可以修改：

```yaml
# 修改检测头示例
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  # 增加A2C2f的重复次数（第二个参数：2→3）
  - [-1, 3, A2C2f, [512, False, -1]] # 增加重复次数

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]] # 14
  
  # 后续保持原样...
```

### 4.2 修改区域注意力参数

YOLOv12的核心创新是A2C2f模块中的区域注意力机制。可以通过修改以下参数进行优化：

#### 4.2.1 修改区域数量

区域数量决定了特征图被分割成多少区域进行注意力计算：

```python
# 在配置文件中修改区域数量
- [-1, 4, A2C2f, [512, True, 4]]  # 原始：4个区域
- [-1, 4, A2C2f, [512, True, 2]]  # 修改：2个区域
```

或者在代码中修改A2C2f模块：

```python
# 自定义A2C2f模块
class CustomA2C2f(nn.Module):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        # 自定义初始化参数
        self.area = 2  # 修改默认区域数为2
        # 其他初始化代码...
```

#### 4.2.2 修改MLP比率

多层感知机(MLP)比率影响注意力模块的表达能力：

```python
# 在配置文件中添加自定义MLP比率参数
- [-1, 4, A2C2f, [512, True, 4, 1.5]]  # 最后一个参数为自定义MLP比率
```

### 4.3 自定义类别数和任务

#### 4.3.1 修改类别数

对于自定义数据集，需要修改检测的类别数：

```yaml
# 在配置文件中修改类别数
nc: 20  # 原始为80，修改为自己的类别数
```

或在代码中指定：

```python
# 在训练时指定类别数
model.train(data='custom_data.yaml', nc=20)
```

#### 4.3.2 修改任务类型

YOLOv12支持多种计算机视觉任务，可以通过修改任务类型来适应不同需求：

```python
# 检测任务（默认）
model = YOLO('yolov12n.pt')

# 分割任务（需要使用相应的预训练模型）
model = YOLO('yolov12n-seg.pt')

# 关键点检测任务
model = YOLO('yolov12n-pose.pt')
```

### 4.4 高级修改

#### 4.4.1 修改注意力机制

如果需要深度自定义注意力机制，可以修改AAttn和ABlock模块：

```python
# 自定义区域注意力模块
class CustomAAttn(nn.Module):
    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        # 自定义区域注意力实现
        # ...
        
# 自定义ABlock模块
class CustomABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()
        # 使用自定义区域注意力
        self.attn = CustomAAttn(dim, num_heads=num_heads, area=area)
        # ...
```

#### 4.4.2 添加新功能模块

可以通过添加新的功能模块来增强YOLOv12的能力：

```python
# 定义新模块
class EnhancedA2C2f(nn.Module):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False):
        super().__init__()
        # 增强的A2C2f实现
        # ...

# 在配置文件中使用新模块
backbone:
  # ...
  - [-1, 4, EnhancedA2C2f, [512, True, 4]]  # 使用增强模块
  # ...
```

## 5. 实践示例

### 5.1 创建轻量化YOLOv12模型

创建一个比YOLOv12n更轻量的模型：

```yaml
# 超轻量YOLOv12-nano配置
nc: 80  # 类别数
scales:
  # [depth, width, max_channels]
  nano: [0.25, 0.125, 1024]  # 极小版本

# 其他配置保持不变...
```

### 5.2 高精度YOLOv12模型

创建一个优化精度的YOLOv12模型：

```yaml
# 高精度YOLOv12配置
backbone:
  # [from, repeats, module, args]
  # ...
  # 增加A2C2f的区域数和重复次数
  - [-1, 6, A2C2f, [512, True, 8]]  # 增加区域数和重复次数
  # ...

head:
  # ...
  # 使用更多重复的A2C2f模块
  - [-1, 4, A2C2f, [512, False, 4]]
  # ...
```

### 5.3 训练自定义数据集的完整示例

```python
from ultralytics import YOLO

# 1. 创建或加载模型
model = YOLO('yolov12n.yaml')  # 从配置创建
# model = YOLO('yolov12n.pt')  # 或加载预训练模型

# 2. 训练模型
results = model.train(
    data='custom_data.yaml',  # 数据集配置
    epochs=300,
    batch=64,
    imgsz=640,
    patience=50,  # 早停参数
    optimizer='AdamW',  # 优化器选择
    lr0=0.001,  # 初始学习率
    lrf=0.01,  # 最终学习率因子
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    device=0,
    project='YOLOv12-custom',
    name='run1'
)

# 3. 评估模型
metrics = model.val()

# 4. 导出模型
model.export(format='onnx')  # 导出为ONNX格式
```

## 6. 添加新模型到YOLOv12项目

YOLOv12项目的模块化架构设计允许研究者和开发者轻松地添加新模型或修改现有模型。本节将详细介绍如何向项目中添加新模型。

### 6.1 模型配置规范

YOLOv12项目中的模型配置遵循一定的规范，确保所有模型能被框架正确识别和加载。

#### 6.1.1 配置文件命名规范

```
ultralytics/cfg/models/[版本号]/[模型名].yaml
```

其中：
- **版本号**：通常为"v12"或其他版本标识，表示模型所属的YOLO版本
- **模型名**：描述性名称，通常包含模型版本和特点，如"yolov12"、"yolov12-p6"等

例如：
- `ultralytics/cfg/models/v12/yolov12.yaml` - 标准YOLOv12模型
- `ultralytics/cfg/models/v12/yolov12-seg.yaml` - 分割任务的YOLOv12模型

#### 6.1.2 配置文件内容结构规范

配置文件必须包含以下几个主要部分：

1. **元信息**：顶部注释，包含许可信息和模型简介
2. **参数部分**：必须定义`nc`（类别数）和`scales`（各规模模型的参数）
3. **backbone部分**：定义特征提取网络
4. **head部分**：定义特征融合和目标检测/分割头

基本模板：

```yaml
# 元信息 - 版权和说明
# YOLOv12-[自定义名称] 🚀, AGPL-3.0 license
# [模型简介和输出说明]

# 参数部分
nc: 80  # 类别数量
scales:  # 缩放系数
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]  # nano模型参数
  s: [0.50, 0.50, 1024]  # small模型参数
  m: [0.50, 1.00, 512]   # medium模型参数
  l: [1.00, 1.00, 512]   # large模型参数
  x: [1.00, 1.50, 512]   # xlarge模型参数

# backbone部分
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 第一层
  # ... 其他层定义 ...

# head部分
head:
  # [from, repeats, module, args]
  # ... 头部层定义 ...
  - [[最终层索引列表], 1, Detect/Segment/Pose, [nc]]  # 输出层
```

### 6.2 添加新模型的步骤

以下是向YOLOv12项目添加新模型的完整流程：

#### 6.2.1 创建新的配置文件

1. 在适当的目录创建新的YAML配置文件：

```bash
# 例如创建一个特殊的YOLOv12变体
touch ultralytics/cfg/models/v12/yolov12-custom.yaml
```

2. 基于规范编写配置文件内容：

```yaml
# YOLOv12-Custom 🚀, AGPL-3.0 license
# 自定义的YOLOv12模型，添加了[描述你的改进点]

# 参数部分
nc: 80
scales:
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  # ... 其他规模定义 ...

# backbone部分 - 这里进行自定义修改
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]
  # ... 你的自定义backbone设计 ...
  - [-1, 4, CustomModule, [512, True, 4]]  # 新增自定义模块

# head部分
head:
  # ... 你的自定义head设计 ...
  - [[输出层索引], 1, Detect, [nc]]
```

#### 6.2.2 实现自定义模块（如需要）

如果你的模型包含新的模块类型，需要在`ultralytics/nn/modules/`目录下实现它：

1. 选择合适的文件，如`block.py`或创建新文件
2. 实现你的自定义模块：

```python
# 在 ultralytics/nn/modules/block.py 中添加

class CustomModule(nn.Module):
    """
    自定义模块说明文档
    """
    def __init__(self, c1, c2, custom_param=1):
        super().__init__()
        # 初始化层和参数
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.custom_param = custom_param
        # ... 其他层定义 ...
        
    def forward(self, x):
        """模块的前向传播"""
        # 实现前向传播逻辑
        return self.conv(x)
```

#### 6.2.3 注册模型到框架

确保框架能够识别和加载你的模型：

1. 可能需要更新`ultralytics/nn/tasks.py`文件，确保你的自定义模块被正确导入：

```python
# 在 ultralytics/nn/tasks.py 中
from ultralytics.nn.modules.block import CustomModule  # 导入自定义模块
```

2. 如果你创建了全新的模型类型（如YOLOv12的一个变种），可能需要在`ultralytics/models/__init__.py`中注册：

```python
# 在 ultralytics/models/__init__.py 中
from ultralytics.models import yolo  # 确保YOLO模型被正确导入
```

#### 6.2.4 测试新模型

完成上述步骤后，测试你的新模型：

```python
from ultralytics import YOLO

# 从你的自定义配置创建模型
model = YOLO('ultralytics/cfg/models/v12/yolov12-custom.yaml')

# 测试训练
results = model.train(data='coco128.yaml', epochs=1)

# 验证模型是否能够正常加载和运行
model.info()  # 打印模型信息
model('path/to/image.jpg')  # 测试推理
```

### 6.3 高级模型定制示例

下面提供几个高级模型定制的具体示例，展示如何添加不同类型的新模型。

#### 6.3.1 添加轻量化YOLOv12变体

这个例子展示如何创建一个更加轻量化的YOLOv12变体，适用于资源受限设备：

```yaml
# ultralytics/cfg/models/v12/yolov12-mobile.yaml

# YOLOv12-Mobile 🚀, AGPL-3.0 license
# 轻量化的YOLOv12模型，针对移动设备优化

nc: 80
scales:
  # 增加更轻量的版本
  tiny: [0.25, 0.125, 512]  # 极小版本
  n: [0.33, 0.25, 768]      # 调整nano版参数

# 轻量化的backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [32, 3, 2]]          # 减少初始通道数
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, C3k2, [128, False, 0.25]]  # 减少模块重复和通道数
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 1, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, A2C2f, [256, True, 2]]     # 减少区域数量和重复次数
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, A2C2f, [512, True, 1]]

# 简化的head
head:
  # ... 轻量化的检测头设计 ...
  - [[最终层索引], 1, Detect, [nc]]
```

#### 6.3.2 添加高精度YOLOv12变体

这个例子展示如何创建一个优化精度的YOLOv12变体，适用于需要高精度的场景：

```yaml
# ultralytics/cfg/models/v12/yolov12-hd.yaml

# YOLOv12-HD 🚀, AGPL-3.0 license
# 高精度YOLOv12模型，优化了检测性能

nc: 80
scales:
  # [depth, width, max_channels]
  # 保持原有缩放比例但增加通道数
  l: [1.00, 1.00, 768]   # 增加large模型的通道上限
  x: [1.00, 1.50, 768]   # 增加xlarge模型的通道上限
  # 添加超大规模版本
  xx: [1.25, 2.00, 768]  # 新增xxlarge版本

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]
  # ... 基本结构与标准版相同 ...
  # 增强注意力模块，增加区域数和通道数
  - [-1, 6, A2C2f, [512, True, 8]]     # 增加A2C2f的复杂度
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 6, A2C2f, [1024, True, 4]]    # 增加重复次数和区域数

# 增强的head
head:
  # ... 优化的检测头设计 ...
  # 可能包含更多的特征融合层或更复杂的注意力模块
  - [[最终层索引], 1, Detect, [nc]]
```

#### 6.3.3 添加多任务YOLOv12模型

这个例子展示如何创建一个支持多任务学习的YOLOv12变体：

```yaml
# ultralytics/cfg/models/v12/yolov12-multi.yaml

# YOLOv12-Multi 🚀, AGPL-3.0 license
# 多任务YOLOv12模型，同时支持检测和分割

nc: 80  # 检测类别数
nm: 32  # 分割掩码通道数（新增参数）
scales:
  # [depth, width, max_channels]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  # ... 其他规模 ...

backbone:
  # 标准backbone结构
  # ... 主干网络定义 ...

head:
  # 共享的特征提取层
  # ... 共享头部设计 ...
  
  # 检测分支
  - [[-1, 特征层1, 特征层2], 1, Detect, [nc]]
  
  # 分割分支（新增）
  - [[-1, 特征层1, 特征层2], 1, Segment, [nc, nm]]
```

### 6.4 模型发布和共享

完成新模型开发并验证其性能后，可以选择以下方式发布和共享：

1. **提交Pull Request到Ultralytics官方仓库**：
   - 确保代码质量和文档完整性
   - 提供模型性能对比数据
   - 遵循项目的代码风格和贡献指南

2. **发布预训练权重**：
   - 使用标准数据集(如COCO)训练模型
   - 导出模型权重：`model.export()`
   - 上传到模型仓库如Hugging Face或自己的服务器

3. **编写使用文档**：
   - 说明模型的创新点和使用场景
   - 提供基本的使用示例
   - 包含性能对比和资源需求

### 6.5 模型配置规范检查清单

在添加新模型时，使用以下检查清单确保你的配置符合项目规范：

- [ ] 配置文件位于正确的目录结构中
- [ ] 配置文件包含完整的文档说明和许可信息
- [ ] 参数部分包含必要的`nc`和`scales`定义
- [ ] backbone和head结构设计合理，层连接正确
- [ ] 自定义模块已正确实现并文档化
- [ ] 模型能够成功加载并进行训练和推理
- [ ] 提供了模型性能基准测试数据（速度、精度等）

按照上述规范和步骤，你可以成功地向YOLOv12项目添加新的模型变体，扩展其功能或优化特定场景下的性能。

## 7. 添加新任务类型到YOLOv12项目

除了修改和创建新模型外，YOLOv12项目还支持添加全新的任务类型。本节将详细说明如何扩展YOLOv12框架以支持新任务，以人脸关键点检测任务为例。

### 7.1 任务类型扩展规范

YOLOv12的任务扩展需要遵循一定的规范，确保新任务能无缝集成到现有框架中。

#### 7.1.1 任务集成的核心组件和正确结构

根据项目规范，添加新任务类型需要实现以下组件，并遵循特定的文件组织结构：

1. **模型定义类**：在`ultralytics/nn/tasks.py`中定义，继承自基础模型类
2. **任务特定组件**：在`ultralytics/models/yolo/[task_name]/`目录中实现以下组件：
   - **训练器**：负责该任务的训练逻辑
   - **验证器**：评估模型在该任务上的性能
   - **预测器**：使用训练好的模型执行推理

这种分离是YOLOv12项目的标准架构，类似于项目中其他任务类型（如检测、分割、关键点检测等）的组织方式。遵循这种结构可以确保新任务与框架的其他部分保持一致性和兼容性。

#### 7.1.2 正确的文件组织结构

```
ultralytics/
├── nn/
│   └── tasks.py        # 所有模型类定义的地方（包括新任务的FacePoseModel）
└── models/
    └── yolo/
        └── [task_name]/  # 如 "face"
            ├── __init__.py       # 导出接口
            ├── train.py          # 训练器实现
            ├── val.py            # 验证器实现
            └── predict.py        # 预测器实现
```

模型类命名应遵循`<TaskName>Model`格式，任务组件应遵循`<TaskName><Component>`格式，例如：
- 模型类：`FacePoseModel`（在`nn/tasks.py`中）
- 训练器：`FacePoseTrainer`
- 验证器：`FacePoseValidator`
- 预测器：`FacePosePredictor`

### 7.2 添加新任务的详细步骤

以添加人脸关键点检测任务为例，详细说明完整流程：

#### 7.2.1 在nn/tasks.py中实现模型类

首先，在`ultralytics/nn/tasks.py`文件中添加新任务的模型类定义：

```python
# 在 ultralytics/nn/tasks.py 中添加

class FacePoseModel(DetectionModel):
    """YOLOv12 face pose model."""

    def __init__(self, cfg="yolov12n-face.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv12 Face Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the FacePoseModel."""
        from ultralytics.utils.loss import v12FacePoseLoss
        return v12FacePoseLoss(self)
```

#### 7.2.2 创建任务目录和文件

创建任务所需的目录和文件：

```bash
# 创建任务目录
mkdir -p ultralytics/models/yolo/face

# 创建必要的Python文件
touch ultralytics/models/yolo/face/__init__.py
touch ultralytics/models/yolo/face/train.py
touch ultralytics/models/yolo/face/val.py
touch ultralytics/models/yolo/face/predict.py
```

#### 7.2.3 实现训练器组件

在`train.py`中实现训练逻辑：

```python
# ultralytics/models/yolo/face/train.py

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.tasks import FacePoseModel  # 从nn.tasks导入模型类
from ultralytics.utils import LOGGER, RANK, DEFAULT_CFG

class FacePoseTrainer(PoseTrainer):
    """人脸关键点检测训练器，继承自PoseTrainer"""
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """初始化FacePoseTrainer"""
        if overrides is None:
            overrides = {}
        overrides["task"] = "face"  # 设置任务类型为人脸检测
        super().__init__(cfg, overrides, _callbacks)
    
    def preprocess_batch(self, batch):
        """
        预处理批次数据
        
        参数:
            batch: 输入批次数据
            
        返回:
            处理后的批次数据
        """
        batch = super().preprocess_batch(batch)
        # 添加人脸检测特定的数据预处理（如需要）
        return batch
        
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        获取模型
        
        参数:
            cfg: 模型配置
            weights: 预训练权重
            verbose: 是否打印详细信息
            
        返回:
            初始化的模型
        """
        # 直接从nn.tasks导入FacePoseModel，而不是从本地model.py导入
        model = FacePoseModel(cfg, verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
```

#### 7.2.4 实现验证器组件

在`val.py`中实现验证逻辑：

```python
# ultralytics/models/yolo/face/val.py

from ultralytics.models.yolo.pose.val import PoseValidator

class FacePoseValidator(PoseValidator):
    """人脸关键点检测验证器，继承自PoseValidator"""
    
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """初始化FacePoseValidator"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.is_face = True  # 指示这是一个面部关键点验证器
        # 设置特定于人脸关键点的度量标准或后处理
        
    def postprocess(self, preds):
        """
        后处理预测结果
        
        参数:
            preds: 原始预测结果
            
        返回:
            处理后的预测结果
        """
        preds = super().postprocess(preds)
        # 添加特定于人脸关键点的后处理
        return preds
        
    def metrics(self, preds, batch):
        """
        计算人脸关键点的评估指标
        
        参数:
            preds: 预测结果
            batch: 批次数据
            
        返回:
            计算的度量值
        """
        # 计算特定于人脸关键点的度量标准（如PCK）
        return super().metrics(preds, batch)
```

#### 7.2.5 实现预测器组件

在`predict.py`中实现推理逻辑：

```python
# ultralytics/models/yolo/face/predict.py

from ultralytics.models.yolo.pose.predict import PosePredictor

class FacePosePredictor(PosePredictor):
    """人脸关键点检测预测器，继承自PosePredictor"""
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """初始化FacePosePredictor"""
        super().__init__(cfg, overrides, _callbacks)
        self.is_face = True  # 指示这是一个面部关键点预测器
        # 设置特定于人脸关键点的参数
        
    def postprocess(self, preds, img, orig_imgs):
        """
        后处理预测结果
        
        参数:
            preds: 原始预测结果
            img: 处理后的输入图像
            orig_imgs: 原始输入图像
            
        返回:
            处理后的预测结果
        """
        results = super().postprocess(preds, img, orig_imgs)
        # 添加特定于人脸关键点的后处理
        # 例如，可以添加面部特定的可视化或面部特征提取
        return results
```

#### 7.2.6 初始化模块和导出接口

在`__init__.py`中导出相关类，但注意**不要**导出模型类（因为它已在nn/tasks.py中定义）：

```python
# ultralytics/models/yolo/face/__init__.py

# 注意：FacePoseModel已在nn.tasks.py中定义，不需要在此导入
from .train import FacePoseTrainer
from .val import FacePoseValidator
from .predict import FacePosePredictor

__all__ = ['FacePoseTrainer', 'FacePoseValidator', 'FacePosePredictor']
```

#### 7.2.7 在任务映射中注册新任务

在`ultralytics/models/yolo/model.py`的`YOLO`类中更新`task_map`方法，确保从正确的位置导入模型类：

```python
# 首先更新导入语句
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, FacePoseModel, OBBModel, PoseModel, SegmentationModel, WorldModel

@property
def task_map(self):
    """映射头部到模型、训练器、验证器和预测器类。"""
    return {
        # 现有任务...
        "classify": {
            "model": ClassificationModel,
            "trainer": yolo.classify.ClassificationTrainer,
            "validator": yolo.classify.ClassificationValidator,
            "predictor": yolo.classify.ClassificationPredictor,
        },
        # ... 其他现有任务 ...
        
        # 添加新的人脸关键点检测任务
        "face": {
            "model": FacePoseModel,  # 注意：从nn.tasks直接导入
            "trainer": yolo.face.FacePoseTrainer,
            "validator": yolo.face.FacePoseValidator,
            "predictor": yolo.face.FacePosePredictor,
        },
    }
```

### 7.3 创建任务特定的配置文件

为新任务创建特定的YAML配置文件：

```yaml
# ultralytics/cfg/models/v12/yolov12-face.yaml

# YOLOv12-face 🚀, AGPL-3.0 license
# YOLOv12 面部关键点检测模型

# 参数
nc: 1  # 只有一个类别 - 人脸
kpt_shape: [5, 3]  # 5个关键点, 每个点有x,y坐标和可见性
scales:
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# backbone部分 - 与标准YOLOv12相同或根据需要修改
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  # ... 其他层 ...

# head部分 - 修改以适应面部关键点检测
head:
  # ... 特征提取和融合层 ...
  
  # 最后一层使用Pose而不是Detect
  - [[14, 17, 20], 1, Pose, [nc, kpt_shape]]  # Pose检测层，使用nc和kpt_shape参数
```

### 7.4 准备人脸关键点数据集配置

创建适用于人脸关键点检测的数据集配置文件：

```yaml
# 示例：ultralytics/cfg/datasets/wflw.yaml

# WFLW数据集配置（人脸关键点检测）
# 路径和结构
path: ../datasets/wflw  # 数据集根目录
train: images/train     # 训练图像相对路径
val: images/val         # 验证图像相对路径
test: images/test       # 测试图像相对路径 (可选)

# 类别信息
nc: 1                   # 类别数量，这里只有人脸一个类
names: ['face']         # 类别名称

# 关键点信息
kpt_shape: [5, 3]       # 5个关键点，每个有3个值(x,y,visible)
# 关键点名称 (通常是面部的5个关键点)
keypoints_names: ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']

# 数据增强参数
degrees: 45             # 旋转角度范围
translate: 0.1          # 平移范围
scale: 0.5              # 缩放范围
shear: 0.0              # 剪切范围
perspective: 0.0        # 透视变换范围
flipud: 0.0             # 上下翻转概率 (通常面部不翻转)
fliplr: 0.5             # 左右翻转概率
mosaic: 1.0             # 马赛克增强概率
mixup: 0.0              # 混合增强概率
```

### 7.5 测试和使用新任务

完成上述步骤后，可以测试和使用新添加的人脸关键点检测任务：

```python
from ultralytics import YOLO

# 创建或加载人脸关键点检测模型
model = YOLO('yolov12n-face.yaml')  # 从配置创建
# 或者加载预训练模型
# model = YOLO('yolov12n-face.pt')

# 训练模型
results = model.train(
    data='wflw.yaml',  # 人脸关键点数据集
    epochs=100,
    imgsz=640,
    task='face',  # 指定任务类型为'face'
    batch=16
)

# 验证模型
metrics = model.val()

# 使用模型进行预测
results = model('path/to/image.jpg')
```

### 7.6 任务添加检查清单

在添加新任务时，使用以下检查清单确保所有必要步骤都已完成：

- [ ] 在`nn/tasks.py`中定义模型类
- [ ] 创建任务目录和基本文件结构（训练器、验证器、预测器）
- [ ] 在模型映射中正确注册新任务
- [ ] 为任务创建特定的配置文件
- [ ] 准备适用于该任务的数据集配置
- [ ] 实现任务特定的预处理、后处理和评估指标
- [ ] 测试训练、验证和推理功能
- [ ] 编写使用文档和示例

完成这些步骤后，你就成功地向YOLOv12项目添加了一个全新的任务类型，扩展了框架的功能范围。

## 8. 总结

YOLOv12作为注意力驱动的目标检测模型，提供了强大的性能和灵活的架构。通过本指南中的方法，可以根据具体需求创建和修改YOLOv12模型，实现定制化的目标检测解决方案。无论是提高精度、减小模型体积还是适应特定任务，YOLOv12都提供了丰富的修改空间和可能性。 

## 9. YOLOv12模型加载机制和任务注册流程

要深入理解YOLOv12的工作原理和如何正确添加新任务，了解其模型加载机制和任务注册流程是必不可少的。本章将详细说明YOLOv12的架构设计和任务处理流程。

### 9.1 模型加载机制

YOLOv12采用了模块化设计，支持多种加载方式，主要包括从配置文件创建新模型和从预训练权重加载模型两种方式。

#### 9.1.1 整体架构与继承关系

```
Model (基类, ultralytics/engine/model.py)
  └── YOLO (主要实现, ultralytics/models/yolo/model.py)
        └── Task-specific Models (任务特定模型实现)
```

- **Model类**：框架的核心基类，定义了所有模型共有的接口和功能
- **YOLO类**：继承自Model类，实现YOLO系列模型的特定功能
- **任务特定模型**：如DetectionModel、SegmentationModel等，在nn/tasks.py中定义

#### 9.1.2 从配置文件创建模型流程

当使用`model = YOLO('yolov12.yaml')`创建模型时，内部执行以下步骤：

1. **解析配置文件**：
   ```python
   # Model._new方法调用yaml_model_load加载配置
   cfg_dict = yaml_model_load(cfg)
   ```

2. **推断任务类型**：
   ```python
   # 从配置推断任务类型
   self.task = task or guess_model_task(cfg_dict)
   ```

3. **动态加载模型类**：
   ```python
   # 通过_smart_load和task_map查找对应的模型类
   self.model = self._smart_load("model")(cfg_dict, verbose=verbose)
   ```

4. **初始化模型**：
   - 模型类(如DetectionModel)接收配置字典
   - 创建网络结构并初始化权重

#### 9.1.3 从预训练权重加载模型流程

当使用`model = YOLO('yolov12n.pt')`加载预训练模型时，内部执行以下步骤：

1. **加载权重文件**：
   ```python
   # Model._load方法调用attempt_load_one_weight加载权重
   self.model, self.ckpt = attempt_load_one_weight(weights)
   ```

2. **获取任务信息**：
   ```python
   # 从模型参数中获取任务信息
   self.task = self.model.args["task"]
   ```

3. **设置模型参数**：
   ```python
   # 恢复模型参数和覆盖设置
   self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
   ```

#### 9.1.4 任务映射机制

YOLOv12使用`task_map`属性将任务名称映射到相应的组件类：

```python
@property
def task_map(self):
    return {
        "detect": {
            "model": DetectionModel,
            "trainer": yolo.detect.DetectionTrainer,
            "validator": yolo.detect.DetectionValidator,
            "predictor": yolo.detect.DetectionPredictor,
        },
        # 其他任务...
    }
```

通过`_smart_load`方法，框架可以根据当前任务动态加载对应的组件：

```python
def _smart_load(self, key):
    try:
        return self.task_map[self.task][key]
    except Exception as e:
        # 处理错误...
```

### 9.2 任务注册流程

要向YOLOv12添加新任务，需要完成以下步骤，确保正确遵循框架的架构规范。

#### 9.2.1 任务注册完整流程

1. **在nn/tasks.py中定义模型类**
   新任务的核心模型类必须在`ultralytics/nn/tasks.py`中定义，而不是放在任务目录中：

   ```python
   # 在ultralytics/nn/tasks.py中添加
   class NewTaskModel(DetectionModel):  # 通常继承自DetectionModel或其他基础模型
       def __init__(self, cfg='yolov12n-newtask.yaml', ch=3, nc=None, verbose=True):
           # 初始化代码...
           super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
           
       def init_criterion(self):
           # 定义损失函数
           return NewTaskLoss(self)
   ```

2. **创建任务目录结构**
   按照规范在`ultralytics/models/yolo/`下创建新任务目录，但不包含模型定义：

   ```
   ultralytics/models/yolo/newtask/
   ├── __init__.py        # 导出接口
   ├── train.py           # 训练器
   ├── val.py             # 验证器
   └── predict.py         # 预测器
   ```

3. **实现任务特定组件**
   在任务目录中实现三个主要组件，并确保从正确位置导入模型类：

   ```python
   # ultralytics/models/yolo/newtask/train.py
   from ultralytics.nn.tasks import NewTaskModel  # 从nn.tasks导入模型类
   
   class NewTaskTrainer(BaseTrainer):
       # 训练器实现...
       
       def get_model(self, cfg=None, weights=None, verbose=True):
           # 使用正确导入的模型类
           model = NewTaskModel(cfg, verbose=verbose)
           if weights:
               model.load(weights)
           return model
   ```

4. **导出任务组件接口**
   在`__init__.py`中导出任务组件，但不导出模型类（因为它在nn/tasks.py中）：

   ```python
   # ultralytics/models/yolo/newtask/__init__.py
   from .train import NewTaskTrainer
   from .val import NewTaskValidator
   from .predict import NewTaskPredictor
   
   __all__ = ['NewTaskTrainer', 'NewTaskValidator', 'NewTaskPredictor']
   ```

5. **更新任务映射**
   在`ultralytics/models/yolo/model.py`的`YOLO.task_map`方法中添加新任务：

   ```python
   # 首先更新导入
   from ultralytics.nn.tasks import ClassificationModel, DetectionModel, NewTaskModel
   
   @property
   def task_map(self):
       return {
           # 任务映射...
           "newtask": {
               "model": NewTaskModel,  # 直接从nn.tasks导入
               "trainer": yolo.newtask.NewTaskTrainer,
               "validator": yolo.newtask.NewTaskValidator,
               "predictor": yolo.newtask.NewTaskPredictor,
           },
       }
   ```

6. **更新任务注册表**
   在`ultralytics/cfg/__init__.py`中更新任务注册表：

   ```python
   # 添加到TASKS集合
   TASKS = {"detect", "segment", "classify", "pose", "obb", "newtask"}
   
   # 添加任务对应的默认数据集和模型
   TASK2DATA["newtask"] = "newtask-dataset.yaml"
   TASK2MODEL["newtask"] = "yolov12n-newtask.pt"
   TASK2METRIC["newtask"] = "metrics/newtask_metric"
   ```

7. **创建任务特定配置文件**
   在`ultralytics/cfg/models/v12/`中创建新任务的模型配置文件：

   ```yaml
   # ultralytics/cfg/models/v12/yolov12-newtask.yaml
   # 配置文件内容...
   ```

#### 9.2.2 动态加载原理与实现细节

YOLOv12的动态加载机制依赖于以下关键概念：

1. **属性映射**：`task_map`属性将任务名映射到对应的组件类
2. **惰性加载**：组件只在需要时才被加载，减少内存占用
3. **类型推断**：框架能够从配置文件或权重文件推断任务类型
4. **组件分离**：模型定义与实现组件分离，保持架构清晰

当执行某个操作（如训练或预测）时，框架会：
1. 获取当前任务名称
2. 通过`task_map`查找对应的组件类
3. 实例化需要的组件并执行操作

#### 9.2.3 正确的导入与注册模式

添加新任务时，应遵循以下导入与注册模式：

```python
# 在ultralytics/nn/tasks.py中
class NewTaskModel(DetectionModel):
    # 模型定义...
    
# 在ultralytics/models/yolo/model.py中
from ultralytics.nn.tasks import NewTaskModel

@property
def task_map(self):
    return {
        # 任务映射...
        "newtask": {
            "model": NewTaskModel,  # 直接从nn.tasks导入
            "trainer": yolo.newtask.NewTaskTrainer,
            "validator": yolo.newtask.NewTaskValidator,
            "predictor": yolo.newtask.NewTaskPredictor,
        },
    }
```

### 9.3 任务组件执行流程

当执行某个操作时，YOLOv12会按照以下流程加载和使用组件：

#### 9.3.1 训练流程

```python
model = YOLO('yolov12.yaml')
model.train(data='dataset.yaml')
```

1. 框架通过`task_map`查找对应的`trainer`类
2. 实例化训练器，并传入参数：`trainer = self._smart_load("trainer")(overrides=args)`
3. 训练器获取模型：`self.trainer.model = self.trainer.get_model()`
4. 执行训练流程：`self.trainer.train()`

#### 9.3.2 预测流程

```python
model = YOLO('yolov12.pt')
results = model.predict('image.jpg')
```

1. 框架通过`task_map`查找对应的`predictor`类
2. 实例化预测器：`predictor = self._smart_load("predictor")(overrides=args)`
3. 执行预测流程：`return predictor(source, stream, **kwargs)`

#### 9.3.3 验证流程

```python
model = YOLO('yolov12.pt')
metrics = model.val(data='valid.yaml')
```

1. 框架通过`task_map`查找对应的`validator`类
2. 实例化验证器：`validator = self._smart_load("validator")(args=args)`
3. 执行验证流程：`return validator(model=self.model)`

### 9.4 添加新任务的最佳实践

在向YOLOv12添加新任务时，建议遵循以下最佳实践：

1. **遵循架构分离原则**：模型定义在nn/tasks.py，组件实现在models/yolo/任务目录
2. **复用现有组件**：尽可能继承现有组件，只修改必要的部分
3. **保持命名一致性**：使用`<TaskName>Model`、`<TaskName>Trainer`等命名模式
4. **确保导入正确**：从正确的位置导入类，避免循环导入
5. **添加完整文档**：为新添加的类和方法提供详细文档
6. **编写单元测试**：确保新任务功能正常工作
7. **维护向后兼容性**：确保不破坏现有功能

遵循上述模型加载机制和任务注册流程，可以正确地向YOLOv12项目添加新的任务类型，扩展框架的功能范围。 

## 10. 自定义模块与模块注册

YOLOv12的模块化架构允许研究者和开发者自由地定义新模块并将其整合到模型中。本章将详细介绍如何创建自定义模块、将其注册到YOLOv12框架并在模型中使用。

### 10.1 创建自定义模块的基本步骤

创建并注册自定义模块到YOLOv12框架包含以下核心步骤：

1. **设计并实现模块类**：创建继承自`nn.Module`的模块类
2. **将模块添加到适当的模块文件**：通常位于`ultralytics/nn/modules/`目录下
3. **导出并注册模块**：确保模块被正确导入和注册
4. **在配置文件中使用模块**：在YAML文件中使用自定义模块名称

### 10.2 自定义模块类的实现

#### 10.2.1 标准模块实现规范

YOLOv12中的模块通常遵循以下实现规范：

```python
# 模块实现示例 - 在ultralytics/nn/modules/block.py或自定义文件中

import torch
import torch.nn as nn

class CustomModule(nn.Module):
    """
    自定义模块实现。
    参数说明:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        custom_param (int): 自定义参数，默认值为1
    """
    def __init__(self, c1, c2, custom_param=1):
        super().__init__()
        # 保存参数
        self.custom_param = custom_param
        
        # 创建层
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        
        # 可选：自定义初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        """模块的前向传播"""
        return self.act(self.bn(self.conv(x)))
```

#### 10.2.2 参数命名约定

为了与YOLOv12现有模块保持一致，建议在自定义模块中使用以下参数命名约定：

- `c1`: 输入通道数
- `c2`: 输出通道数
- `k`: 卷积核大小
- `s`: 步长
- `p`: 填充
- `g`: 分组
- `e`: 扩展/压缩系数

遵循这些命名约定可以使模块更容易集成到YAML配置中。

### 10.3 模块注册与导入

#### 10.3.1 模块注册机制

YOLOv12使用显式导入机制注册模块。要将自定义模块正确注册到框架，需要完成以下步骤：

1. **将模块添加到适当的模块文件中**：
   
   ```python
   # 例如在ultralytics/nn/modules/block.py添加
   class CustomModule(nn.Module):
       # 模块实现...
   ```

2. **从模块文件导出**：在模块文件的`__all__`列表中添加模块名称
   
   ```python
   # 在模块文件末尾更新
   __all__ = [..., "CustomModule"]  # 添加自定义模块到导出列表
   ```

3. **更新主模块导入**：在`ultralytics/nn/modules/__init__.py`中导入
   
   ```python
   # 在ultralytics/nn/modules/__init__.py中
   from .block import (..., CustomModule)
   
   __all__ = [..., "CustomModule"]  # 添加到整体导出列表
   ```

4. **确保在tasks.py中可用**：在`ultralytics/nn/tasks.py`的导入语句中添加模块

   ```python
   # 在ultralytics/nn/tasks.py顶部添加到导入列表
   from ultralytics.nn.modules import (
       ...,
       CustomModule,
       ...
   )
   ```

5. **在parse_model函数中注册模块**：更新`parse_model`函数中需要处理参数的模块列表

   ```python
   # 在ultralytics/nn/tasks.py的parse_model函数中
   if m in {
       Conv,
       ...,
       CustomModule,  # 添加自定义模块到列表
       ...
   }:
       c1, c2 = ch[f], args[0]
       # ...处理参数逻辑
   ```

#### 10.3.2 模块解析机制

深入理解YOLOv12的`parse_model`函数对于正确注册模块非常重要。以下是该函数的关键步骤：

1. 函数从配置文件中读取模型结构
2. 对于每个定义的层，查找对应的模块类
3. 处理模块参数并创建模块实例
4. 构建层之间的连接关系
5. 返回完整的模型

函数中有几个关键部分需要特别关注：

```python
# 1. 模块类加载 - 从字符串获取模块类
m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module

# 2. 参数处理 - 根据模块类型处理特定参数
if m in {Conv, ..., CustomModule}:
    c1, c2 = ch[f], args[0]
    if c2 != nc:
        c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2, *args[1:]]
```

当YOLOv12解析YAML配置文件并构建模型时，框架会尝试查找与YAML中指定的模块名匹配的Python类。

### 10.4 在YAML配置文件中使用自定义模块

在YAML配置文件中使用自定义模块的格式与内置模块相同：

```yaml
# 在backbone或head部分使用自定义模块
backbone:
  # [从哪层, 重复次数, 模块名, 参数列表]
  - [-1, 1, CustomModule, [256, 1]]  # 输入通道自动计算, 输出256通道, custom_param=1
  - [-1, 3, CustomModule, [512, 2]]  # 重复3次, 输出512通道, custom_param=2
```

在YAML中使用自定义模块时的参数列表对应模块`__init__`方法的参数，但注意第一个参数通常是输入通道数`c1`，这会由框架自动计算并传入，因此不需要在YAML中指定。

### 10.5 模块注册示例

下面提供一个完整的自定义模块示例，演示整个流程：

#### 10.5.1 创建自定义注意力模块

假设我们要创建一个名为`CustomAttention`的新注意力模块：

```python
# 在ultralytics/nn/modules/attn.py文件中添加

import torch
import torch.nn as nn

class CustomAttention(nn.Module):
    """
    自定义注意力模块，结合空间和通道注意力。
    
    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        ratio (float): 压缩比例，默认为0.5
    """
    def __init__(self, c1, c2, ratio=0.5):
        super().__init__()
        self.c = c2
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        mid_channels = int(c1 * ratio)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, mid_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, c1, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(c1, c2, kernel_size=1)
        
    def forward(self, x):
        # 空间注意力
        spatial_attn = self.spatial_gate(x)
        # 通道注意力
        channel_attn = self.channel_gate(x)
        # 组合注意力
        attn = x * spatial_attn * channel_attn
        # 投影到输出通道
        return self.proj(attn)
```

#### 10.5.2 更新模块导出

更新`attn.py`文件中的导出列表：

```python
# 在attn.py末尾
__all__ = [..., "CustomAttention"]
```

更新`ultralytics/nn/modules/__init__.py`文件：

```python
# 在__init__.py中
from .attn import (..., CustomAttention)

__all__ = [..., "CustomAttention"]
```

#### 10.5.3 在tasks.py中注册模块

更新`ultralytics/nn/tasks.py`文件：

```python
# 顶部导入部分
from ultralytics.nn.modules import (
    ...,
    CustomAttention,
    ...
)

# 在parse_model函数中
if m in {
    Conv,
    ...,
    CustomAttention,  # 添加自定义模块
    ...
}:
    c1, c2 = ch[f], args[0]
    if c2 != nc:
        c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2, *args[1:]]
```

#### 10.5.4 创建使用自定义模块的配置文件

创建一个使用自定义注意力模块的YAML配置文件：

```yaml
# custom_yolov12.yaml
# YOLOv12-Custom 🚀, AGPL-3.0 license
# 带有自定义注意力模块的YOLOv12模型

# 参数部分
nc: 80  # 类别数
scales:
  # [深度缩放, 宽度缩放, 最大通道数]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# 主干网络
backbone:
  # [来源层, 重复次数, 模块, 参数]
  - [-1, 1, Conv,  [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv,  [128, 3, 2, 1, 2]]  # 1-P2/4
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, Conv,  [256, 3, 2, 1, 4]]  # 3-P3/8
  - [-1, 1, CustomAttention, [256, 0.5]]  # 添加自定义注意力
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, Conv,  [512, 3, 2]]  # 5-P4/16
  - [-1, 1, CustomAttention, [512, 0.5]]  # 添加自定义注意力
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv,  [1024, 3, 2]]  # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]]

# 检测头
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[14, 17, 20], 1, Detect, [nc]]
```

#### 10.5.5 使用自定义模块创建和训练模型

```python
from ultralytics import YOLO

# 创建带有自定义模块的模型
model = YOLO('custom_yolov12.yaml')

# 查看模型结构
model.info()

# 训练模型
model.train(
    data='coco128.yaml',
    epochs=10,
    batch=16,
    imgsz=640
)
```

### 10.6 高级模块注册技巧

#### 10.6.1 动态修改模块参数

有时我们需要根据模型规模动态修改模块参数。YOLOv12中的`parse_model`函数提供了这种能力：

```python
if m is CustomModule:  # 在parse_model中为特定规模定制参数
    legacy = False
    if scale in "lx":  # 仅在L和X规模模型中应用特定参数
        args.append(True)  # 例如添加额外参数
```

#### 10.6.2 使用Pytoch原生模块

YOLOv12还支持直接使用PyTorch原生模块：

```yaml
# 在配置文件中使用PyTorch原生模块
- [-1, 1, nn.Dropout, [0.2]]  # 使用nn.前缀表示PyTorch模块
```

当模块名称以"nn."开头时，框架会从`torch.nn`中加载相应的模块。

#### 10.6.3 参数传递技巧

在YAML配置中，可以创建变量并在后续层中引用：

```yaml
# 定义供后续使用的变量
width_val: 256

backbone:
  # 使用变量
  - [-1, 1, CustomModule, [width_val, 1]]
```

### 10.7 常见问题与解决方案

#### 10.7.1 模块未找到错误

如果遇到`NameError: name 'CustomModule' is not defined`错误，通常是因为：

1. **模块未正确导入**：检查模块是否已在`tasks.py`头部导入
2. **模块未正确导出**：检查模块是否已添加到模块文件和`__init__.py`的`__all__`列表中
3. **模块文件未被识别**：确保模块文件是框架的一部分并且路径正确

解决方法：仔细检查所有导入和导出语句，确保模块名称拼写一致。

#### 10.7.2 参数不匹配错误

如果遇到参数不匹配错误，通常是因为：

1. **参数顺序错误**：YOLOv12默认输入通道`c1`是自动计算的，不要在YAML中提供
2. **参数类型错误**：确保在YAML中提供的参数类型与模块期望的参数类型匹配
3. **参数处理未注册**：确保模块在`parse_model`函数中正确注册

解决方法：仔细检查模块的`__init__`方法和YAML配置中的参数列表。

#### 10.7.3 内存或性能问题

如果自定义模块导致内存或性能问题：

1. **检查前向传播实现**：确保高效实现，避免不必要的内存分配
2. **使用内置优化**：利用PyTorch的`torch.jit.script`、`torch.fx`等优化工具
3. **考虑缩放与规模**：为不同规模的模型定制参数，小模型使用更少的资源

### 10.8 总结

创建自定义模块并将其注册到YOLOv12框架是扩展和定制YOLOv12模型能力的强大方式。通过遵循本章介绍的步骤和最佳实践，研究者和开发者可以实现各种创新模块，针对特定应用场景优化模型性能。

关键记住：
1. 模块必须继承自`nn.Module`并实现标准接口
2. 正确注册模块需要更新多个文件中的导入和导出语句
3. 在`parse_model`函数中注册模块以处理参数
4. 在YAML配置文件中使用模块名称并提供适当的参数

通过创建自定义模块，可以为YOLOv12添加新功能，如特殊的注意力机制、创新的卷积操作或特定任务的处理单元，从而拓展模型的应用范围和性能边界。

## 11. 自定义任务和模型注册

YOLOv12架构支持自定义任务和注册新任务到模型中。本章将详细介绍如何创建自定义任务、构建任务相关代码以及将新任务注册到YOLOv12模型中。

### 11.1 YOLOv12任务架构概述

YOLOv12采用模块化设计，使得添加新任务变得简单。每个任务通常包含以下组件：

1. **任务模型类**：继承自`BaseModel`或其子类（如`DetectionModel`）
2. **训练器类**：继承自`BaseTrainer`
3. **验证器类**：用于验证模型性能
4. **预测器类**：用于模型推理
5. **任务特定的头部**：如Detect, Segment, Pose等

目前YOLOv12支持的核心任务包括：
- 目标检测(detect)
- 实例分割(segment)
- 姿态估计(pose) 
- 分类(classify)
- 有向边界框(obb)
- 多模态理解(world)

### 11.2 自定义新任务的步骤

#### 11.2.1 创建任务目录结构

首先，在`ultralytics/models/yolo/`目录下创建新任务的目录，例如`my_task`：

```
ultralytics/
└── models/
    └── yolo/
        └── my_task/
            ├── __init__.py
            ├── train.py
            ├── val.py
            └── predict.py
```

#### 11.2.2 定义模型类

在`ultralytics/nn/tasks.py`中定义新的模型类，继承适当的基类：

```python
class MyTaskModel(DetectionModel):
    """自定义任务模型"""
    
    def __init__(self, cfg="yolov12n-mytask.yaml", ch=3, nc=None, verbose=True):
        """
        初始化MyTask模型
        
        Args:
            cfg (str): 配置文件路径
            ch (int): 输入通道数
            nc (int): 类别数
            verbose (bool): 是否打印详细信息
        """
        super().__init__(cfg, ch, nc, verbose)
        
    def init_criterion(self):
        """初始化损失函数"""
        return MyTaskLoss(self)
```

#### 11.2.3 实现训练器类

在`my_task/train.py`中实现训练器：

```python
from ultralytics.engine.trainer import BaseTrainer

class MyTaskTrainer(BaseTrainer):
    """自定义任务训练器"""
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """初始化训练器"""
        super().__init__(cfg, overrides, _callbacks)
        
    def build_dataset(self, img_path, mode="train", batch=None):
        """构建数据集"""
        # 自定义数据集构建逻辑
        
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """获取数据加载器"""
        # 自定义数据加载器逻辑
        
    def preprocess_batch(self, batch):
        """预处理批次数据"""
        # 自定义批次预处理逻辑
        
    def get_model(self, cfg=None, weights=None, verbose=True):
        """获取模型实例"""
        from ultralytics.nn.tasks import MyTaskModel
        model = MyTaskModel(cfg, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model
        
    def get_validator(self):
        """获取验证器"""
        from ultralytics.models.yolo.my_task import MyTaskValidator
        return MyTaskValidator(self.test_loader, save_dir=self.save_dir, args=self.args)
```

#### 11.2.4 实现验证器和预测器

分别在`my_task/val.py`和`my_task/predict.py`中实现验证器和预测器类。

#### 11.2.5 创建任务特定的头部模块

在`ultralytics/nn/modules`中定义任务特定的头部模块：

```python
class MyTaskHead(nn.Module):
    """自定义任务头部"""
    
    def __init__(self, nc=80, anchors=None):
        """初始化头部"""
        super().__init__()
        # 自定义层和参数
        
    def forward(self, x):
        """前向传播"""
        # 自定义前向传播逻辑
```

#### 11.2.6 创建任务配置文件

在`ultralytics/cfg/models/`目录下创建任务配置文件，例如`yolov12n-mytask.yaml`：

```yaml
# YOLOv12 配置
# 自定义任务: MyTask

# 参数
nc: 80  # 类别数

# 骨干网络
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  # ...其他层

# 检测头
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  # ...其他层
  - [[14, 17, 20], 1, MyTaskHead, [nc]]  # 使用自定义头部
```

### 11.3 将自定义任务注册到模型

在以下文件中注册新任务：

#### 11.3.1 注册到`ultralytics/models/yolo/model.py`

修改`YOLO`类的`task_map`属性：

```python
@property
def task_map(self):
    """映射头部到模型、训练器、验证器和预测器类"""
    return {
        # 现有任务
        "detect": {
            "model": DetectionModel,
            "trainer": yolo.detect.DetectionTrainer,
            "validator": yolo.detect.DetectionValidator,
            "predictor": yolo.detect.DetectionPredictor,
        },
        # 添加新任务
        "my_task": {
            "model": MyTaskModel,
            "trainer": yolo.my_task.MyTaskTrainer,
            "validator": yolo.my_task.MyTaskValidator,
            "predictor": yolo.my_task.MyTaskPredictor,
        },
    }
```

#### 11.3.2 更新`ultralytics/models/yolo/__init__.py`

导入新任务模块：

```python
from . import detect, segment, pose, classify, obb, my_task
```

#### 11.3.3 创建`ultralytics/models/yolo/my_task/__init__.py`

导出任务类：

```python
from .predict import MyTaskPredictor
from .train import MyTaskTrainer
from .val import MyTaskValidator

__all__ = "MyTaskPredictor", "MyTaskTrainer", "MyTaskValidator"
```

### 11.4 模型训练和使用示例

完成自定义任务后，可以通过以下方式使用它：

```python
from ultralytics import YOLO

# 创建模型
model = YOLO('yolov12n-mytask.yaml')

# 训练模型
model.train(
    data='custom_dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=640
)

# 验证模型
model.val()

# 预测
results = model.predict('path/to/image.jpg')
```

### 11.5 自定义损失函数

为自定义任务创建损失函数：

```python
from ultralytics.utils.loss import BaseLoss

class MyTaskLoss(BaseLoss):
    """自定义任务损失函数"""
    
    def __init__(self, model):
        """初始化损失函数"""
        super().__init__(model)
        
    def __call__(self, preds, batch):
        """计算损失"""
        # 自定义损失计算逻辑
        return loss_dict
```

### 11.6 常见问题与解决方案

1. **任务不被识别**：确保在`task_map`中正确注册了任务
2. **模型加载失败**：检查模型配置文件和头部模块是否匹配
3. **训练过程出错**：检查数据集格式是否符合自定义任务要求
4. **自定义模块未导入**：确保在相应的`__init__.py`文件中正确导出了模块

通过以上步骤，您可以在YOLOv12中成功自定义和注册新任务，扩展模型的功能。

## 12. 自定义损失函数与注册

YOLOv12框架支持自定义和注册新的损失函数，以满足特定任务的需求。本章详细介绍如何在YOLOv12中创建、注册和使用自定义损失函数。

### 12.1 YOLOv12损失函数架构

YOLOv12的损失函数体系结构主要包含以下几个部分：

1. **基础损失类**：如`BboxLoss`、`DFLoss`和`VarifocalLoss`等，位于`ultralytics/utils/loss.py`中
2. **任务特定损失类**：如`v8DetectionLoss`、`v8SegmentationLoss`和`v8PoseLoss`等，同样位于`loss.py`中
3. **模型中的损失函数初始化**：在`nn/tasks.py`中，每个模型类通过`init_criterion`方法指定其使用的损失函数

YOLOv12中常用的损失函数组件包括：

- **分类损失**：BCE损失、焦点损失(FL)和变焦点损失(VFL)
- **边界框回归损失**：CIoU损失和L1损失
- **分布焦点损失(DFL)**：用于边界框微调
- **关键点损失**：用于姿态估计任务

### 12.2 创建自定义损失函数

#### 12.2.1 基础损失函数组件

创建基础损失函数组件，继承自`nn.Module`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    """
    自定义基础损失函数组件
    """
    def __init__(self, alpha=0.5, beta=1.0):
        """初始化自定义损失函数"""
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, target, weight=None):
        """
        计算损失值
        
        参数:
            pred (torch.Tensor): 预测值
            target (torch.Tensor): 目标值
            weight (torch.Tensor, optional): 样本权重
            
        返回:
            torch.Tensor: 计算的损失值
        """
        # 自定义损失计算逻辑
        loss = F.mse_loss(pred, target, reduction='none')
        
        if weight is not None:
            loss = loss * weight
            
        # 应用自定义参数
        loss = self.alpha * torch.pow(loss, self.beta)
        
        return loss.mean()
```

#### 12.2.2 任务特定损失类

为特定任务创建综合损失函数类：

```python
class CustomDetectionLoss:
    """自定义目标检测任务的损失函数"""
    
    def __init__(self, model):
        """
        初始化损失函数
        
        参数:
            model: 模型实例，用于获取模型参数
        """
        device = next(model.parameters()).device
        h = model.args  # 超参数
        
        self.custom_loss = CustomLoss(alpha=0.5, beta=1.0)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = model.stride
        self.nc = model.nc  # 类别数
        self.device = device
        
        # 可以从模型参数中读取额外配置
        self.custom_param = getattr(h, 'custom_param', 1.0)
        
    def __call__(self, preds, batch):
        """
        计算损失
        
        参数:
            preds: 模型预测输出
            batch: 输入批次数据
            
        返回:
            (torch.Tensor): 总损失
            (torch.Tensor): 损失明细
        """
        # 解析预测结果和目标
        pred_boxes, pred_cls = preds
        targets = batch['bboxes']
        
        # 计算自定义损失
        box_loss = self.custom_loss(pred_boxes, targets)
        cls_loss = self.bce(pred_cls, batch['cls'])
        
        # 组合损失
        loss = box_loss * self.hyp.box + cls_loss * self.hyp.cls
        
        # 返回总损失和明细
        return loss, torch.cat((box_loss, cls_loss)).detach()
```

### 12.3 注册自定义损失函数

要将自定义损失函数注册到YOLOv12框架中，需要完成以下步骤：

#### 12.3.1 在loss.py中添加自定义损失

在`ultralytics/utils/loss.py`中添加自定义损失类：

```python
# 首先添加基础损失组件
class CustomLoss(nn.Module):
    # 实现如上述代码...

# 然后添加任务损失类
class CustomDetectionLoss:
    # 实现如上述代码...
```

#### 12.3.2 在模型类中使用自定义损失

在`nn/tasks.py`中，修改或创建新的模型类，并在其`init_criterion`方法中使用自定义损失函数：

```python
class CustomDetectionModel(DetectionModel):
    """使用自定义损失函数的检测模型"""
    
    def __init__(self, cfg='yolov12n.yaml', ch=3, nc=None, verbose=True):
        """初始化模型"""
        super().__init__(cfg, ch, nc, verbose)
    
    def init_criterion(self):
        """初始化损失函数"""
        from ultralytics.utils.loss import CustomDetectionLoss
        return CustomDetectionLoss(self)
```

#### 12.3.3 在模型注册中添加自定义模型

在`ultralytics/models/yolo/model.py`的`YOLO.task_map`中注册自定义模型：

```python
@property
def task_map(self):
    """映射头部到模型类"""
    return {
        # 现有任务...
        "custom_detect": {
            "model": CustomDetectionModel,
            "trainer": yolo.detect.DetectionTrainer,  # 可以复用标准训练器
            "validator": yolo.detect.DetectionValidator,
            "predictor": yolo.detect.DetectionPredictor,
        },
    }
```

### 12.4 使用自定义损失函数的方法

有几种方式可以在YOLOv12中使用自定义损失函数：

#### 12.4.1 使用自定义模型类

```python
from ultralytics import YOLO

# 加载自定义模型类
model = YOLO('yolov12n.yaml', task='custom_detect')

# 训练模型
results = model.train(data='custom.yaml', epochs=100)
```

#### 12.4.2 修改损失权重

可以直接通过命令行参数调整现有损失函数的权重：

```bash
# 修改box loss和cls loss的权重
yolo train model=yolov12n.pt data=custom.yaml box=7.5 cls=0.5
```

或在Python代码中：

```python
model = YOLO('yolov12n.pt')
model.train(data='custom.yaml', box=7.5, cls=0.5)
```

#### 12.4.3 通过子类化现有损失函数

也可以通过继承并扩展现有损失函数类实现自定义：

```python
from ultralytics.utils.loss import v8DetectionLoss

class EnhancedDetectionLoss(v8DetectionLoss):
    """增强型检测损失"""
    
    def __init__(self, model):
        """初始化"""
        super().__init__(model)
        # 添加额外组件
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        
    def __call__(self, preds, batch):
        """重写损失计算"""
        # 首先计算标准损失
        loss, loss_items = super().__call__(preds, batch)
        
        # 添加额外损失组件
        # ...
        
        return loss, loss_items
```

### 12.5 高级损失函数自定义示例

以下是一些高级损失函数自定义的实际示例：

#### 12.5.1 添加特征蒸馏损失

特征蒸馏损失用于知识蒸馏，将大模型的知识转移到小模型：

```python
class DistillationLoss(nn.Module):
    """特征蒸馏损失"""
    
    def __init__(self, temp=4.0):
        """
        初始化蒸馏损失
        
        参数:
            temp (float): 温度参数
        """
        super().__init__()
        self.temp = temp
        
    def forward(self, student_feat, teacher_feat):
        """
        计算学生和教师特征的蒸馏损失
        
        参数:
            student_feat (torch.Tensor): 学生模型特征
            teacher_feat (torch.Tensor): 教师模型特征
            
        返回:
            torch.Tensor: 蒸馏损失
        """
        # 使用KL散度计算软目标损失
        s_logits = student_feat / self.temp
        t_logits = teacher_feat.detach() / self.temp
        
        loss = F.kl_div(
            F.log_softmax(s_logits, dim=1),
            F.softmax(t_logits, dim=1),
            reduction='batchmean'
        ) * (self.temp ** 2)
        
        return loss

class DistillationDetectionLoss(v8DetectionLoss):
    """包含蒸馏损失的检测损失"""
    
    def __init__(self, model, teacher_model=None):
        """
        初始化
        
        参数:
            model: 学生模型
            teacher_model: 教师模型
        """
        super().__init__(model)
        self.teacher_model = teacher_model
        self.distill_loss = DistillationLoss(temp=4.0)
        
    def __call__(self, preds, batch):
        """计算总损失"""
        # 计算标准检测损失
        det_loss, det_loss_items = super().__call__(preds, batch)
        
        # 如果没有教师模型，只返回检测损失
        if self.teacher_model is None:
            return det_loss, det_loss_items
            
        # 获取教师模型特征
        with torch.no_grad():
            teacher_preds = self.teacher_model(batch['img'])
            
        # 计算蒸馏损失
        # 假设preds包含特征图，可以根据实际情况修改
        distill_loss = self.distill_loss(preds[2], teacher_preds[2])
        
        # 组合损失
        total_loss = det_loss + 0.5 * distill_loss
        
        return total_loss, torch.cat((det_loss_items, distill_loss.unsqueeze(0).detach()))
```

#### 12.5.2 添加正则化损失

为防止过拟合，添加权重正则化损失：

```python
class RegularizedDetectionLoss(v8DetectionLoss):
    """带正则化的检测损失"""
    
    def __init__(self, model, weight_decay=0.0005, orthogonal_reg=False):
        """
        初始化
        
        参数:
            model: 检测模型
            weight_decay: L2正则化系数
            orthogonal_reg: 是否使用正交正则化
        """
        super().__init__(model)
        self.weight_decay = weight_decay
        self.orthogonal_reg = orthogonal_reg
        
    def get_weight_reg(self):
        """计算权重正则化项"""
        l2_reg = 0.0
        ortho_reg = 0.0
        
        # 遍历所有卷积层权重
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                # L2正则化
                l2_reg += torch.norm(m.weight, 2)
                
                # 正交正则化
                if self.orthogonal_reg:
                    w = m.weight.view(m.weight.size(0), -1)
                    wt_w = torch.mm(w, w.t())
                    identity = torch.eye(w.size(0), device=w.device)
                    ortho_reg += torch.norm(wt_w - identity, 1)
        
        return l2_reg * self.weight_decay + (ortho_reg * 0.01 if self.orthogonal_reg else 0)
        
    def __call__(self, preds, batch):
        """计算总损失"""
        # 基础检测损失
        det_loss, det_loss_items = super().__call__(preds, batch)
        
        # 添加正则化损失
        reg_loss = self.get_weight_reg()
        total_loss = det_loss + reg_loss
        
        return total_loss, torch.cat((det_loss_items, reg_loss.unsqueeze(0).detach()))
```

#### 12.5.3 Adding Focal-IoU Loss for Improved Localization

结合焦点机制和IoU损失，提高小目标和困难样本的定位精度：

```python
class FocalIoULoss(nn.Module):
    """结合焦点机制的IoU损失"""
    
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        """
        初始化
        
        参数:
            alpha (float): 焦点损失参数
            gamma (float): 焦点损失参数
            reduction (str): 损失归约方式
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        计算焦点IoU损失
        
        参数:
            pred (torch.Tensor): 预测边界框 [x1, y1, x2, y2]
            target (torch.Tensor): 目标边界框 [x1, y1, x2, y2]
            
        返回:
            torch.Tensor: 损失值
        """
        # 计算IoU
        iou = bbox_iou(pred, target, xywh=False, CIoU=True)
        
        # 转换为损失
        iou_loss = 1.0 - iou
        
        # 应用焦点机制
        focal_weight = self.alpha * torch.pow(iou_loss, self.gamma)
        focal_iou_loss = focal_weight * iou_loss
        
        # 归约
        if self.reduction == "mean":
            return focal_iou_loss.mean()
        elif self.reduction == "sum":
            return focal_iou_loss.sum()
        else:
            return focal_iou_loss

class FocalIoUBboxLoss(BboxLoss):
    """使用焦点IoU的边界框损失"""
    
    def __init__(self, reg_max=16, alpha=0.5, gamma=2.0):
        """初始化"""
        super().__init__(reg_max)
        self.focal_iou = FocalIoULoss(alpha=alpha, gamma=gamma, reduction="none")
        
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU损失"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # 使用焦点IoU损失替代标准IoU损失
        iou_loss = self.focal_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = (iou_loss * weight).sum() / target_scores_sum

        # DFL损失部分保持不变
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
```

### 12.6 实用技巧与最佳实践

以下是在YOLOv12中自定义损失函数的一些实用技巧：

#### 12.6.1 损失函数设计原则

1. **可逐元素计算**：确保损失函数支持逐元素计算，便于应用样本权重
2. **优先使用矢量化操作**：使用PyTorch的矢量化操作避免循环，提高计算效率
3. **保持数值稳定性**：添加小的ε值防止除零错误，使用torch.clamp()限制值域
4. **支持梯度流动**：确保自定义损失支持梯度流动，避免不必要的detach()操作
5. **与现有框架兼容**：保持输入输出接口与YOLOv12的其他损失函数一致

#### 12.6.2 高效实现技巧

1. **自动混合精度**：使用torch.cuda.amp包支持混合精度训练
2. **批处理操作**：尽量进行批处理操作，减少CPU-GPU数据传输
3. **内存优化**：在可能的情况下使用inplace操作减少内存使用
4. **使用autocast**：包装计算密集型操作以优化性能

```python
from ultralytics.utils.torch_utils import autocast

def compute_loss(self, pred, target):
    with autocast(enabled=False):
        # 高精度损失计算
        loss = custom_loss_function(pred.float(), target.float())
    return loss
```

#### 12.6.3 调试损失函数

1. **添加中间输出**：在开发阶段添加中间结果的输出
2. **分析梯度流**：使用torch.autograd.detect_anomaly()检测梯度异常
3. **单元测试**：为损失函数创建简单的单元测试用例
4. **可视化损失变化**：使用TensorBoard或其他工具可视化损失变化

```python
# 调试辅助函数示例
def debug_loss(pred, target, loss):
    print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
    print(f"Pred range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
    print(f"Loss: {loss.item():.4f}")
    
    # 检查是否有NaN或Inf
    if torch.isnan(loss) or torch.isinf(loss):
        print("WARNING: Loss contains NaN or Inf!")
        # 打印具体位置
        if torch.isnan(pred).any():
            print("NaN in predictions:", torch.nonzero(torch.isnan(pred)))
        if torch.isnan(target).any():
            print("NaN in targets:", torch.nonzero(torch.isnan(target)))
```

### 12.7 总结

自定义损失函数是提升YOLOv12模型性能的强大工具，特别是针对特定任务或数据集。通过本章介绍的方法，可以创建、注册和使用自定义损失函数，以满足特定应用需求。

关键要点：
1. 基础损失函数组件应继承自nn.Module并实现forward方法
2. 任务特定损失类应实现__init__和__call__方法
3. 通过修改模型类的init_criterion方法注册损失函数
4. 可以通过继承扩展现有损失函数类
5. 高级自定义包括添加蒸馏损失、正则化损失和改进的定位损失

遵循上述原则和最佳实践，可以开发出高效、稳定且性能优越的自定义损失函数，进一步提升YOLOv12在特定应用场景中的性能。


