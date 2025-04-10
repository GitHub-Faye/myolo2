from ultralytics import YOLO
import cv2
import numpy as np

# 创建一个简单的测试图像，包含一个人物轮廓
def create_test_image(width=640, height=480):
    # 创建黑色背景
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # 绘制人形轮廓
    # 头部
    cv2.circle(img, (width//2, height//4), 50, (0, 255, 0), -1)
    # 身体
    cv2.rectangle(img, (width//2-40, height//4+50), (width//2+40, height//4+200), (0, 255, 0), -1)
    # 手臂
    cv2.line(img, (width//2, height//4+80), (width//2-100, height//4+150), (0, 255, 0), 20)
    cv2.line(img, (width//2, height//4+80), (width//2+100, height//4+150), (0, 255, 0), 20)
    # 腿
    cv2.line(img, (width//2-20, height//4+200), (width//2-50, height//4+350), (0, 255, 0), 20)
    cv2.line(img, (width//2+20, height//4+200), (width//2+50, height//4+350), (0, 255, 0), 20)
    
    return img

# 推理
print("加载模型...")
model = YOLO('yolov12n-custompose.yaml', task='custompose')
print(f"模型任务类型: {model.task}")
print(f"模型架构: {type(model.model).__name__}")

# 尝试获取配置中的关键点信息
try:
    yaml_data = model.model.yaml
    if 'kpt_shape' in yaml_data:
        print(f"配置中的关键点形状: {yaml_data['kpt_shape']}")
except Exception as e:
    print(f"无法获取关键点形状: {e}")

# 使用合成图像进行测试
print("创建测试图像...")
test_image = create_test_image()
cv2.imwrite("test_image.jpg", test_image)
print("测试图像已保存为 test_image.jpg")

print("执行推理...")
# 设置低置信度阈值进行测试
results = model(test_image, verbose=True, conf=0.01)[0]

# 检查结果对象
print(f"结果类型: {type(results)}")
print(f"检测到的目标数量: {len(results.boxes)}")

# 如果有检测结果，打印详细信息并可视化
if len(results.boxes) > 0:
    # 打印第一个检测框的信息
    print(f"第一个目标置信度: {results.boxes.conf[0].item():.4f}")
    print(f"第一个目标类别: {results.boxes.cls[0].item()}")
    print(f"第一个目标边界框: {results.boxes.xyxy[0].tolist()}")
    
    # 打印关键点信息(如果有)
    if results.keypoints is not None:
        print(f"关键点形状: {results.keypoints.shape}")
        print(f"第一个目标关键点数量: {results.keypoints.shape[1]}")
        # 打印每个关键点的坐标 - 使用正确的属性访问方式
        for i in range(results.keypoints.shape[1]):
            # 使用.xy属性访问坐标
            kpt_xy = results.keypoints.xy[0, i].tolist()
            # 如果有置信度信息，也打印出来
            if hasattr(results.keypoints, 'conf') and results.keypoints.conf is not None:
                kpt_conf = results.keypoints.conf[0, i].item()
                print(f"关键点 {i}: 坐标={kpt_xy}, 置信度={kpt_conf:.4f}")
            else:
                print(f"关键点 {i}: 坐标={kpt_xy}")
else:
    print("未检测到任何目标")

# 尝试使用现有的dog.jpg图像
print("\n使用dog.jpg图像测试...")
dog_image = cv2.imread("dog.jpg")
if dog_image is not None:
    print(f"图片尺寸: {dog_image.shape}")
    dog_results = model(dog_image, verbose=True, conf=0.01)[0]
    print(f"检测到的目标数量: {len(dog_results.boxes)}")
    
    # 可视化结果
    dog_result_image = dog_results.plot()
    cv2.imwrite("dog_result.jpg", dog_result_image)
    print("结果已保存至 dog_result.jpg")
else:
    print("无法读取dog.jpg图像")

# 可视化结果
result_image = results.plot()  # 绘制结果（包括关键点）
cv2.imwrite("test_result.jpg", result_image)  # 保存结果图像
print("结果已保存至 test_result.jpg")

# 尝试访问不同属性检查数据流
print("\n数据流检查:")
for attr in dir(results):
    if not attr.startswith('_') and not callable(getattr(results, attr)):
        try:
            value = getattr(results, attr)
            print(f"{attr}: {type(value)}")
        except Exception as e:
            print(f"{attr}: 访问错误 - {e}")

