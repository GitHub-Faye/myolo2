import os
import sys
import cv2
import numpy as np
import shutil
import zipfile
import gdown
import argparse
from tqdm import tqdm
import torch
import torch.utils.data as data

class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc  # 预处理器，用于数据增强
        self.imgs_path = []     # 存储图片路径的列表
        self.words = []         # 存储图片标签的列表
        f = open(txt_path, 'r')  # 打开标签文件
        lines = f.readlines()    # 读取所有行
        isFirst = True           # 标记是否是第一个图片
        labels = []              # 当前图片的标签列表
        
        for line in lines:
            line = line.rstrip()  # 去除行尾空白
            if line.startswith('#'):  # 如果行以#开头，表示新图片
                if isFirst is True:
                    isFirst = False  # 第一张图片只更新标记
                else:
                    # 不是第一张图片，保存前一张图片的标签
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]  # 提取图片路径（去掉#和空格）
                
                # 获取所在阶段(train, val, test)
                phase = None
                if 'train' in txt_path:
                    phase = 'train'
                elif 'val' in txt_path:
                    phase = 'val'
                elif 'test' in txt_path:
                    phase = 'test'
                else:
                    phase = 'train'  # 默认使用train
                
                # 直接使用WIDER_*目录下的图片
                image_path = os.path.join(os.path.dirname(os.path.dirname(txt_path)), f'WIDER_{phase}', 'images', path)
                
                # 立即检查图片是否存在
                if not os.path.exists(image_path):
                    print(f"错误: 未找到图片文件: {image_path}")
                    sys.exit(1)  # 图片不存在时立即终止程序
                
                self.imgs_path.append(image_path)
            else:
                try:
                    # 解析标签行，格式如下：
                    # x y w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4 x5 y5 v5 score
                    # 其中(x,y,w,h)为边界框，(x1-5,y1-5)为五个关键点坐标，v1-5为关键点可见性，score为置信度
                    parts = line.split()
                    
                    # 确保至少有边界框信息(x,y,w,h)
                    if len(parts) >= 4:
                        # 边界框
                        bbox = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
                        
                        # 如果有关键点信息(至少需要x1,y1,v1)
                        landmarks = []
                        if len(parts) >= 7:
                            # 读取所有关键点和可见性
                            for i in range(4, min(len(parts)-1, 19), 3):
                                lx = float(parts[i])
                                ly = float(parts[i+1])
                                vis = float(parts[i+2])  # 可见性
                                landmarks.extend([lx, ly, vis])
                            
                            # 如果关键点数量不足5个，补充到5个
                            while len(landmarks) < 15:
                                landmarks.extend([-1.0, -1.0, -1.0])  # 添加不可见关键点
                        
                        # 添加置信度（如果有）
                        confidence = float(parts[-1]) if len(parts) > 4 else 1.0
                        
                        # 构建完整标签
                        full_label = bbox + landmarks
                        if confidence < 1.0:
                            full_label.append(confidence)
                        
                        labels.append(full_label)
                except Exception as e:
                    print(f"警告: 解析标签行失败: {line}, 错误: {str(e)}")
                    continue

        # 添加最后一张图片的标签
        if not isFirst and labels:
            self.words.append(labels)
        
        print(f"加载了{len(self.imgs_path)}张图片，共{sum(len(words) for words in self.words)}个人脸标签")

    def __len__(self):
        return len(self.imgs_path)

def xywh2xxyy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return x1, x2, y1, y2

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def setup_folders(output_dir):
    """创建必要的文件夹结构"""
    # 创建下载目录
    os.makedirs('downloads', exist_ok=True)
    
    # 创建输出数据集目录结构
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            path = os.path.join(output_dir, split, subdir)
            os.makedirs(path, exist_ok=True)
    
    print(f"已创建必要的文件夹结构在: {output_dir}")

def download_file(url, output_path):
    """下载文件，如果文件已存在则跳过"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 如果文件已存在，跳过下载
    if os.path.exists(output_path):
        print(f"文件已存在: {output_path}，跳过下载")
        return False
    
    # 创建进度条
    print(f"正在下载: {url} 到 {output_path}")
    
    try:
        if 'drive.google.com' in url:
            gdown.download(url, output_path, quiet=False)
        else:
            # 使用tqdm创建进度条
            response = gdown.download(url, output_path, quiet=False)
            
        if os.path.exists(output_path):
            print(f"下载成功: {output_path}")
            return True
        else:
            print(f"下载失败: {url}")
            return False
    except Exception as e:
        print(f"下载时出错: {str(e)}")
        return False

def extract_zip(zip_path, extract_to, force=False):
    """解压ZIP文件到指定目录"""
    # 检查文件是否存在
    if not os.path.exists(zip_path):
        print(f"错误: ZIP文件不存在: {zip_path}")
        sys.exit(1)
    
    # 检查是否需要强制解压
    if not force:
        # 获取zip文件内容的名称
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                first_item = zip_ref.namelist()[0].split('/')[0]
                if os.path.exists(os.path.join(extract_to, first_item)):
                    print(f"目标目录已存在: {os.path.join(extract_to, first_item)}，跳过解压")
                    return
        except Exception as e:
            print(f"警告: 检查ZIP内容时出错: {str(e)}")
    
    # 确保目标目录存在
    os.makedirs(extract_to, exist_ok=True)
    
    # 解压文件
    print(f"正在解压: {zip_path} 到 {extract_to}")
    try:
        # 尝试使用zipfile库解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取文件总数用于进度条
            total_files = len(zip_ref.namelist())
            
            # 解压所有文件
            for file in tqdm(zip_ref.namelist(), total=total_files, desc=f"解压 {os.path.basename(zip_path)}"):
                zip_ref.extract(file, extract_to)
        
        print(f"解压完成: {zip_path}")
    except Exception as e:
        print(f"zipfile解压失败: {str(e)}，尝试使用系统命令解压...")
        
        # 如果zipfile解压失败，尝试使用系统命令
        if sys.platform == 'win32':
            # Windows系统
            try:
                import subprocess
                # 使用PowerShell的Expand-Archive命令
                cmd = f'powershell -command "Expand-Archive -Path \'{zip_path}\' -DestinationPath \'{extract_to}\' -Force"'
                result = subprocess.run(cmd, shell=True, check=True)
                print(f"PowerShell解压完成: {zip_path}")
            except Exception as e2:
                print(f"PowerShell解压失败: {str(e2)}")
                sys.exit(1)
        else:
            # Linux/Mac系统
            try:
                import subprocess
                # 使用unzip命令
                cmd = f'unzip -o "{zip_path}" -d "{extract_to}"'
                result = subprocess.run(cmd, shell=True, check=True)
                print(f"unzip命令解压完成: {zip_path}")
            except Exception as e2:
                print(f"unzip命令解压失败: {str(e2)}")
                sys.exit(1)

def check_directory_structure(data_dir):
    """检查并修复数据目录结构"""
    print("检查目录结构...")
    
    # 检查wider_face_split目录
    split_dir = os.path.join(data_dir, 'wider_face_split')
    if not os.path.exists(split_dir):
        print(f"未找到wider_face_split目录，尝试查找解压后的目录...")
        
        # 尝试在data_dir中寻找包含"wider_face_split"的目录
        for root, dirs, files in os.walk(data_dir):
            for d in dirs:
                if 'wider_face_split' in d.lower():
                    src_path = os.path.join(root, d)
                    print(f"找到替代目录: {src_path}，移动到正确位置...")
                    if not os.path.exists(split_dir):
                        os.makedirs(os.path.dirname(split_dir), exist_ok=True)
                        shutil.move(src_path, split_dir)
                    break
    
    # 检查训练、验证和测试数据目录
    for phase in ['train', 'val', 'test']:
        phase_dir = os.path.join(data_dir, f'WIDER_{phase}')
        if not os.path.exists(phase_dir):
            print(f"未找到{phase}数据目录: {phase_dir}，尝试自动查找...")
            
            # 尝试在data_dir中查找可能的目录
            found = False
            
            # 策略1: 直接查找WIDER_{phase}目录
            for root, dirs, files in os.walk(data_dir):
                for d in dirs:
                    if f'WIDER_{phase}' in d:
                        src_path = os.path.join(root, d)
                        print(f"找到替代目录: {src_path}，移动到正确位置...")
                        os.makedirs(os.path.dirname(phase_dir), exist_ok=True)
                        shutil.move(src_path, phase_dir)
                        found = True
                        break
                if found:
                    break
            
            # 策略2: 查找WIDER目录下的{phase}目录
            if not found:
                wider_dir = None
                for root, dirs, files in os.walk(data_dir):
                    for d in dirs:
                        if d.lower() == 'wider':
                            wider_dir = os.path.join(root, d)
                            break
                    if wider_dir:
                        break
                
                if wider_dir:
                    wider_phase_dir = os.path.join(wider_dir, phase)
                    if os.path.exists(wider_phase_dir):
                        print(f"找到替代目录: {wider_phase_dir}，复制到正确位置...")
                        os.makedirs(os.path.dirname(phase_dir), exist_ok=True)
                        shutil.copytree(wider_phase_dir, phase_dir)
                        found = True
            
            # 如果仍然未找到，尝试从原始zip文件中提取
            if not found:
                zip_file = f'downloads/WIDER_{phase}.zip'
                if os.path.exists(zip_file):
                    print(f"尝试从{zip_file}中提取{phase}数据...")
                    # 强制解压特定目录
                    extract_zip(zip_file, data_dir, force=True)
        
        # 检查images子目录
        images_dir = os.path.join(phase_dir, 'images')
        if os.path.exists(phase_dir) and not os.path.exists(images_dir):
            print(f"未找到images子目录: {images_dir}，尝试自动修复...")
            
            # 策略1: 检查是否有直接的图片文件，如果有，创建images目录并移动
            has_images = False
            for item in os.listdir(phase_dir):
                if item.lower().endswith(('.jpg', '.jpeg', '.png')):
                    has_images = True
                    break
            
            if has_images:
                print(f"在{phase_dir}中发现图片文件，创建images目录并移动...")
                os.makedirs(images_dir, exist_ok=True)
                for item in os.listdir(phase_dir):
                    if item.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_file = os.path.join(phase_dir, item)
                        dst_file = os.path.join(images_dir, item)
                        shutil.move(src_file, dst_file)
            else:
                # 策略2: 检查是否有事件子目录(WIDER Face的组织方式)
                event_dirs = []
                for item in os.listdir(phase_dir):
                    item_path = os.path.join(phase_dir, item)
                    if os.path.isdir(item_path) and not item == 'images':
                        event_dirs.append(item)
                
                if event_dirs:
                    print(f"在{phase_dir}中发现{len(event_dirs)}个事件目录，移动到images目录...")
                    os.makedirs(images_dir, exist_ok=True)
                    for event_dir in event_dirs:
                        src_dir = os.path.join(phase_dir, event_dir)
                        dst_dir = os.path.join(images_dir, event_dir)
                        shutil.move(src_dir, dst_dir)
    
    # 最后检查所有必要的目录和文件是否存在
    required_files = [
        os.path.join(data_dir, 'wider_face_split', 'wider_face_train_bbx_gt.txt'),
        os.path.join(data_dir, 'wider_face_split', 'wider_face_val_bbx_gt.txt'),
        os.path.join(data_dir, 'WIDER_train', 'images'),
        os.path.join(data_dir, 'WIDER_val', 'images')
    ]
    
    all_exist = True
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"警告: 仍然缺失必要的文件或目录: {file_path}")
            all_exist = False
    
    if all_exist:
        print("目录结构检查完成，所有必要的文件和目录都已存在")
    else:
        print("目录结构检查完成，但仍有一些文件或目录缺失，处理过程可能会出错")

def process_train_data(data_dir, output_dir):
    """处理训练数据，转换为YOLOv12格式"""
    print(f"处理训练数据...")
    
    # 确保输出目录存在
    images_out_dir = os.path.join(output_dir, 'images')
    labels_out_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)
    
    # 查找训练标签文件
    label_file = None
    
    # 先尝试找到直接的标签文件
    possible_label_file = os.path.join(data_dir, 'label.txt')
    if os.path.exists(possible_label_file):
        label_file = possible_label_file
    
    # 如果没找到，尝试wider_face_split目录
    if label_file is None:
        wider_face_train = os.path.join(os.path.dirname(data_dir), 'wider_face_split', 'wider_face_train_bbx_gt.txt')
        if os.path.exists(wider_face_train):
            label_file = wider_face_train
    
    if label_file is None:
        print(f"未找到训练标签文件!")
        return
    
    print(f"使用标签文件: {label_file}")
    
    # 加载数据集
    dataset = WiderFaceDetection(label_file)
    print(f"找到 {len(dataset)} 张训练图片")
    
    # 处理每张图片和标签
    for i in tqdm(range(len(dataset)), desc="处理训练图片"):
        img_path = dataset.imgs_path[i]
        if not os.path.exists(img_path):
            print(f"错误: 图片不存在 {img_path}")
            sys.exit(1)  # 图片不存在时立即终止程序
        
        # 读取图片获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"错误: 无法读取图片 {img_path}")
            sys.exit(1)  # 无法读取图片时立即终止程序
        
        img_height, img_width = img.shape[:2]
        
        # 构建输出文件名
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        
        # 复制图片到输出目录
        dst_img_path = os.path.join(images_out_dir, img_name)
        shutil.copy(img_path, dst_img_path)
        
        # 转换标签为YOLO格式
        bboxes = dataset.words[i]
        yolo_labels = []
        
        for bbox in bboxes:
            # 基本的边界框坐标(x, y, w, h)
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # 确保边界框在图像范围内
            x = max(0, x)
            y = max(0, y)
            w = min(img_width - x, w)
            h = min(img_height - y, h)
            
            # 标准化为YOLO格式 (center_x, center_y, width, height)
            center_x = (x + w/2) / img_width
            center_y = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height
            
            # 检查有效性
            if width <= 0 or height <= 0:
                continue
            
            # 创建标签字符串，起始为类别ID (人脸=0) 和边界框信息
            label_parts = [0, center_x, center_y, width, height]
            
            # 添加关键点信息（如果有）
            has_landmarks = len(bbox) > 4
            if has_landmarks:
                # 处理关键点，格式为：x1,y1,v1,x2,y2,v2,...,x5,y5,v5
                # 每组关键点占用3个值：x坐标、y坐标和可见性
                for k in range(0, min(15, len(bbox) - 4), 3):
                    kpt_x = bbox[4 + k]
                    kpt_y = bbox[4 + k + 1]
                    vis = bbox[4 + k + 2]
                    
                    # 归一化坐标
                    if kpt_x >= 0 and kpt_y >= 0:  # 有效关键点
                        norm_kpt_x = kpt_x / img_width
                        norm_kpt_y = kpt_y / img_height
                    else:
                        # 关键点不可见
                        norm_kpt_x = 0
                        norm_kpt_y = 0
                    
                    # 确定可见性标志
                    # vis值: 0=不可见, 1=可见但被遮挡, 2=完全可见
                    visibility = 0
                    if vis > 0:
                        visibility = 2  # 假设大于0的值表示可见
                    
                    # 添加到标签
                    label_parts.extend([norm_kpt_x, norm_kpt_y, visibility])
            
            # 将列表转换为空格分隔的字符串
            yolo_labels.append(' '.join(map(str, label_parts)))
        
        # 写入标签文件
        with open(os.path.join(labels_out_dir, label_name), 'w') as f:
            f.write('\n'.join(yolo_labels))
    
    print(f"训练数据处理完成，保存到 {output_dir}")

def process_val_test_data(data_dir, output_dir, phase='val'):
    """处理验证或测试数据，转换为YOLOv12格式"""
    print(f"处理{phase}数据...")
    
    # 确保输出目录存在
    images_out_dir = os.path.join(output_dir, 'images')
    labels_out_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)
    
    # 查找标签文件
    label_file = None
    
    # 先尝试找到直接的标签文件
    possible_label_file = os.path.join(data_dir, phase, 'label.txt')
    if os.path.exists(possible_label_file):
        label_file = possible_label_file
    
    # 如果没找到，尝试wider_face_split目录
    if label_file is None:
        wider_face_label = os.path.join(data_dir, 'wider_face_split', f'wider_face_{phase}_bbx_gt.txt')
        if os.path.exists(wider_face_label):
            label_file = wider_face_label
    
    # 如果是测试集，可能没有标签文件或标签是空的
    if label_file is None or phase == 'test':
        print(f"{phase}集标签文件未找到或不包含标签，将只复制图片")
        # 查找测试集图片目录
        test_images_dir = os.path.join(data_dir, f'WIDER_{phase}', 'images')
        if os.path.exists(test_images_dir):
            # 复制所有图片
            image_count = 0
            for root, dirs, files in os.walk(test_images_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = os.path.join(root, file)
                        if not os.path.exists(src_path):
                            print(f"错误: 图片不存在 {src_path}")
                            sys.exit(1)  # 图片不存在时立即终止程序
                        
                        # 保持目录结构
                        rel_path = os.path.relpath(root, test_images_dir)
                        if rel_path != '.':
                            target_dir = os.path.join(images_out_dir, rel_path)
                            os.makedirs(target_dir, exist_ok=True)
                            dst_path = os.path.join(target_dir, file)
                        else:
                            dst_path = os.path.join(images_out_dir, file)
                            
                        shutil.copy(src_path, dst_path)
                        
                        # 为每个图片创建一个空的标签文件
                        label_path = os.path.join(labels_out_dir, os.path.splitext(os.path.basename(file))[0] + '.txt')
                        with open(label_path, 'w') as f:
                            pass  # 创建空文件
                            
                        image_count += 1
            print(f"已复制 {image_count} 张{phase}图片")
        return
    
    if label_file is None:
        print(f"未找到{phase}标签文件!")
        return
    
    print(f"使用标签文件: {label_file}")
    
    # 加载数据集
    dataset = WiderFaceDetection(label_file)
    print(f"找到 {len(dataset)} 张{phase}图片")
    
    # 检查数据集是否为空
    if len(dataset.words) == 0 or sum(len(words) for words in dataset.words) == 0:
        print(f"{phase}数据集没有标签，将只复制图片")
        # 处理每张图片，但不处理标签
        for i in tqdm(range(len(dataset.imgs_path)), desc=f"处理{phase}图片"):
            img_path = dataset.imgs_path[i]
            if not os.path.exists(img_path):
                print(f"错误: 图片不存在 {img_path}")
                sys.exit(1)
            
            # 复制图片到输出目录
            img_name = os.path.basename(img_path)
            dst_img_path = os.path.join(images_out_dir, img_name)
            shutil.copy(img_path, dst_img_path)
            
            # 创建空标签文件
            label_name = os.path.splitext(img_name)[0] + '.txt'
            with open(os.path.join(labels_out_dir, label_name), 'w') as f:
                pass  # 创建空文件
        
        print(f"{phase}数据处理完成，保存到 {output_dir}")
        return
    
    # 确保标签列表和图片列表长度一致
    if len(dataset.words) != len(dataset.imgs_path):
        print(f"警告: 标签数量({len(dataset.words)})与图片数量({len(dataset.imgs_path)})不一致，将使用较小值")
        length = min(len(dataset.words), len(dataset.imgs_path))
    else:
        length = len(dataset.imgs_path)
    
    # 处理每张图片和标签
    for i in tqdm(range(length), desc=f"处理{phase}图片"):
        img_path = dataset.imgs_path[i]
        if not os.path.exists(img_path):
            print(f"错误: 图片不存在 {img_path}")
            sys.exit(1)  # 图片不存在时立即终止程序
        
        # 读取图片获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"错误: 无法读取图片 {img_path}")
            sys.exit(1)  # 无法读取图片时立即终止程序
        
        img_height, img_width = img.shape[:2]
        
        # 构建输出文件名
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        
        # 复制图片到输出目录
        dst_img_path = os.path.join(images_out_dir, img_name)
        shutil.copy(img_path, dst_img_path)
        
        # 转换标签为YOLO格式
        bboxes = dataset.words[i]
        yolo_labels = []
        
        for bbox in bboxes:
            # 基本的边界框坐标(x, y, w, h)
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # 确保边界框在图像范围内
            x = max(0, x)
            y = max(0, y)
            w = min(img_width - x, w)
            h = min(img_height - y, h)
            
            # 标准化为YOLO格式 (center_x, center_y, width, height)
            center_x = (x + w/2) / img_width
            center_y = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height
            
            # 检查有效性
            if width <= 0 or height <= 0:
                continue
            
            # 创建标签字符串，起始为类别ID (人脸=0) 和边界框信息
            label_parts = [0, center_x, center_y, width, height]
            
            # 添加关键点信息（如果有）
            has_landmarks = len(bbox) > 4
            if has_landmarks:
                # 处理关键点，格式为：x1,y1,v1,x2,y2,v2,...,x5,y5,v5
                # 每组关键点占用3个值：x坐标、y坐标和可见性
                for k in range(0, min(15, len(bbox) - 4), 3):
                    kpt_x = bbox[4 + k]
                    kpt_y = bbox[4 + k + 1]
                    vis = bbox[4 + k + 2]
                    
                    # 归一化坐标
                    if kpt_x >= 0 and kpt_y >= 0:  # 有效关键点
                        norm_kpt_x = kpt_x / img_width
                        norm_kpt_y = kpt_y / img_height
                    else:
                        # 关键点不可见
                        norm_kpt_x = 0
                        norm_kpt_y = 0
                    
                    # 确定可见性标志
                    # vis值: 0=不可见, 1=可见但被遮挡, 2=完全可见
                    visibility = 0
                    if vis > 0:
                        visibility = 2  # 假设大于0的值表示可见
                    
                    # 添加到标签
                    label_parts.extend([norm_kpt_x, norm_kpt_y, visibility])
            
            # 将列表转换为空格分隔的字符串
            yolo_labels.append(' '.join(map(str, label_parts)))
        
        # 写入标签文件，如果没有标签则创建空文件
        with open(os.path.join(labels_out_dir, label_name), 'w') as f:
            if yolo_labels:
                f.write('\n'.join(yolo_labels))
    
    print(f"{phase}数据处理完成，保存到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='准备WIDER Face数据集为YOLOv12格式，包含人脸检测和关键点标注')
    parser.add_argument('--download', action='store_true', help='下载数据集')
    parser.add_argument('--process', action='store_true', help='处理数据集')
    parser.add_argument('--force_download', action='store_true', help='强制重新下载已存在的文件')
    parser.add_argument('--force_extract', action='store_true', help='强制重新解压已存在的文件')
    parser.add_argument('--data_dir', type=str, default='widerface_data', help='数据存储路径')
    parser.add_argument('--output_dir', type=str, default='widerface', help='输出目录')
    args = parser.parse_args()

    if not args.download and not args.process:
        args.download = True
        args.process = True

    setup_folders(args.output_dir)
    
    annotation_url = "https://drive.google.com/uc?id=1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8"
    train_url = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip"
    val_url = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip"
    test_url = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_test.zip"
    
    annotation_file = "downloads/wider_face_split.zip"
    train_file = "downloads/WIDER_train.zip"
    val_file = "downloads/WIDER_val.zip" 
    test_file = "downloads/WIDER_test.zip"
    
    if args.download:
        print("开始下载数据...")
        # 如果强制下载，先删除已存在的文件
        if args.force_download:
            for file_path in [annotation_file, train_file, val_file, test_file]:
                if os.path.exists(file_path):
                    print(f"删除已存在的文件: {file_path}")
                    os.remove(file_path)
            
        # 下载文件
        annotation_downloaded = download_file(annotation_url, annotation_file)
        train_downloaded = download_file(train_url, train_file)
        val_downloaded = download_file(val_url, val_file)
        test_downloaded = download_file(test_url, test_file)
        
        print("解压文件...")
        # 只解压成功下载的文件
        if annotation_downloaded:
            extract_zip(annotation_file, args.data_dir, force=args.force_extract)
        if train_downloaded:
            extract_zip(train_file, args.data_dir, force=args.force_extract)
        if val_downloaded:
            extract_zip(val_file, args.data_dir, force=args.force_extract)
        if test_downloaded:
            extract_zip(test_file, args.data_dir, force=args.force_extract)
        
        # 检查并修复目录结构
        check_directory_structure(args.data_dir)

    if args.process:
        print("开始处理数据...")
        process_train_data(
            os.path.join(args.data_dir, 'train'),
            os.path.join(args.output_dir, 'train')
        )
        
        process_val_test_data(
            args.data_dir,
            os.path.join(args.output_dir, 'val'),
            phase='val'
        )
        
        process_val_test_data(
            args.data_dir,
            os.path.join(args.output_dir, 'test'),
            phase='test'
        )
        
        # 创建数据集配置文件
        with open(os.path.join(args.output_dir, 'data.yaml'), 'w') as f:
            f.write(f"""path: {os.path.abspath(args.output_dir)}
train: train
val: val
test: test

nc: 1
names: ['face']

# 关键点信息
kpt_shape: [5, 3]  # 5个关键点，每个点有3个值 [x,y,visible]
flip_idx: [1, 0, 2, 4, 3]  # 水平翻转时的关键点对应关系：右眼<->左眼，右嘴角<->左嘴角
# 关键点名称：左眼、右眼、鼻子、左嘴角、右嘴角
kpt_names: ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
""")
        
        print(f"处理完成! 数据集已保存到 {args.output_dir}")
        print(f"训练配置文件已创建: {os.path.join(args.output_dir, 'data.yaml')}")
        print(f"数据集包含人脸检测和5个关键点(双眼、鼻子、双嘴角)的标注")
        print(f"使用说明：")
        print(f"1. 训练YOLOv12模型：")
        print(f"   yolo train data={os.path.join(args.output_dir, 'data.yaml')} model=yolov12n-pose.yaml epochs=100")
        print(f"2. 验证模型：")
        print(f"   yolo val data={os.path.join(args.output_dir, 'data.yaml')} model=path/to/weights.pt")
        print(f"3. 测试检测：")
        print(f"   yolo predict model=path/to/weights.pt source=path/to/image.jpg")

if __name__ == '__main__':
    main() 