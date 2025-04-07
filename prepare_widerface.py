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

# 添加坐标裁剪函数
def clip_value(value, min_val=0.0, max_val=1.0):
    """确保值在[min_val, max_val]范围内"""
    return max(min_val, min(value, max_val))

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
            
            # 确保坐标在[0,1]范围内
            center_x = clip_value(center_x)
            center_y = clip_value(center_y)
            width = clip_value(width)
            height = clip_value(height)
            
            # 检查有效性
            if width <= 0 or height <= 0:
                continue
            
            # 创建标签字符串，起始为类别ID (人脸=0) 和边界框信息
            label_parts = [0, center_x, center_y, width, height]
            
            # 添加关键点信息（如果有）
            has_landmarks = len(bbox) > 4
            landmarks_added = 0
            
            if has_landmarks:
                # 处理关键点，格式为：x1,y1,v1,x2,y2,v2,...,x5,y5,v5
                # 每组关键点占用3个值：x坐标、y坐标和可见性
                for k in range(0, min(15, len(bbox) - 4), 3):
                    kpt_x = bbox[4 + k]
                    kpt_y = bbox[4 + k + 1]
                    vis = bbox[4 + k + 2]
                    
                    # 归一化坐标
                    if kpt_x >= 0 and kpt_y >= 0:  # 有效关键点
                        norm_kpt_x = clip_value(kpt_x / img_width)
                        norm_kpt_y = clip_value(kpt_y / img_height)
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
                    landmarks_added += 1
            
            # 确保有5个关键点，不足的补充为不可见点
            while landmarks_added < 5:
                label_parts.extend([0.0, 0.0, 0.0])
                landmarks_added += 1
            
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
                        
                        # 为每个图片创建标签文件
                        label_path = os.path.join(labels_out_dir, os.path.splitext(os.path.basename(file))[0] + '.txt')
                        # 创建空标签文件即可，YOLOv12可以处理空标签文件
                        with open(label_path, 'w') as f:
                            pass
                            
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
                # 创建空标签文件即可，YOLOv12可以处理空标签文件
                pass
        
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
            
            # 确保坐标在[0,1]范围内
            center_x = clip_value(center_x)
            center_y = clip_value(center_y)
            width = clip_value(width)
            height = clip_value(height)
            
            # 检查有效性
            if width <= 0 or height <= 0:
                continue
            
            # 创建标签字符串，起始为类别ID (人脸=0) 和边界框信息
            label_parts = [0, center_x, center_y, width, height]
            
            # 添加关键点信息（如果有）
            has_landmarks = len(bbox) > 4
            landmarks_added = 0
            
            if has_landmarks:
                # 处理关键点，格式为：x1,y1,v1,x2,y2,v2,...,x5,y5,v5
                # 每组关键点占用3个值：x坐标、y坐标和可见性
                for k in range(0, min(15, len(bbox) - 4), 3):
                    kpt_x = bbox[4 + k]
                    kpt_y = bbox[4 + k + 1]
                    vis = bbox[4 + k + 2]
                    
                    # 归一化坐标
                    if kpt_x >= 0 and kpt_y >= 0:  # 有效关键点
                        norm_kpt_x = clip_value(kpt_x / img_width)
                        norm_kpt_y = clip_value(kpt_y / img_height)
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
                    landmarks_added += 1
            
            # 确保有5个关键点，不足的补充为不可见点
            while landmarks_added < 5:
                label_parts.extend([0.0, 0.0, 0.0])
                landmarks_added += 1
            
            # 将列表转换为空格分隔的字符串
            yolo_labels.append(' '.join(map(str, label_parts)))
        
        # 写入标签文件，如果没有标签则创建空文件
        with open(os.path.join(labels_out_dir, label_name), 'w') as f:
            if yolo_labels:
                f.write('\n'.join(yolo_labels))
    
    print(f"{phase}数据处理完成，保存到 {output_dir}") 