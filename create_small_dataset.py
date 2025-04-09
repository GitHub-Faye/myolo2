"""创建小型WiderFace数据集
从train、val、test各抽取16张图片及其标签
"""
import os
import shutil
import random
from pathlib import Path

def create_small_dataset():
    """创建小型数据集的主函数"""
    # 源数据集和目标数据集路径
    src_root = Path("widerface")
    dst_root = Path("widerface_litel") 
    
    # 确保目标目录存在
    dst_root.mkdir(exist_ok=True)
    
    # 复制data.yaml
    shutil.copy2(src_root / "data.yaml", dst_root / "data.yaml")
    
    # 处理每个子目录
    for subset in ["train", "val", "test"]:
        src_dir = src_root / subset
        dst_dir = dst_root / subset
        
        # 创建目标子目录结构
        (dst_dir / "images").mkdir(parents=True, exist_ok=True)
        (dst_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片文件
        images = list((src_dir / "images").glob("*.jpg"))
        if not images:
            print(f"Warning: No images found in {src_dir}/images")
            continue
            
        # 随机选择16张图片
        selected_images = random.sample(images, min(16, len(images)))
        
        # 复制图片和对应的标签
        for img_path in selected_images:
            # 复制图片
            dst_img = dst_dir / "images" / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # 复制对应的标签文件
            label_name = img_path.stem + '.txt'
            src_label = src_dir / "labels" / label_name
            if src_label.exists():
                dst_label = dst_dir / "labels" / label_name
                shutil.copy2(src_label, dst_label)
            else:
                print(f"Warning: Label file not found for {img_path}")

if __name__ == "__main__":
    create_small_dataset()
    print("小型数据集创建完成!") 