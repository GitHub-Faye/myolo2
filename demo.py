#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv12-Face演示脚本
用于展示人脸检测和5点关键点检测的结果
"""

import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv12-Face演示')
    parser.add_argument('--model', type=str, default='runs/train/yolov12-face/weights/best.pt', 
                        help='模型路径')
    parser.add_argument('--source', type=str, default='0', 
                        help='图像/视频源，可以是文件路径、URL或摄像头编号(0)')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='置信度阈值')
    parser.add_argument('--save', action='store_true', 
                        help='保存结果')
    parser.add_argument('--show', action='store_true', 
                        help='显示结果')
    parser.add_argument('--device', type=str, default='', 
                        help='运行设备，例如cuda:0')
    return parser.parse_args()

def visualize_results(img, results, conf=0.25):
    """可视化检测结果"""
    # 深拷贝原始图像，避免修改原图
    vis_img = img.copy()
    
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return vis_img
    
    # 获取检测框和关键点
    boxes = results[0].boxes.cpu().numpy()
    keypoints = results[0].keypoints.cpu().data.numpy() if results[0].keypoints is not None else None
    
    # 关键点名称和颜色
    kpt_names = ["左眼", "右眼", "鼻尖", "左嘴角", "右嘴角"]
    kpt_colors = [
        (0, 0, 255),    # 左眼 - 红色
        (0, 0, 255),    # 右眼 - 红色
        (0, 255, 0),    # 鼻尖 - 绿色
        (255, 0, 0),    # 左嘴角 - 蓝色
        (255, 0, 0)     # 右嘴角 - 蓝色
    ]
    
    # 绘制每个检测结果
    for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
        # 检查置信度是否高于阈值
        if box.conf < conf:
            continue
            
        # 获取边界框坐标和置信度
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score = float(box.conf[0])
        
        # 绘制边界框
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制置信度文本
        label = f"人脸: {conf_score:.2f}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
        cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 绘制关键点
        if kpts is not None:
            for j, pt in enumerate(kpts[0]):
                x, y, conf = pt
                # 如果关键点可见
                if conf > 0:
                    # 绘制关键点
                    cv2.circle(vis_img, (int(x), int(y)), 3, kpt_colors[j], -1)
                    
                    # 添加关键点标签
                    if j < len(kpt_names):
                        cv2.putText(vis_img, kpt_names[j], (int(x), int(y) - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, kpt_colors[j], 1)
            
            # 连接关键点，创建面部轮廓
            # 连接左眼-鼻尖-右眼
            cv2.line(vis_img, (int(kpts[0, 0, 0]), int(kpts[0, 0, 1])), 
                   (int(kpts[0, 2, 0]), int(kpts[0, 2, 1])), (255, 255, 0), 1)
            cv2.line(vis_img, (int(kpts[0, 2, 0]), int(kpts[0, 2, 1])), 
                   (int(kpts[0, 1, 0]), int(kpts[0, 1, 1])), (255, 255, 0), 1)
            # 连接鼻尖-左嘴角-右嘴角-鼻尖
            cv2.line(vis_img, (int(kpts[0, 2, 0]), int(kpts[0, 2, 1])), 
                   (int(kpts[0, 3, 0]), int(kpts[0, 3, 1])), (255, 255, 0), 1)
            cv2.line(vis_img, (int(kpts[0, 3, 0]), int(kpts[0, 3, 1])), 
                   (int(kpts[0, 4, 0]), int(kpts[0, 4, 1])), (255, 255, 0), 1)
            cv2.line(vis_img, (int(kpts[0, 4, 0]), int(kpts[0, 4, 1])), 
                   (int(kpts[0, 2, 0]), int(kpts[0, 2, 1])), (255, 255, 0), 1)
    
    return vis_img

def main(args):
    """主函数"""
    print(f"{'='*20} YOLOv12-Face 演示 {'='*20}")
    print(f"模型: {args.model}")
    print(f"来源: {args.source}")
    
    # 加载模型
    print("加载模型...")
    model = YOLO(args.model)
    model.to(args.device)
    
    # 处理输入源
    source = args.source
    try:
        if source.isdigit():
            source = int(source)  # 摄像头
    except:
        pass
    
    # 图像或视频的处理
    if isinstance(source, (str, Path)) and not isinstance(source, int):
        # 检查是否是图像文件
        is_image = Path(source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        if is_image:
            # 处理单张图像
            print(f"处理图像: {source}")
            img = cv2.imread(source)
            results = model(img, conf=args.conf)
            
            # 可视化结果
            vis_img = visualize_results(img, results, args.conf)
            
            # 显示结果
            if args.show:
                cv2.imshow("YOLOv12-Face", vis_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # 保存结果
            if args.save:
                save_path = f"results_{Path(source).stem}.jpg"
                cv2.imwrite(save_path, vis_img)
                print(f"结果已保存至: {save_path}")
        else:
            # 处理视频
            print(f"处理视频: {source}")
            cap = cv2.VideoCapture(source)
            
            # 创建视频写入器
            if args.save:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                save_path = f"results_{Path(source).stem}.mp4"
                out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理每一帧
                results = model(frame, conf=args.conf)
                
                # 可视化结果
                vis_frame = visualize_results(frame, results, args.conf)
                
                # 在帧上添加FPS信息
                fps_text = f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}"
                cv2.putText(vis_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示结果
                if args.show:
                    cv2.imshow("YOLOv12-Face", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 保存结果
                if args.save:
                    out.write(vis_frame)
            
            # 释放资源
            cap.release()
            if args.save:
                out.release()
                print(f"结果已保存至: {save_path}")
            
            if args.show:
                cv2.destroyAllWindows()
    else:
        # 处理摄像头
        print(f"从摄像头 {source} 获取视频")
        cap = cv2.VideoCapture(source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理每一帧
            results = model(frame, conf=args.conf)
            
            # 可视化结果
            vis_frame = visualize_results(frame, results, args.conf)
            
            # 显示结果
            if args.show:
                cv2.imshow("YOLOv12-Face", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 不保存实时摄像头视频，但可以截图
            if args.save and cv2.waitKey(1) & 0xFF == ord('s'):
                save_path = f"screenshot_{len(Path('.').glob('screenshot_*.jpg')) + 1}.jpg"
                cv2.imwrite(save_path, vis_frame)
                print(f"截图已保存至: {save_path}")
        
        # 释放资源
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
    
    print(f"{'='*20} 演示结束 {'='*20}")

if __name__ == "__main__":
    args = parse_args()
    args.show = True if not args.save else args.show  # 如果不保存，则默认显示
    main(args) 