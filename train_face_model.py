#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv12-Face训练脚本
用于训练人脸检测和5点关键点检测模型
"""

import argparse
from ultralytics import YOLO

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练YOLOv12人脸检测和关键点模型')
    parser.add_argument('--model', type=str, default='yolov12-face.yaml', help='模型配置文件或预训练模型路径')
    parser.add_argument('--data', type=str, default='face-landmarks.yaml', help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批处理大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    parser.add_argument('--device', type=str, default='', help='训练设备，例如cuda:0')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器工作线程数')
    parser.add_argument('--name', type=str, default='yolov12-face', help='保存结果的名称')
    parser.add_argument('--pretrained', type=str, default='', help='预训练模型路径')
    parser.add_argument('--patience', type=int, default=50, help='早停patience')
    return parser.parse_args()

def main(args):
    """主训练函数"""
    print(f"{'='*20} YOLOv12-Face 训练 {'='*20}")
    print(f"模型配置: {args.model}")
    print(f"数据集配置: {args.data}")
    print(f"训练轮数: {args.epochs}")
    print(f"批处理大小: {args.batch}")
    
    # 初始化模型
    if args.pretrained and not args.model.endswith('.yaml'):
        # 从预训练模型加载
        print(f"从预训练模型加载: {args.pretrained}")
        model = YOLO(args.pretrained)
    else:
        # 从配置文件创建新模型
        print(f"从配置文件创建新模型")
        model = YOLO(args.model)
    
    # 开始训练
    print("开始训练...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        name=args.name,
        patience=args.patience,
        val=True,
        pretrained=args.pretrained if args.model.endswith('.yaml') else None
    )
    
    # 训练完成，评估模型
    print("训练完成, 开始评估模型...")
    model = YOLO(f'runs/train/{args.name}/weights/best.pt')
    metrics = model.val(data=args.data)
    
    print(f"{'='*20} 评估结果 {'='*20}")
    print(f"边界框mAP50-95: {metrics.box.map:.4f}")
    print(f"边界框mAP50: {metrics.box.map50:.4f}")
    print(f"姿态mAP50-95: {metrics.pose.map:.4f}")
    print(f"姿态mAP50: {metrics.pose.map50:.4f}")
    
    print(f"\n模型已保存在: runs/train/{args.name}/weights/")
    print(f"最佳模型: runs/train/{args.name}/weights/best.pt")
    print(f"最后模型: runs/train/{args.name}/weights/last.pt")

if __name__ == "__main__":
    args = parse_args()
    main(args) 