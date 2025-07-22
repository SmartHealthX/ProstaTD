#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
training example code
"""

import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

# Patch modules will be imported dynamically when needed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/ssd/prostate/dataset_tool_only/dataset_triplet_only.yaml', help='config file path')
    parser.add_argument('--model', type=str, default='yolov12n.pt', help='model file or config file path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--name', type=str, default='yolo11l', help='save results folder name')
    parser.add_argument('--exist-ok', action='store_true', help='overwrite existing experiment folder')
    parser.add_argument('--patience', type=int, default=50, help='early stopping epochs')
    parser.add_argument('--test-only', action='store_true', help='only test')
    parser.add_argument('--weights', type=str, default=None, help='weights file path for testing')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold for testing')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold for testing')
    parser.add_argument('--save-txt', action='store_true', help='save predictions to txt file')
    parser.add_argument('--save-conf', action='store_true', help='save confidence scores')
    parser.add_argument('--project', type=str, default='runs', help='save results project name')
    parser.add_argument('--optimizer', type=str, default='auto', help='optimizer selection (SGD, Adam, AdamW, etc.)')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate = initial learning rate * lrf')
    parser.add_argument('--cos-lr', action='store_true', help='use cosine learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=float, default=3.0, help='warmup epochs')
    parser.add_argument('--warmup-momentum', type=float, default=0.8, help='warmup momentum')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--agnostic-nms', type=bool, default=False, help='use class-agnostic NMS (default False)')
    parser.add_argument('--tool-nms', action='store_true', help='Apply tool-based NMS patch')
    parser.add_argument('--mapping-file', type=str, default='/ssd/prostate/prostate_track_v2/triplet_maps_v2.txt', help='Path to triplet to tool mapping file')
    parser.add_argument('--apply-ivt-metrics', type=bool, default=True, help='Apply IVT metrics patch for AP50-95 calculation (default True)')
    return parser.parse_args()


def apply_ivt_metrics_patch(mapping_file):
    """Apply IVT metrics patch"""
    import ivt_metrics_patch
    success = ivt_metrics_patch.apply_patch(mapping_file)
    if success:
        print("IVT metrics patch applied")
    return success


def apply_tool_nms_if_requested(mapping_file):
    """Apply tool-based NMS patch if requested"""
    import tool_based_nms_patch
    success = tool_based_nms_patch.apply_patch(mapping_file)
    return

def print_metrics(metrics, class_names=None):
    """Print evaluation metrics"""
    if class_names is not None and len(class_names) > 0:
        print("\n--- Average metrics ---")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mPrecision: {metrics.box.mp:.4f}")
        print(f"mRecall: {metrics.box.mr:.4f}")
        print(f"mF1: {np.mean(metrics.box.f1):.4f}")


def train(args):
    """Train YOLO model"""
    model = YOLO(args.model)
    
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'patience': args.patience,
        'verbose': True,
        'project': args.project,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'cos_lr': args.cos_lr,
        'warmup_epochs': args.warmup_epochs,
        'warmup_momentum': args.warmup_momentum,
        'dropout': args.dropout
    }
    
    results = model.train(**train_args)
    return model, results


def test(model, args):
    """Test YOLO model"""
    if args.weights and os.path.exists(args.weights):
        model = YOLO(args.weights)
    
    test_args = {
        'data': args.data,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'conf': args.conf,
        'iou': args.iou,
        'verbose': True,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'project': args.project,
        'name': args.name + '_test',
        'split': 'test',
        'plots': True,
        'agnostic_nms': args.agnostic_nms,
    }
    
    results = model.val(**test_args)
    
    # Read class names
    class_names = []
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
        if 'names' in data_config:
            class_names = data_config['names']

    
    print_metrics(results, class_names)
    return results


def main():
    """Main function"""
    args = parse_args()
    
    # Apply tool-based NMS patch if requested
    if args.tool_nms:
        tool_nms_applied = apply_tool_nms_if_requested(args.mapping_file)
    
    # Apply IVT metrics patch if requested
    if args.apply_ivt_metrics:
        ivt_metrics_applied = apply_ivt_metrics_patch(args.mapping_file)
    
    # Test-only mode
    if args.test_only:
        if args.weights is None:
            args.weights = os.path.join(args.project, args.name, 'weights/best.pt')
            if not os.path.exists(args.weights):
                return
        
        model = YOLO(args.weights)
        test(model, args)
        return
    
    # Training mode
    model, train_results = train(args)
    
    # Test after training
    best_weights = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    args.weights = str(best_weights)
    test_results = test(model, args)


if __name__ == '__main__':
    main()
