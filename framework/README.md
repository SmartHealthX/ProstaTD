# Surgical Triplet Detection

A simple training script for surgical action triplet detection using YOLO.

## Installation

```bash
# Clone the repository
git clone https://github.com/SmartHealthX/ProstaTD
cd prostaTD/framework/lib
# Install ultralytics
pip install -e ./
cd ..
```
If you want to use our calculation metrics, please follow this [ivtdmetrics](https://github.com/SmartHealthX/ProstaTD/tree/main/framework).

## Dataset Structure

The dataset should be organized as follows:

```
dataset/
├── dataset.yaml    # Dataset configuration file
├── triplet_maps_v2.txt          # Triplet to tool mapping file
├── train/
│   ├── images/                  # Training images (.jpg)
│   └── labels/                  # Training labels (.txt)
├── val/
│   ├── images/                  # Validation images (.jpg)
│   └── labels/                  # Validation labels (.txt)
└── test/
    ├── images/                  # Test images (.jpg)
    └── labels/                  # Test labels (.txt)
```

### Label Format

Each label file contains annotations in YOLO format:
```
class_id center_x center_y width height
```

Where:
- `class_id`: Triplet class ID (0-88)
- `center_x, center_y`: Normalized center coordinates
- `width, height`: Normalized bounding box dimensions

### Dataset Configuration

The `dataset.yaml` file contains:
- Dataset paths
- Number of classes (89 triplet classes)
- Class names (instrument_verb_target format)

## Usage

### Training

```bash
python train_yolo.py \
    --data /path/to/dataset.yaml \
    --model yolo11l.pt \
    --epochs 100 \
    --batch 16 \
    --mapping-file /path/to/triplet_maps.txt \
    --apply-ivt-metrics true
```

### Testing

```bash
python train_yolo.py \
    --data /path/to/dataset.yaml \
    --test-only \
    --model yolo11l.pt \
    --name,  yolov11l, \
    --weights /path/to/best.pt \
    --mapping-file /path/to/triplet_maps.txt
```

## Parameters

- **`--agnostic-nms`**: Use class-agnostic NMS (default: False)
- **`--tool-nms`**: Apply tool-based NMS patch
- **`--mapping-file`**: Path to triplet to tool mapping file
- **`--apply-ivt-metrics`**: Apply IVT Detection metrics patch for AP50-95 calculation (default: True)

## Acknowledgments

This code is based on the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework. 
