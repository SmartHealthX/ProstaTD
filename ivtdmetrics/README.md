# IVTDMetrics

An extended metrics package for surgical triplet detection, based on [ivtmetrics](https://github.com/CAMMA-public/ivtmetrics).

## Overview

**IVTDMetrics** This package addresses and fixes several calculation errors in the original ivtmetrics detection module, and provide more accurate evaluation metrics for surgical triplet detection.

## Features

- **Recognition Metrics**: Same as ivtmetrics, compute Average Precision (AP) for multi-label classification tasks
- **Detection Metrics**: Evaluate object detection performance with IoU-based matching

## Installation

### From source

```bash
cd ivtdmetrics
pip install -e ./
```

### Requirements

- Python >= 3.6
- scikit-learn >= 1.0.2
- numpy >= 1.21

## Quick Start

### Detection Evaluation

```python
from ivtdmetrics import Detection

# Initialize detection evaluator
detector = Detection(
    num_class=89,
    num_tool=7,
    threshold=0.5
)

# Prepare detection data
# Format: [class_id, confidence, x, y, width, height]
targets = [
    [[1, 1.0, 0.1, 0.1, 0.2, 0.3], [3, 1.0, 0.5, 0.5, 0.2, 0.2]],  # Frame 1
    [[2, 1.0, 0.2, 0.3, 0.3, 0.4]]                                   # Frame 2
]

predictions = [
    [[1, 0.9, 0.12, 0.11, 0.18, 0.28], [3, 0.8, 0.52, 0.51, 0.19, 0.21]],  # Frame 1
    [[2, 0.7, 0.22, 0.32, 0.28, 0.38]] # Frame 2
]

# Update with detection data
detector.update(targets, predictions, format="list")

# Signal end of video
detector.video_end()

# Compute detection metrics
results = detector.compute_video_AP(component="ivt")
print(f"Detection mAP: {results['mAP']:.4f}")
print(f"Association metrics - LM: {results['lm']:.4f}, IDS: {results['ids']:.4f}")
```

#### Input Format

Detection data should be provided as lists of detections per frame:
- **List format**: `[[class_id, confidence, x, y, width, height], ...]`
- **Dict format**: `[{"triplet": id, "instrument": [id, conf, x, y, w, h]}, ...]`


#### Supported Components

- `"ivt"`: Instrument-Verb-Target (full triplet)
- `"i"`: Instrument only

## Advanced Usage

### Multi-Video Detection Evaluation

```python
from ivtdmetrics import Detection

detector = Detection(num_class=89, num_tool=7, threshold=0.5)

# Process multiple videos
for video_data in video_dataset:
    for targets, predictions in video_data:
        detector.update(targets, predictions, format="list")
    detector.video_end()  # Signal end of current video

# Get video-wise performance
video_results = detector.compute_video_AP()
print(f"Video-wise mAP: {video_results['mAP']:.4f}")

# Get global performance across all videos
global_results = detector.compute_global_AP()
print(f"Global mAP: {global_results['mAP']:.4f}")
```

### Custom IoU Threshold

```python
from ivtdmetrics import Detection

# Use COCO-style AP calculation
results = detector.compute_video_AP(style="coco")
```

## To-Do List
- ✅ Release package
- ⭕️ **Component Disentanglement**: Currently, we have not fully implemented component filtering features such as `iv` (instrument-verb) and `it` (instrument-target) pair evaluations, as their practical significance may be limited for surgical triplet detection tasks.
- ⭕️ **Triplet Tracking Metric** 

## Citation

If you use this package in your research, please cite:

```bibtex
@article{chen2025prostatd,
  title     = {ProstaTD: A Large-scale Multi-source Dataset for Structured Surgical Triplet Detection},
  author    = {Chen, Yiliang and Li, Zhixi and Xu, Cheng and Liu, Alex Qinyang and Xu, Xuemiao and Teoh, Jeremy Yuen-Chun and He, Shengfeng and Qin, Jing},
  journal   = {arXiv preprint arXiv:2506.01130},
  year      = {2025}
}
```

## Acknowledgments

This work is based on [ivtmetrics](https://github.com/CAMMA-public/ivtmetrics).

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer. 
