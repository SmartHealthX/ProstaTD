# IVTDMetrics

An extended metrics package for surgical triplet detection.

## Overview

**IVTDMetrics** This package addresses several calculation errors in the original ivtmetrics detection module, and provide more accurate evaluation metrics for surgical triplet detection.

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
- numpy >= 1.21

## Quick Start

#### Input Format

Detection data should be provided as lists of detections per frame:
- **List format**: ` [[tripletID, toolID, Confidences, x, y, w, h], [tripletID, toolID, Confidences, x, y, w, h], ...]`
- **Dict format**: `[{"triplet":tripletID, "instrument":[toolID, Confidences, x, y, w, h]}, {"triplet":tripletID, "instrument":[toolID, Confidences, x, y, w, h]}, ...]`


#### Supported Components

- `"ivt"`: Instrument-Verb-Target (full triplet)
- `"i"`: Instrument only

#### Multi-Video Detection Evaluation Example

```python
from ivtdmetrics.detection import Detection
import numpy as np

# Test data
video1_targets = [
    [[0, 1, 1.0, 0.1, 0.1, 0.2, 0.2], [1, 2, 1.0, 0.5, 0.5, 0.2, 0.2]],
    [[2, 3, 1.0, 0.3, 0.3, 0.2, 0.2]]
]

video1_predictions = [
    [[0, 1, 0.9, 0.1, 0.1, 0.2, 0.2], [1, 2, 0.8, 0.7, 0.7, 0.1, 0.1], [3, 4, 0.7, 0.8, 0.8, 0.1, 0.1]],
    []
]

video2_targets = [
    [[2, 5, 1.0, 0.1, 0.1, 0.3, 0.3]],
    [[4, 5, 1.0, 0.2, 0.2, 0.3, 0.3]]
]

video2_predictions = [
    [[2, 5, 0.9, 0.1, 0.1, 0.3, 0.3], [6, 2, 0.5, 0.8, 0.8, 0.1, 0.1]],
    [[4, 5, 0.95, 0.2, 0.2, 0.3, 0.3], [7, 1, 0.4, 0.6, 0.6, 0.1, 0.1]]
]

video3_targets = [
    [[8, 0, 1.0, 0.0, 0.0, 0.3, 0.3], [9, 1, 1.0, 0.4, 0.4, 0.2, 0.2], [10, 2, 1.0, 0.7, 0.7, 0.2, 0.2]],
    [[11, 3, 1.0, 0.1, 0.1, 0.4, 0.4]]
]

video3_predictions = [
    [[8, 0, 0.9, 0.0, 0.0, 0.3, 0.3], [9, 1, 0.8, 0.45, 0.45, 0.15, 0.15], [12, 4, 0.6, 0.8, 0.8, 0.1, 0.1]],
    [[11, 3, 0.85, 0.15, 0.15, 0.35, 0.35], [11, 3, 0.7, 0.2, 0.2, 0.3, 0.3]]
]

all_targets = [video1_targets, video2_targets, video3_targets]
all_predictions = [video1_predictions, video2_predictions, video3_predictions]

detector = Detection(num_class=89, num_tool=7, threshold=0.5)

for video_idx, (targets, predictions) in enumerate(zip(all_targets, all_predictions)):
    detector.update(targets, predictions, format="list")
    detector.video_end()

# style = "coco" is default setting
video_ap_ivt = detector.compute_video_AP(component="ivt", style="coco")
global_ap_ivt = detector.compute_global_AP(component="ivt", style="coco")
video_ap_i = detector.compute_video_AP(component="i", style="coco")
global_ap_i = detector.compute_global_AP(component="i", style="coco")

print(f"IVT Video-wise mAP:  {video_ap_ivt['mAP']:.4f}")
print(f"IVT Global mAP:      {global_ap_ivt['mAP']:.4f}")
print(f"I Video-wise mAP:    {video_ap_i['mAP']:.4f}")
print(f"I Global mAP:        {global_ap_i['mAP']:.4f}") 
```

### Calculation ###

```python
# Use ultralytics AP calculation
results = detector.compute_video_AP(style="coco") # default

# Use orginal AP calculation
results = detector.compute_video_AP(style="11point")

....
....

# Other metrics (based on optimal global F1 threshold)
print(f"IVT Video-wise mRec: {video_ap_ivt['mRec']:.4f}") 
print(f"IVT Video-wise mPre: {video_ap_ivt['mPre']:.4f}")
print(f"IVT Video-wise mF1:  {video_ap_ivt['mF1']:.4f}") 
```

## To-Do List
- ✅ Release package
- ⭕️ **MAP50_95 Support** 
- ⭕️ **AR metric for DETR-based models** 
- ⭕️ **Component Disentanglement**: Currently, we have not fully implemented component filtering features such as `iv` (instrument-verb) and `it` (instrument-target) pair evaluations, as their practical significance may be limited for surgical triplet detection tasks.
- ⭕️ **Triplet Tracking Metric** 

## Citation

If you use this package in your research, please cite:

```bibtex
@misc{chen2025prostatd,
      title={ProstaTD: A Large-scale Multi-source Dataset for Structured Surgical Triplet Detection}, 
      author={Yiliang Chen and Zhixi Li and Cheng Xu and Alex Qinyang Liu and Xuemiao Xu and Jeremy Yuen-Chun Teoh and Shengfeng He and Jing Qin},
      year={2025},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2506.01130}, 
}
```

## Acknowledgments

This work is based on [ivtmetrics](https://github.com/CAMMA-public/ivtmetrics). The enhancements include global confidence ranking (video/global-level), handling of pseudo-detections when test set ground truth doesn't contain specific classes, 101-point interpolation, and various other corrections.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer. 
