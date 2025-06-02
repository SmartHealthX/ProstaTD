# ProstaTD

### **ProstaTD: A Large-scale Multi-source Dataset for Structured Surgical Triplet Detection**
*Yiliang Chen, Zhixi Li, Cheng Xu, Alex Qinyang Liu, Xuemiao Xu, Jeremy Yuen-Chun Teoh, Shengfeng He, Jing Qin*

## Abstract
<div align="center">
  <img src="images/dataset.png" width="100%"/>
</div><br/>
ProstaTD is a large-scale surgical triplet detection dataset curated from 21 robot-assisted prostatectomy videos, collectively spanning full surgical procedures across multiple institutions, featuring 60,529 annotated frames with 165,567 structured surgical triplet instances (instrument-verb-target) that provide precise bounding box localization for all instruments alongside clinically validated temporal action boundaries. The dataset incorporates the ESAD and PSI-AVA datasets with our own added annotations (without using the original data annotations). We also include our own collected videos. It delivers pixel-level annotations for 7 instrument types, 10 actions, 10 anatomical/non-anatomical targets, and 89 triplet combinations. Dataset is partitioned into training (14 videos), validation (2 videos), and test sets (5 videos), with annotations provided at 1 frame per second.

### Dataset format:
Our dataset format is: [triplet id, instrument id, verb id, target id, track id, triplet track id, cx, cy, w, h]. In the current release, the track id, triplet track id, cx, cy, w, h have not been officially released yet and are temporarily replaced with the value -1. They will be released soon.

## News: 
- [ **02/06/2025** ]: Release of the ProstaTDv1.1 dataset on github.
- [ **16/05/2025** ]: Release of the ProstaTDv1.0 dataset on kaggle.


## Download Access:
To request access to the ProstaTD Dataset, please fill out our [request form].

## License
This repository is available for non-commercial scientific research purposes as defined in the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). By downloading and using this code, you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.
