#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides a non_max_suppression function
to perform NMS based on tool categories instead of triplet categories.
(Tool-based NMS: Groups detections by tool class ID (mapped from triplet ID))
"""

import torch
import numpy as np
from pathlib import Path
import time
from ultralytics.utils import LOGGER

# Import original ops module for fallback
try:
    from ultralytics.utils import ops
    ORIGINAL_NMS = ops.non_max_suppression
except ImportError as e:
    print(f"Import failed: {e}")
    raise


class TripletToToolMapper:
    """
    Mapper class to convert triplet IDs to tool IDs based on mapping file.
    
    This class loads and manages the mapping between surgical triplet IDs 
    (instrument-verb-target combinations) and their corresponding tool IDs.
    """
    
    def __init__(self, mapping_file):
        """
        Initialize the mapper with triplet-to-tool mapping.
        
        Args:
            mapping_file (str): Path to the triplet mapping file (required)
        """
        if not mapping_file:
            raise ValueError("mapping_file parameter is required")
        self.triplet_to_tool = {}
        self.max_tool_id = 0
        self.load_mapping(mapping_file)
    
    def load_mapping(self, mapping_file):
        """
        Load triplet to tool mapping from file.
        """
        try:
            mapping_path = Path(mapping_file)
            if not mapping_path.exists():
                raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
            
            with open(mapping_path, 'r') as f:
                lines = f.readlines()
            
            # Skip comment lines
            for line in lines[1:]:  # Skip header
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 6:
                        triplet_id = int(parts[0])  # IVT ID
                        tool_id = int(parts[1])     # I ID (tool/instrument ID)
                        self.triplet_to_tool[triplet_id] = tool_id
                        self.max_tool_id = max(self.max_tool_id, tool_id)
            
        except Exception as e:
            LOGGER.error(f"Error loading mapping file {mapping_file}: {e}")
            raise
    
    def get_tool_id(self, triplet_id):
        """
        Get tool ID for a given triplet ID.
        """
        return self.triplet_to_tool.get(int(triplet_id), int(triplet_id))
    
    def convert_classes_to_tools(self, classes_tensor):
        """
        Convert a tensor of triplet class IDs to tool IDs.
        """
        if len(classes_tensor) == 0:
            return classes_tensor
        
        # Convert to numpy for mapping, then back to tensor
        classes_np = classes_tensor.cpu().numpy()
        tool_classes = np.array([self.get_tool_id(cls) for cls in classes_np])
        
        return torch.tensor(tool_classes, device=classes_tensor.device, dtype=classes_tensor.dtype)


# Global mapper instance
_mapper = None

def get_mapper(mapping_file=None):
    """
    Get or create the global triplet-to-tool mapper.
    """
    global _mapper
    if _mapper is None:
        if not mapping_file:
            raise ValueError("mapping_file is required for first-time mapper initialization")
        _mapper = TripletToToolMapper(mapping_file)
    return _mapper

def set_mapping_file(mapping_file):
    """
    Set the mapping file and reinitialize the mapper.
    """
    global _mapper
    _mapper = TripletToToolMapper(mapping_file)


def tool_based_non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    end2end=False,
    return_idxs=False,
    mapping_file=None,
):
    """
    Perform tool-based non-maximum suppression (NMS) on a set of boxes.

    Args:
        prediction (torch.Tensor): Model predictions of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
        conf_thres (float): Confidence threshold for filtering
        iou_thres (float): IoU threshold for NMS
        classes (List[int], optional): List of class indices to consider
        agnostic (bool): If True, perform class-agnostic NMS
        multi_label (bool): If True, each box may have multiple labels
        labels (List): Apriori labels for each image
        max_det (int): Maximum number of detections to keep
        nc (int): Number of classes
        max_time_img (float): Maximum time per image
        max_nms (int): Maximum number of boxes for NMS
        max_wh (int): Maximum box width and height
        in_place (bool): Modify input tensor in place
        rotated (bool): Whether boxes are rotated
        end2end (bool): Whether model is end-to-end
        return_idxs (bool): Whether to return detection indices
        
    Returns:
        List[torch.Tensor]: List of detections after tool-based NMS
    """
    
    # For end-to-end models or simple cases, use original NMS
    if end2end or (isinstance(prediction, (list, tuple)) and prediction[0].shape[-1] == 6):
        return ORIGINAL_NMS(
            prediction, conf_thres, iou_thres, classes, agnostic, multi_label, 
            labels, max_det, nc, max_time_img, max_nms, max_wh, in_place, 
            rotated, end2end, return_idxs
        )
    
    # Get the mapper
    try:
        mapper = get_mapper(mapping_file)
    except ValueError as e:
        LOGGER.error(f"Tool-based NMS requires mapping file: {e}")
        # Fallback to original NMS if no mapping file provided
        return ORIGINAL_NMS(
            prediction, conf_thres, iou_thres, classes, agnostic, multi_label, 
            labels, max_det, nc, max_time_img, max_nms, max_wh, in_place, 
            rotated, end2end, return_idxs
        )
    
    # Log the NMS mode being used
    if not hasattr(tool_based_non_max_suppression, '_logged'):
        tool_based_non_max_suppression._logged = True
    
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xinds = torch.stack([torch.arange(len(i), device=prediction.device) for i in xc])[..., None]  # to track idxs

    # Settings
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = ops.xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((ops.xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs
    
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        # Apply constraints
        filt = xc[xi]  # confidence
        x, xk = x[filt], xk[filt]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = ops.xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            xk = xk[i]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            xk = xk[filt]

        # Filter by class
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x, xk = x[filt], xk[filt]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            x, xk = x[filt], xk[filt]

        # TOOL-BASED NMS: Convert triplet classes to tool classes
        original_classes = x[:, 5].clone()  # Keep original classes for output
        tool_classes = mapper.convert_classes_to_tools(x[:, 5])  # Convert to tool classes
        
        # Replace classes with tool classes for NMS grouping
        x[:, 5] = tool_classes.float()

        # Batched NMS using tool classes
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes (now tool-based)
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = ops.nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by tool class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        # 🔧 Restore original triplet classes in output
        x[i, 5] = original_classes[i]  # Restore original triplet classes
        
        output[xi], keepi[xi] = x[i], xk[i].reshape(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return (output, keepi) if return_idxs else output


def apply_patch(mapping_file=None):
    """Apply tool-based NMS patch"""
    # Initialize mapper with mapping_file
    if mapping_file:
        set_mapping_file(mapping_file)
    
    # Patch ops module
    import ultralytics.utils.ops as ops_module
    import ultralytics.utils as utils_module
    
    # Store original function for restoration
    if not hasattr(ops_module, '_original_non_max_suppression'):
        ops_module._original_non_max_suppression = ops_module.non_max_suppression
    
    # Create wrapper function
    def patched_nms(*args, **kwargs):
        if 'mapping_file' not in kwargs and mapping_file:
            kwargs['mapping_file'] = mapping_file
        return tool_based_non_max_suppression(*args, **kwargs)
    
    # Apply patches
    ops_module.non_max_suppression = patched_nms
    utils_module.ops.non_max_suppression = patched_nms
    
    print("Tool-based NMS patch applied")
    return True


if __name__ == "__main__":
    # Test the mapper with default mapping file
    default_mapping_file = "./triplet_maps_v2.txt"
    
    mapper = TripletToToolMapper(default_mapping_file)
    print(f"Loaded {len(mapper.triplet_to_tool)} mappings")
    print(f"Max tool ID: {mapper.max_tool_id}")
    
    # Test some mappings
    for triplet_id in [0, 1, 2, 5, 10]:
        tool_id = mapper.get_tool_id(triplet_id)
        print(f"Triplet {triplet_id} -> Tool {tool_id}")
    
    # Test tensor conversion
    test_classes = torch.tensor([0, 1, 2, 5, 10])
    tool_classes = mapper.convert_classes_to_tools(test_classes)
    print(f"Tensor conversion: {test_classes.tolist()} -> {tool_classes.tolist()}")
        
