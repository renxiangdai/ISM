from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from pycocotools import mask as mask_utils
import logging
import numpy as np
import os.path as osp
import cv2
from typing import Any, Dict, List
import torch
import torch.nn.functional as F
import sam2

def load_sam2(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use absolute paths for both config and checkpoint
    root_dir = "/data/rxdai/Instance_Segmentation_Model"
    
    # SAM2 config file path
    config_file = osp.join(root_dir, "sam2/configs/sam2.1", f"{model_name}.yaml")
    
    # Load pretrained SAM2.1 model
    checkpoint_path = osp.join(root_dir, "checkpoints/sam2", "sam2.1_hiera_base_plus.pt")
    
    logging.info(f"Loading SAM2 model from {checkpoint_path}")
    model = build_sam2(
        config_file=config_file,
        ckpt_path=checkpoint_path,
        device=device
    )
    return model

class CustomSAM2AutomaticMaskGenerator:
    def __init__(
        self,
        sam2_model_name,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        crop_nms_thresh=0.7,
        min_mask_region_area=0,
        output_mode="coco_rle",
        segmentor_width_size=None,
        **kwargs,
    ):
        if segmentor_width_size is None:
            segmentor_width_size = 640
        self.segmentor_width_size = segmentor_width_size
        self.model = load_sam2(sam2_model_name)
        
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            min_mask_region_area=min_mask_region_area,
            output_mode=output_mode,
        )
        logging.info(f"Init CustomSAM2AutomaticMaskGenerator done!")

    def preprocess_resize(self, image: np.ndarray):
        orig_size = image.shape[:2]
        height_size = int(self.segmentor_width_size * orig_size[0] / orig_size[1])
        resized_image = cv2.resize(
            image.copy(), (self.segmentor_width_size, height_size)  # (width, height)
        )
        return resized_image, orig_size

    def postprocess_resize(self, masks, boxes, orig_size):
        """
        Resize masks and boxes back to original image size
        """
        device = masks.device
        
        # Resize masks using interpolate
        masks = F.interpolate(
            masks.unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]

        # Scale boxes
        scale_w = orig_size[1] / self.segmentor_width_size  
        scale_h = scale_w  # Keep aspect ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h
        
        # Clamp boxes using torch operations
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0, max=orig_size[1] - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0, max=orig_size[0] - 1)

        return masks, boxes
    
    @torch.no_grad()
    def generate_masks(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Generate masks for the input image.
        Args:
            image: RGB input image in HWC format
        Returns:
            dict: Dictionary containing masks and boxes tensors.
        """
        device = self.model.device
        
        # Resize image if needed
        if self.segmentor_width_size:
            image_resized, orig_size = self.preprocess_resize(image)
        else:
            image_resized = image
            orig_size = image.shape[:2]

        # Generate masks
        masks = self.mask_generator.generate(image_resized)
        print(f"Generated {len(masks)} masks")
    
        # Convert format
        all_masks = []
        all_boxes = []
        
        for mask in masks:
            # Process mask
            if isinstance(mask["segmentation"], dict):
                # Handle RLE format using pycocotools
                rle = mask["segmentation"]
                mask_tensor = mask_utils.decode(rle)
            else:
                # Handle binary mask format
                mask_tensor = mask["segmentation"]
            mask_tensor = torch.from_numpy(mask_tensor).to(device=device)
            all_masks.append(mask_tensor)
            
            # Process bbox: convert from xywh to xyxy format
            bbox = np.array(mask["bbox"])
            xyxy_box = np.array([
                bbox[0],                # x1
                bbox[1],                # y1
                bbox[0] + bbox[2],      # x2
                bbox[1] + bbox[3]       # y2
            ])
            all_boxes.append(xyxy_box)
        
        # Convert to tensors
        masks_tensor = torch.stack(all_masks)
        boxes_tensor = torch.tensor(np.stack(all_boxes), device=device)
        
        # Resize if needed
        if self.segmentor_width_size:
            masks_tensor, boxes_tensor = self.postprocess_resize(
                masks_tensor, 
                boxes_tensor, 
                orig_size
            )
        
        return {
            "masks": masks_tensor,
            "boxes": boxes_tensor
        }
