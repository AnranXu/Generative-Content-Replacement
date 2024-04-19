import numpy as np
from typing import Any, Dict, List
import skimage.measure
from segment_anything import SamPredictor, sam_model_registry
import torch
class mask_segment:
    def __init__(
        self,
        model_type: str,
        ckpt_p: str,
        device="cuda"):
        sam = sam_model_registry[model_type](checkpoint=ckpt_p)
        sam.to(device=device)
        #create a predictor object
        self.predictor = SamPredictor(sam)
        self.device = device
        print('finish loading model')
    
    def predict_masks_with_sam(
        self,
        img: np.ndarray,
        point_coords: List[List[float]] = None,
        point_labels: List[int] = None,
        box: List[float] = None,
        multimask_output: bool = True,
    ):
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
          With multimask_output=True (the default setting), SAM outputs 3 masks, 
          where scores gives the model's own estimation of the quality of these masks. 
          This setting is intended for ambiguous input prompts, and helps the model disambiguate different objects consistent with the prompt.
          When False, it will return a single mask. 
          For ambiguous prompts such as a single point, it is recommended to use multimask_output=True even if only a single mask is desired; 
          the best single mask can be chosen by picking the one with the highest score returned in scores. This will often result in a better mask.
        """
        if point_coords is not None:
          point_coords = np.array(point_coords)
        if point_labels is not None:
          point_labels = np.array(point_labels)
        #print('img shape', img.shape)
        self.predictor.set_image(img)
        #The model can also take a box as input, provided in xyxy format.
        if box is not None and box.shape[0] > 1:
          box = torch.tensor(box, device=self.device)
          transformed_boxes = self.predictor.transform.apply_boxes_torch(box, img.shape[:2])
          masks, scores, logits = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        else:
          masks, scores, logits = self.predictor.predict(
              point_coords=point_coords,
              point_labels=point_labels,
              box=box,
              multimask_output=multimask_output,
          )
        #self.output_polygon(masks, scores)
        return masks, scores, logits
    
    def output_polygon(self, masks, scores):
        mask = masks[np.argmax(scores)]
        def _get_contour_length(contour):
          contour_start = contour
          contour_end = np.r_[contour[1:], contour[0:1]]
          return np.linalg.norm(contour_end - contour_start, axis=1).sum()
        
        # it will return a list of contours, each contour is a list of points
        # so it's a 3-D array with shape (n, m, 2)
        contours = skimage.measure.find_contours(np.pad(mask, pad_width=1))
        #print the arrary shape
        # print('countours', contours)
        # print('countours shape', np.array(contours).shape)
        contour = max(contours, key=_get_contour_length)
        # print('max contour', contour)
        # print('max contour shape', np.array(contour).shape)
        polygon = skimage.measure.approximate_polygon(
            coords=contour,
            tolerance=np.ptp(contour, axis=0).max() / 100,
        )
        #print('polygon', polygon)
        polygon = np.clip(polygon, (0, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
        polygon = polygon[:-1]  # drop last point that is duplicate of first point
        #print('final polygon', polygon)
        return polygon
    
    
