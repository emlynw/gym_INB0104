import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2 import model_zoo

class DetectionAreaWrapper(gym.ObservationWrapper):
    """
    A Gymnasium wrapper that adds a 'detection' key to the observation dict,
    based on Detectron2 inference. The detection is a 4D vector:

        (found, x_norm, y_norm, mask_area)

      - found = 1 if class=1 instance is found, else 0
      - (x_norm, y_norm) is the centroid of that mask in [-1, 1],
        with (0, 0) being the image center
      - mask_area is the fraction of the image covered by the mask

    The predictor is initialized in __init__.
    """

    def __init__(self, env, detect_cam = "wrist1", model_weights_path="/home/emlyn/Downloads/aoc_model.pth"):
        super().__init__(env)

        self.detect_cam = detect_cam
        
        # -----------------------------
        # 1) Initialize Detectron2 predictor here in the wrapper
        # -----------------------------
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.WEIGHTS = model_weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        cfg.MODEL.DEVICE = "cuda"  # or "cpu" if GPU is unavailable
        self.predictor = DefaultPredictor(cfg)
        
        # -----------------------------
        # 2) Extend observation space
        # -----------------------------
        # We assume the original env returns a Dict space with at least "images" key.
        original_spaces = env.observation_space.spaces.copy()
        
        # Our detection vector = (found_flag, x_center_norm, y_center_norm, mask_area)
        # Ranges:
        #   found_flag in {0 or 1}, x/y in [-1, 1], area in [0, 1]
        detection_low  = np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32)
        detection_high = np.array([1.0,  1.0,  1.0, 1.0],  dtype=np.float32)
        
        original_spaces["detection"] = spaces.Box(
            low=detection_low,
            high=detection_high,
            shape=(4,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict(original_spaces)

    def observation(self, obs):
        """
        Runs the Detectron2 model on obs["images"][detect_camera], finds the class=1
        instance with the LARGEST mask area, and adds detection to obs.
        The (x_norm, y_norm) is the centroid of the mask in [-1,1].
        """
        # Fetch image
        detect_img = obs["images"][self.detect_cam]
        H, W, _ = detect_img.shape

        # Run prediction (you can disable autocast if you prefer)
        with torch.amp.autocast(device_type='cuda', enabled=True):
            outputs = self.predictor(detect_img)
        instances = outputs["instances"].to("cpu")

        # If no predictions, or no class=1 predictions
        if len(instances) == 0:
            obs["detection"] = np.array([0, 0, 0, 0], dtype=np.float32)
            return obs
        
        pred_classes = instances.pred_classes
        mask_class1 = (pred_classes == 0)
        
        if not mask_class1.any():
            obs["detection"] = np.array([0, 0, 0, 0], dtype=np.float32)
            return obs
        
        # Among class=1, pick the one with the LARGEST mask area
        class1_indices = torch.where(mask_class1)[0]
        
        areas = []
        for idx in class1_indices:
            mask = instances.pred_masks[idx].numpy().astype(np.uint8)
            area = mask.sum()  # pixel count
            areas.append(area)
        
        largest_area_idx_local = np.argmax(areas)
        best_idx = class1_indices[largest_area_idx_local]

        # Extract mask
        mask = instances.pred_masks[best_idx].numpy().astype(np.uint8)
        mask_area_frac = mask.sum() / float(H * W)

        # ---------------------------------
        # Compute centroid from the mask
        # ---------------------------------
        # np.argwhere(mask) -> array of (row, col) for all nonzero pixels
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            # If mask is empty for some reason, mark no detection
            obs["detection"] = np.array([0, 0, 0, 0], dtype=np.float32)
            return obs
        
        # coords[:, 0] is row (y), coords[:, 1] is col (x)
        mean_y, mean_x = coords.mean(axis=0)  # float mean
        # Convert to normalized coordinates in [-1,1],
        # with (0,0) at center of image
        x_norm = (mean_x - (W / 2.0)) / (W / 2.0)
        y_norm = (mean_y - (H / 2.0)) / (H / 2.0)

        obs["detection"] = np.array([1.0, x_norm, y_norm, mask_area_frac], dtype=np.float32)
        return obs