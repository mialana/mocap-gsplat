"""
Provides the interface between Blender and the VGGT model.
This module handles model initialization, inference, and prediction processing.
"""

import os
import gc
import glob
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


@dataclass
class VGGTPredictions:
    """
    Data class for storing VGGT model predictions.
    
    Attributes:
        world_points: 3D point coordinates from pointmap branch (S, H, W, 3)
        world_points_conf: Confidence scores for pointmap (S, H, W)
        world_points_from_depth: 3D points from depth map (S, H, W, 3)
        depth_conf: Confidence scores for depth-based points (S, H, W)
        images: Input images (S, H, W, 3) or (S, 3, H, W)
        extrinsic: Camera extrinsic matrices (S, 3, 4)
        intrinsic: Camera intrinsic matrices (S, 3, 3)
        depth: Raw depth maps (S, H, W, 1)
    """
    world_points: Optional[np.ndarray] = None
    world_points_conf: Optional[np.ndarray] = None
    world_points_from_depth: Optional[np.ndarray] = None
    depth_conf: Optional[np.ndarray] = None
    images: Optional[np.ndarray] = None
    extrinsic: Optional[np.ndarray] = None
    intrinsic: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    
    @property
    def num_cameras(self) -> int:
        """Return the number of cameras in the predictions."""
        if self.images is not None:
            return self.images.shape[0]
        if self.world_points is not None:
            return self.world_points.shape[0]
        if self.world_points_from_depth is not None:
            return self.world_points_from_depth.shape[0]
        return 0
    
    def get_total_points(self) -> int:
        """Return the total number of 3D points across all cameras."""
        if self.world_points is not None:
            return np.prod(self.world_points.shape[:-1])
        if self.world_points_from_depth is not None:
            return np.prod(self.world_points_from_depth.shape[:-1])
        return 0
    
    def get_filtered_points(
        self,
        conf_thres: float = 50.0,
        mask_black_bg: bool = False,
        mask_white_bg: bool = False,
        prediction_mode: str = 'DEPTHMAP_CAMERA',
        frame_filter: str = 'ALL'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get filtered 3D points based on the specified parameters.
        
        This method mirrors the filtering logic from the VGGT Gradio demo.
        
        Args:
            conf_thres: Percentage of low-confidence points to filter out (0-100)
            mask_black_bg: Remove points with black background
            mask_white_bg: Remove points with white background
            prediction_mode: 'POINTMAP' or 'DEPTHMAP_CAMERA'
            frame_filter: 'ALL' or 'FRAME_N' where N is the frame index
            
        Returns:
            Tuple of (points, colors, confidence) arrays
        """
        # Select prediction source based on mode
        if prediction_mode == 'POINTMAP' and self.world_points is not None:
            pred_world_points = self.world_points
            pred_conf = self.world_points_conf if self.world_points_conf is not None else \
                       np.ones(pred_world_points.shape[:-1])
        else:
            pred_world_points = self.world_points_from_depth
            pred_conf = self.depth_conf if self.depth_conf is not None else \
                       np.ones(pred_world_points.shape[:-1])
        
        images = self.images
        
        # Handle frame filtering
        if frame_filter != 'ALL' and frame_filter.startswith('FRAME_'):
            try:
                frame_idx = int(frame_filter.split('_')[1])
                pred_world_points = pred_world_points[frame_idx:frame_idx+1]
                pred_conf = pred_conf[frame_idx:frame_idx+1]
                images = images[frame_idx:frame_idx+1]
            except (ValueError, IndexError):
                pass
        
        # Flatten points for filtering
        vertices_3d = pred_world_points.reshape(-1, 3)
        
        # Handle different image formats (NCHW vs NHWC)
        if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
            colors_rgb = np.transpose(images, (0, 2, 3, 1))
        else:  # Assume already in NHWC format
            colors_rgb = images
        
        # Normalize colors to 0-255 range if needed
        if colors_rgb.max() <= 1.0:
            colors_rgb = (colors_rgb * 255).astype(np.uint8)
        else:
            colors_rgb = colors_rgb.astype(np.uint8)
        
        colors_rgb = colors_rgb.reshape(-1, 3)
        conf = pred_conf.reshape(-1)
        
        # Apply confidence threshold (percentile-based)
        # When conf_thres is 0, keep all points (no filtering)
        # Otherwise, filter by the specified percentile
        conf_threshold = np.percentile(conf, conf_thres) if conf_thres > 0.0 else 0.0
        
        conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
        
        # Apply black background mask
        if mask_black_bg:
            black_bg_mask = colors_rgb.sum(axis=1) >= 16
            conf_mask = conf_mask & black_bg_mask
        
        # Apply white background mask
        if mask_white_bg:
            white_bg_mask = ~(
                (colors_rgb[:, 0] > 240) & 
                (colors_rgb[:, 1] > 240) & 
                (colors_rgb[:, 2] > 240)
            )
            conf_mask = conf_mask & white_bg_mask
        
        # Apply mask to all arrays
        filtered_points = vertices_3d[conf_mask]
        filtered_colors = colors_rgb[conf_mask]
        filtered_conf = conf[conf_mask]
        
        return filtered_points, filtered_colors, filtered_conf


class VGGTInterface:
    """
    Interface class for VGGT model operations.
    
    This class provides methods for:
    - Model initialization and loading
    - Running inference on images
    - Processing and returning predictions
    """
    
    def __init__(self):
        self.model = None
        self.device = None
        self.dtype = None
        self._initialized = False
    
    def initialize_model(self, model_path: str = "facebook/VGGT-1B", cache_dir: Optional[str] = None):
        try:
            import torch
            from vggt.models.vggt import VGGT
        except ImportError as e:
            print(f"Import error: {e}")
            raise ImportError(
                "VGGT dependencies not available. Please ensure the VGGT submodule "
                "is initialized and dependencies are installed."
            )
        
        if self.model:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
        
        # Set up device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch.cuda.is_available():
            print("Cuda is available.")
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
            print("Cuda is not available.")
        
        # Initialize model
        if model_path.startswith("facebook/") or "/" not in model_path:
            # Load from HuggingFace
            kwargs = {}
            if cache_dir:
                kwargs['cache_dir'] = cache_dir
                print("Local cache directory found.")
            self.model = VGGT.from_pretrained(model_path, **kwargs)
            print("Model loaded from pretrained.")
        else:
            # Load from local path
            self.model = VGGT()
            import torch
            # Use weights_only=True for safer loading of model files
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
        
        self.model.eval()
        self.model = self.model.to(self.device)
        self._initialized = True

        print("Model successfully initialized.")
    
    def run_inference(self, images_directory: str) -> VGGTPredictions:
        if not self._initialized or not self.model:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        # garbage collect and clear cache
        gc.collect()
        torch.cuda.empty_cache()
        
        # Find and load images
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_names = []
        for pattern in image_patterns:
            image_names.extend(glob.glob(os.path.join(images_directory, pattern)))
        image_names = sorted(image_names)
        
        if not image_names:
            raise ValueError(f"No images found in {images_directory}")
        
        # Load and preprocess images
        images = load_and_preprocess_images(image_names).to(self.device)
        
        # Run inference with automatic mixed precision when CUDA is available
        with torch.no_grad():
            enabled = self.device == "cuda"
            print(f"Running inference with cuda: {enabled}")
            with torch.amp.autocast(self.device, dtype=self.dtype, enabled=(self.device == "cuda")):
                predictions = self.model(images)
        
        # Convert pose encoding to extrinsic and intrinsic matrices
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], 
            images.shape[-2:]
        )
        
        # Convert tensors to numpy
        result = {}
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                result[key] = predictions[key].cpu().numpy().squeeze(0)
        
        result["extrinsic"] = extrinsic.cpu().numpy().squeeze(0) if isinstance(extrinsic, torch.Tensor) else extrinsic
        result["intrinsic"] = intrinsic.cpu().numpy().squeeze(0) if isinstance(intrinsic, torch.Tensor) else intrinsic
        
        # Generate world points from depth map
        depth_map = result.get("depth")
        if depth_map is not None:
            world_points = unproject_depth_map_to_point_map(
                depth_map, 
                result["extrinsic"], 
                result["intrinsic"]
            )
            result["world_points_from_depth"] = world_points
        
        # clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create VGGTPredictions object
        return VGGTPredictions(
            world_points=result.get("world_points"),
            world_points_conf=result.get("world_points_conf"),
            world_points_from_depth=result.get("world_points_from_depth"),
            depth_conf=result.get("depth_conf"),
            images=result.get("images"),
            extrinsic=result.get("extrinsic"),
            intrinsic=result.get("intrinsic"),
            depth=result.get("depth")
        )
    
    def cleanup(self):
        """Release model resources."""
        if self.model is not None:
            import torch
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
        self._initialized = False
