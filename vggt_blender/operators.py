"""
Defines the operators for the VGGT Blender integration, including:
- Loading images
- Running VGGT inference
- Updating visualization
- Exporting to Gaussian splatting format
"""

import os
import bpy
from bpy.types import Operator
from bpy.props import StringProperty

from .vggt_interface import VGGTInterface, VGGTPredictions


# Prediction mode constants
PREDICTION_MODE_POINTMAP = 'POINTMAP'
PREDICTION_MODE_DEPTHMAP = 'DEPTHMAP_CAMERA'

# Global storage for VGGT data between operator calls
_vggt_interface = None
_vggt_predictions = None


def get_vggt_interface():
    """Get or create the global VGGT interface instance."""
    global _vggt_interface
    if _vggt_interface is None:
        _vggt_interface = VGGTInterface()
    return _vggt_interface


def get_predictions():
    """Get the current VGGT predictions."""
    global _vggt_predictions
    return _vggt_predictions


def set_predictions(predictions):
    """Store VGGT predictions."""
    global _vggt_predictions
    _vggt_predictions = predictions


def clear_predictions():
    """Clear stored VGGT predictions."""
    global _vggt_predictions
    _vggt_predictions = None


class VGGT_OT_load_images(Operator):
    """Load images from the specified directory for VGGT processing."""
    
    bl_idname = "vggt.load_images"
    bl_label = "Load Images"
    bl_description = "Load images from the specified directory"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.vggt_props
        
        if not props.images_directory:
            self.report({'ERROR'}, "Please specify an images directory")
            return {'CANCELLED'}
        
        images_dir = bpy.path.abspath(props.images_directory)
        
        if not os.path.isdir(images_dir):
            self.report({'ERROR'}, f"Directory not found: {images_dir}")
            return {'CANCELLED'}
        
        # Find image files in the directory
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for filename in sorted(os.listdir(images_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_files.append(filename)
        
        if not image_files:
            self.report({'ERROR'}, f"No image files found in: {images_dir}")
            return {'CANCELLED'}
        
        # Update properties
        props.num_frames = len(image_files)
        
        self.report({'INFO'}, f"Found {len(image_files)} images")
        return {'FINISHED'}


class VGGT_OT_run_inference(Operator):
    """Run VGGT inference on the loaded images."""
    
    bl_idname = "vggt.run_inference"
    bl_label = "Run VGGT Inference"
    bl_description = "Run VGGT model inference on the loaded images"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.vggt_props
        
        if not props.images_directory:
            self.report({'ERROR'}, "Please specify an images directory")
            return {'CANCELLED'}
        
        images_dir = bpy.path.abspath(props.images_directory)
        
        if not os.path.isdir(images_dir):
            self.report({'ERROR'}, f"Directory not found: {images_dir}")
            return {'CANCELLED'}
        
        try:
            # Get the VGGT interface
            interface = get_vggt_interface()
            
            # Configure the interface
            cache_dir = bpy.path.abspath(props.cache_directory) if props.cache_directory else None
            interface.initialize_model(props.model_path, cache_dir)
            
            # Run inference
            self.report({'INFO'}, "Running VGGT inference... This may take a moment.")
            predictions = interface.run_inference(images_dir)
            
            # Store predictions
            set_predictions(predictions)
            
            # Update properties
            props.is_loaded = True
            props.num_frames = predictions.num_frames
            props.num_points = predictions.get_total_points()
            
            # Create initial visualization
            self._create_visualization(context, predictions, props)
            
            self.report({'INFO'}, f"VGGT inference complete. Generated {props.num_points:,} points.")
            return {'FINISHED'}
            
        except ImportError as e:
            self.report({'ERROR'}, f"Missing dependencies: {e}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"VGGT inference failed: {e}")
            return {'CANCELLED'}
    
    def _create_visualization(self, context, predictions, props):
        """Create the initial Blender visualization from predictions."""
        from .visualization import create_point_cloud, create_cameras
        
        # Get filtered points based on current settings
        prediction_mode = PREDICTION_MODE_POINTMAP if props.prediction_mode == 'POINTMAP' else PREDICTION_MODE_DEPTHMAP
        points, colors, confidence = predictions.get_filtered_points(
            conf_thres=props.conf_thres,
            mask_black_bg=props.mask_black_bg,
            mask_white_bg=props.mask_white_bg,
            prediction_mode=prediction_mode,
            frame_filter=props.frame_filter
        )
        
        # Create point cloud
        create_point_cloud(
            points, 
            colors, 
            name="VGGT_PointCloud",
            point_size=props.point_size
        )
        
        # Create cameras if enabled
        if props.show_cameras:
            create_cameras(
                predictions.extrinsic,
                name_prefix="VGGT_Camera",
                scale=props.camera_scale
            )


class VGGT_OT_update_visualization(Operator):
    """Update the visualization based on current parameter settings."""
    
    bl_idname = "vggt.update_visualization"
    bl_label = "Update Visualization"
    bl_description = "Update the 3D visualization with current parameter settings"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.scene.vggt_props.is_loaded
    
    def execute(self, context):
        props = context.scene.vggt_props
        predictions = get_predictions()
        
        if predictions is None:
            self.report({'ERROR'}, "No VGGT predictions available. Run inference first.")
            return {'CANCELLED'}
        
        try:
            from .visualization import (
                remove_vggt_objects, 
                create_point_cloud, 
                create_cameras
            )
            
            # Remove existing VGGT objects
            remove_vggt_objects()
            
            # Get filtered points based on current settings
            prediction_mode = PREDICTION_MODE_POINTMAP if props.prediction_mode == 'POINTMAP' else PREDICTION_MODE_DEPTHMAP
            points, colors, confidence = predictions.get_filtered_points(
                conf_thres=props.conf_thres,
                mask_black_bg=props.mask_black_bg,
                mask_white_bg=props.mask_white_bg,
                prediction_mode=prediction_mode,
                frame_filter=props.frame_filter
            )
            
            # Update point count
            props.num_points = len(points)
            
            # Create point cloud
            create_point_cloud(
                points, 
                colors, 
                name="VGGT_PointCloud",
                point_size=props.point_size
            )
            
            # Create cameras if enabled
            if props.show_cameras:
                create_cameras(
                    predictions.extrinsic,
                    name_prefix="VGGT_Camera",
                    scale=props.camera_scale
                )
            
            self.report({'INFO'}, f"Visualization updated with {props.num_points:,} points")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to update visualization: {e}")
            return {'CANCELLED'}


class VGGT_OT_export_gaussian_splat(Operator):
    """Export VGGT data to Gaussian splatting-compatible PLY format."""
    
    bl_idname = "vggt.export_gaussian_splat"
    bl_label = "Export Gaussian Splat"
    bl_description = "Export point cloud to Gaussian splatting-compatible PLY format"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.scene.vggt_props.is_loaded
    
    def execute(self, context):
        props = context.scene.vggt_props
        predictions = get_predictions()
        
        if predictions is None:
            self.report({'ERROR'}, "No VGGT predictions available. Run inference first.")
            return {'CANCELLED'}
        
        export_path = bpy.path.abspath(props.export_path)
        
        if not export_path:
            self.report({'ERROR'}, "Please specify an export path")
            return {'CANCELLED'}
        
        try:
            from .export import export_gaussian_splat_ply
            
            # Get filtered points based on current settings
            prediction_mode = PREDICTION_MODE_POINTMAP if props.prediction_mode == 'POINTMAP' else PREDICTION_MODE_DEPTHMAP
            points, colors, confidence = predictions.get_filtered_points(
                conf_thres=props.conf_thres,
                mask_black_bg=props.mask_black_bg,
                mask_white_bg=props.mask_white_bg,
                prediction_mode=prediction_mode,
                frame_filter=props.frame_filter
            )
            
            # Export to PLY
            export_gaussian_splat_ply(
                export_path,
                points,
                colors=colors if props.include_colors else None,
                confidence=confidence if props.include_confidence else None
            )
            
            self.report({'INFO'}, f"Exported {len(points):,} points to {export_path}")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}


class VGGT_OT_clear_scene(Operator):
    """Clear all VGGT data and objects from the scene."""
    
    bl_idname = "vggt.clear_scene"
    bl_label = "Clear VGGT Data"
    bl_description = "Remove all VGGT objects and clear stored data"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.vggt_props
        
        try:
            from .visualization import remove_vggt_objects
            
            # Remove VGGT objects from scene
            remove_vggt_objects()
            
            # Clear predictions
            clear_predictions()
            
            # Reset properties
            props.is_loaded = False
            props.num_frames = 0
            props.num_points = 0
            
            self.report({'INFO'}, "VGGT data cleared")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to clear scene: {e}")
            return {'CANCELLED'}

