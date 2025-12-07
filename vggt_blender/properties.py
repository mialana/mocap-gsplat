"""
Defines all the properties for the VGGT Blender integration.
"""

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
    CollectionProperty,
)
from bpy.types import PropertyGroup

class VGGTImageItem(PropertyGroup):
    """Property group for storing loaded image paths."""

    filepath: StringProperty(
        name="File Path",
        description="Path to the image file",
        default="",
        subtype="FILE_PATH",
    )
    name: StringProperty(name="Name", description="Image filename", default="")


class VGGTProperties(PropertyGroup):
    """
    Property group containing all VGGT parameters and settings.

    These properties mirror the parameter tuning options available
    in the VGGT Gradio demo for consistency.
    """

    # Image loading properties
    images_directory: StringProperty(
        name="Images Directory",
        description="Directory containing input images",
        default="",
        subtype="DIR_PATH",
    )

    # Model settings
    model_path: StringProperty(
        name="Model Path",
        description="Path to VGGT model weights (or HuggingFace model ID)",
        default="facebook/VGGT-1B",
    )

    cache_directory: StringProperty(
        name="Cache Directory",
        description="Directory for caching model weights",
        default="",
        subtype="DIR_PATH",
    )

    # Parameter tuning options
    conf_thres: FloatProperty(
        name="Confidence Threshold (%)",
        description="Percentage of low-confidence points to filter out",
        default=50.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
        update=lambda self, context: update_visualization_callback(self, context),
    )

    prediction_mode: EnumProperty(
        name="Prediction Mode",
        description="Select the prediction mode for point generation",
        items=[
            (
                "DEPTHMAP_CAMERA",
                "Depthmap and Camera Branch",
                "Use depth map and camera parameters to generate world points",
            ),
            (
                "POINTMAP",
                "Pointmap Branch",
                "Use direct pointmap regression for world points",
            ),
        ],
        default="DEPTHMAP_CAMERA",
        update=lambda self, context: update_visualization_callback(self, context),
    )

    mask_black_bg: BoolProperty(
        name="Filter Black Background",
        description="Remove points with black background (sum of RGB < 16)",
        default=False,
        update=lambda self, context: update_visualization_callback(self, context),
    )

    mask_white_bg: BoolProperty(
        name="Filter White Background",
        description="Remove points with white background (RGB values > 240)",
        default=False,
        update=lambda self, context: update_visualization_callback(self, context),
    )

    mask_sky: BoolProperty(
        name="Filter Sky",
        description="Apply sky segmentation mask to remove sky points",
        default=False,
        update=lambda self, context: update_visualization_callback(self, context),
    )

    show_cameras: BoolProperty(
        name="Show Cameras",
        description="Display camera positions and orientations in the scene",
        default=True,
        update=lambda self, context: update_visualization_callback(self, context),
    )

    # Frame selection
    # Note: EnumProperty with dynamic items requires the callback function
    # to be defined before the class. We use a lambda wrapper to avoid
    # forward reference issues.
    frame_filter: EnumProperty(
        name="Show Points from Frame",
        description="Select which frames to display points from",
        items=lambda self, context: get_frame_filter_items(self, context),
        update=lambda self, context: update_visualization_callback(self, context),
    )

    # Visualization settings
    point_size: FloatProperty(
        name="Point Size",
        description="Size of points in the visualization",
        default=0.01,
        min=0.001,
        max=0.1,
        soft_min=0.001,
        soft_max=0.1,
        step=1,
        precision=3,
    )

    camera_scale: FloatProperty(
        name="Camera Scale",
        description="Scale of camera visualization objects",
        default=0.1,
        min=0.01,
        max=1.0,
        soft_min=0.01,
        soft_max=1.0,
        step=1,
        precision=2,
    )

    # Export settings
    export_path: StringProperty(
        name="Export Path",
        description="Path for exporting Gaussian splatting data",
        default="//vggt_export.ply",
        subtype="FILE_PATH",
    )

    include_confidence: BoolProperty(
        name="Include Confidence",
        description="Include confidence scores in the exported PLY file",
        default=True,
    )

    include_colors: BoolProperty(
        name="Include Colors",
        description="Include RGB color information in the exported PLY file",
        default=True,
    )

    # Internal state tracking
    is_loaded: BoolProperty(
        name="Data Loaded",
        description="Whether VGGT predictions have been loaded",
        default=False,
    )

    num_frames: IntProperty(
        name="Number of Frames",
        description="Number of frames in the current dataset",
        default=0,
    )

    num_points: IntProperty(
        name="Number of Points", description="Total number of 3D points", default=0
    )
    
    # Installation state tracking
    packages_installed: BoolProperty(
        name="Packages Installed",
        description="Whether required pip packages have been installed",
        default=False,
    )
    
    vggt_model_installed: BoolProperty(
        name="VGGT Model Installed",
        description="Whether the VGGT model has been downloaded and installed",
        default=False,
    )
    
    installation_in_progress: BoolProperty(
        name="Installation In Progress",
        description="Whether an installation operation is currently running",
        default=False,
    )
    
    installation_message: StringProperty(
        name="Installation Message",
        description="Current status message during installation",
        default="",
    )
    
    installation_progress: FloatProperty(
        name="Installation Progress",
        description="Progress of current installation (0-100)",
        default=0.0,
        min=0.0,
        max=100.0,
    )


def get_frame_filter_items(self, context):
    """
    Dynamically generate frame filter enum items based on loaded data.

    Returns:
        list: List of enum items for frame selection
    """
    items = [("ALL", "All Frames", "Show points from all frames")]

    if context and hasattr(context, "scene") and context.scene:
        props = context.scene.vggt_props
        for i in range(props.num_frames):
            items.append(
                (f"FRAME_{i}", f"Frame {i}", f"Show points from frame {i} only")
            )

    return items


def update_visualization_callback(self, context):
    """
    Callback function triggered when visualization parameters change.
    This triggers an update of the visualization in Blender.
    """
    # Only update if data is loaded
    if not self.is_loaded:
        return

    # Schedule the visualization update operator to run
    # This avoids calling operators during property updates which can cause issues
    if context.area:
        context.area.tag_redraw()
