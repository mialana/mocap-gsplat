
"""
Defines the UI panels for the VGGT Blender integration.
"""

import bpy
from bpy.types import Panel


class VGGT_PT_main_panel(Panel):
    """Main panel for VGGT integration in the 3D viewport sidebar."""
    
    bl_label = "VGGT Integration"
    bl_idname = "VGGT_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VGGT"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.vggt_props
        
        # Model settings section
        box = layout.box()
        box.label(text="Model Settings", icon='PREFERENCES')
        box.prop(props, "model_path")
        box.prop(props, "cache_directory")
        
        layout.separator()
        
        # Image loading section
        box = layout.box()
        box.label(text="Input Images", icon='IMAGE_DATA')
        box.prop(props, "images_directory")
        
        row = box.row(align=True)
        row.operator("vggt.load_images", text="Load Images", icon='FILEBROWSER')
        
        # Status display
        if props.num_frames > 0:
            box.label(text=f"Loaded: {props.num_frames} frames", icon='INFO')
        
        layout.separator()
        
        # Inference section
        box = layout.box()
        box.label(text="Reconstruction", icon='VIEW_CAMERA')
        
        row = box.row(align=True)
        row.scale_y = 1.5
        row.operator("vggt.run_inference", text="Run VGGT Inference", icon='PLAY')
        
        # Status after inference
        if props.is_loaded:
            box.label(text=f"Points: {props.num_points:,}", icon='MESH_DATA')
        
        layout.separator()
        
        # Clear scene button
        row = layout.row()
        row.operator("vggt.clear_scene", text="Clear VGGT Data", icon='TRASH')


class VGGT_PT_parameters_panel(Panel):
    """Panel for VGGT parameter tuning options."""
    
    bl_label = "Parameter Tuning"
    bl_idname = "VGGT_PT_parameters_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VGGT"
    bl_parent_id = "VGGT_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return context.scene.vggt_props.is_loaded
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.vggt_props
        
        # Prediction mode
        layout.prop(props, "prediction_mode")
        
        layout.separator()
        
        # Confidence threshold
        layout.prop(props, "conf_thres", slider=True)
        
        layout.separator()
        
        # Frame selection
        layout.prop(props, "frame_filter")
        
        layout.separator()
        
        # Masking options
        box = layout.box()
        box.label(text="Filtering Options", icon='FILTER')
        
        col = box.column(align=True)
        col.prop(props, "mask_black_bg")
        col.prop(props, "mask_white_bg")
        col.prop(props, "mask_sky")
        
        layout.separator()
        
        # Visualization options
        box = layout.box()
        box.label(text="Visualization", icon='HIDE_OFF')
        
        col = box.column(align=True)
        col.prop(props, "show_cameras")
        col.prop(props, "point_size")
        col.prop(props, "camera_scale")
        
        layout.separator()
        
        # Update button
        row = layout.row()
        row.scale_y = 1.2
        row.operator("vggt.update_visualization", text="Update Visualization", icon='FILE_REFRESH')


class VGGT_PT_export_panel(Panel):
    """Panel for Gaussian splatting export options."""
    
    bl_label = "Gaussian Splatting Export"
    bl_idname = "VGGT_PT_export_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VGGT"
    bl_parent_id = "VGGT_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return context.scene.vggt_props.is_loaded
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.vggt_props
        
        # Export path
        layout.prop(props, "export_path")
        
        layout.separator()
        
        # Export options
        box = layout.box()
        box.label(text="Export Options", icon='EXPORT')
        
        col = box.column(align=True)
        col.prop(props, "include_confidence")
        col.prop(props, "include_colors")
        
        layout.separator()
        
        # Export button
        row = layout.row()
        row.scale_y = 1.5
        row.operator("vggt.export_gaussian_splat", text="Export PLY", icon='EXPORT')


# Registration handled by __init__.py
