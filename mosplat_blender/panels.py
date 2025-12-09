
"""
Defines the UI panels for the MOSPLAT Blender integration.
"""

import bpy
from bpy.types import Panel


class MOSPLAT_PT_main_panel(Panel):
    """Main panel for MOSPLAT integration in the 3D viewport sidebar."""
    
    bl_label = "Mocap Gsplat Integration"
    bl_idname = "MOSPLAT_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MOSPLAT"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.mosplat_props
        
        # Check installation state
        if not props.packages_installed:
            # Show only install packages button
            box = layout.box()
            box.label(text="Setup Required", icon='INFO')
            
            if props.installation_in_progress:
                # Show installation progress
                col = box.column(align=True)
                col.label(text=props.installation_message)
                col.progress(
                    factor=props.installation_progress / 100.0,
                    type='BAR',
                    text=f"{props.installation_progress:.0f}%"
                )
            else:
                # Show install button
                row = box.row()
                row.scale_y = 1.5
                row.operator("mosplat.install_packages", text="Install Dependencies", icon='PREFERENCES')
            
            return
        
        box = layout.box()

        if props.installation_in_progress:
            # Show installation progress
            col = box.column(align=True)
            col.label(text=props.installation_message)
            col.progress(
                factor=props.installation_progress / 100.0,
                type='BAR',
                text=f"{props.installation_progress:.0f}%"
            )
        else:
            # Show model download options
            box.label(text="Model Settings")

            box.prop(props, "model_path", text="Name")
            box.prop(props, "cache_directory", text="Cache")
            
            row = box.row()
            row.scale_y = 1.5
            row.operator("mosplat.install_vggt_model", text="Install VGGT Model", icon='IMPORT')

        if not props.vggt_model_installed:
            return

        layout.separator()
        
        # Image loading section
        box = layout.box()
        box.label(text="Input Images", icon='IMAGE_DATA')
        box.prop(props, "images_directory", text="Images")
        
        row = box.row(align=True)
        row.operator("mosplat.load_images", text="Load Images", icon='FILEBROWSER')
        
        # Status display
        if props.num_cameras > 0:
            box.label(text=f"Loaded: {props.num_cameras} cameras", icon='INFO')
        
        layout.separator()
        
        # Inference section
        box = layout.box()
        box.label(text="Reconstruction", icon='VIEW_CAMERA')
        
        row = box.row(align=True)
        row.scale_y = 1.5
        row.operator("mosplat.run_inference", text="Run VGGT Inference", icon='PLAY')
        
        # Status after inference
        if props.is_loaded:
            box.label(text=f"Points: {props.num_points:,}", icon='MESH_DATA')
        
        layout.separator()
        
        # Clear scene button
        row = layout.row()
        row.operator("mosplat.clear_scene", text="Clear MOSPLAT Data", icon='TRASH')


class MOSPLAT_PT_parameters_panel(Panel):
    """Panel for VGGT parameter tuning options."""
    
    bl_label = "Parameter Tuning"
    bl_idname = "MOSPLAT_PT_parameters_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MOSPLAT"
    bl_parent_id = "MOSPLAT_PT_main_panel"
    
    @classmethod
    def poll(cls, context):
        props = context.scene.mosplat_props
        return props.vggt_model_installed
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.mosplat_props
        
        # Prediction mode
        layout.prop(props, "prediction_mode")
        
        layout.separator()
        
        # Confidence threshold
        layout.prop(props, "conf_thres", slider=True)
        
        layout.separator()
        
        # Frame selection
        layout.prop(props, "camera_filter")
        
        layout.separator()
        
        # Masking options
        box = layout.box()
        box.label(text="Filtering Options", icon='FILTER')
        
        col = box.column(align=True)
        col.prop(props, "mask_black_bg")
        col.prop(props, "mask_white_bg")
        col.prop(props, "mask_sky")
        col.prop(props, "show_cameras")
        
        layout.separator()
        
        # Update button
        row = layout.row()
        row.scale_y = 1.2
        row.operator("mosplat.update_visualization", text="Update Visualization", icon='FILE_REFRESH')


class MOSPLAT_PT_export_panel(Panel):
    """Panel for Gaussian splatting export options."""
    
    bl_label = "Gaussian Splatting Export"
    bl_idname = "MOSPLAT_PT_export_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "MOSPLAT"
    bl_parent_id = "MOSPLAT_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        props = context.scene.mosplat_props
        return props.vggt_model_installed and props.is_loaded
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.mosplat_props
        
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
        row.operator("mosplat.export_gaussian_splat", text="Export PLY", icon='EXPORT')
