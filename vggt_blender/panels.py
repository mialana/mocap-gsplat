
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
                box.label(text="Install Python packages and VGGT repository")
                row = box.row()
                row.scale_y = 1.5
                row.operator("vggt.install_packages", text="Install Dependencies", icon='PREFERENCES')
            
            return
        
        # Packages installed, check if model is installed
        if not props.vggt_model_installed:
            # Show model download UI with cache path option
            box = layout.box()
            box.label(text="Model Download", icon='IMPORT')
            
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
                box.prop(props, "model_path")
                box.prop(props, "cache_directory")
                
                row = box.row()
                row.scale_y = 1.5
                row.operator("vggt.install_model", text="Initialize VGGT Model", icon='IMPORT')
            
            return
        
        # Everything installed, show normal UI
        # Model settings section

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
            box.prop(props, "model_path")
            box.prop(props, "cache_directory")
            
            row = box.row()
            row.scale_y = 1.5
            row.operator("vggt.install_model", text="Initialize VGGT Model", icon='IMPORT')

        layout.separator()
        
        # Image loading section
        box = layout.box()
        box.label(text="Input Images", icon='IMAGE_DATA')
        box.prop(props, "images_directory")
        
        row = box.row(align=True)
        row.operator("vggt.load_images", text="Load Images", icon='FILEBROWSER')
        
        # Status display
        if props.num_cameras > 0:
            box.label(text=f"Loaded: {props.num_cameras} cameras", icon='INFO')
        
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
        props = context.scene.vggt_props
        return props.vggt_model_installed
    
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
        col.prop(props, "show_cameras")
        
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
        props = context.scene.vggt_props
        return props.vggt_model_installed and props.is_loaded
    
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
