from . import install

from .operators import (
    VGGT_OT_install_packages,
    VGGT_OT_install_model,
    VGGT_OT_load_images,
    VGGT_OT_run_inference,
    VGGT_OT_update_visualization,
    VGGT_OT_export_gaussian_splat,
    VGGT_OT_clear_scene,
)
from .panels import VGGT_PT_main_panel, VGGT_PT_parameters_panel, VGGT_PT_export_panel
from .properties import VGGTProperties

import bpy

bl_info = {
    "name": "VGGT Blender",
    "blender": (4, 2, 0),
    "category": "Pipeline",
    "location": "View3D > Sidebar > VGGT",
    "description": "Integrate VGGT model for 3D reconstruction with Gaussian splatting support",
    "warning": "Requires external dependencies (PyTorch, VGGT)",
    "category": "3D View",
}

classes = (
    VGGTProperties,
    VGGT_OT_install_packages,
    VGGT_OT_install_model,
    VGGT_OT_load_images,
    VGGT_OT_run_inference,
    VGGT_OT_update_visualization,
    VGGT_OT_export_gaussian_splat,
    VGGT_OT_clear_scene,
    VGGT_PT_main_panel,
    VGGT_PT_parameters_panel,
    VGGT_PT_export_panel,
)


def register():
    print("VGGT Blender addon registration starting...")
    
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.vggt_props = bpy.props.PointerProperty(type=VGGTProperties)
    
    print("VGGT Blender addon registration completed. Use the VGGT panel to install dependencies.")


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError as e:
            print(f"Error during unregistration of {cls.__name__}: {e}")
    
    try:
        del bpy.types.Scene.vggt_props
    except AttributeError as e:
        print(f"Error during unregistration of Scene.vggt_props: {e}")
    
    print("VGGT Blender addon unregistration completed")
