from . import install

from .operators import (
    MOSPLAT_OT_install_packages,
    MOSPLAT_OT_install_model,
    MOSPLAT_OT_load_images,
    MOSPLAT_OT_run_inference,
    MOSPLAT_OT_update_visualization,
    MOSPLAT_OT_export_gaussian_splat,
    MOSPLAT_OT_clear_scene,
)
from .panels import MOSPLAT_PT_main_panel, MOSPLAT_PT_parameters_panel, MOSPLAT_PT_export_panel
from .properties import MOSPLATProperties

import bpy

bl_info = {
    "name": "MOSPLAT Blender Old",
    "blender": (4, 2, 0),
    "category": "Pipeline",
    "location": "View3D > Sidebar > MOSPLAT",
    "description": "Integrate MOSPLAT model for 3D reconstruction with Gaussian splatting support",
    "warning": "Requires external dependencies (PyTorch, MOSPLAT)",
    "category": "3D View",
}

classes = (
    MOSPLATProperties,
    MOSPLAT_OT_install_packages,
    MOSPLAT_OT_install_model,
    MOSPLAT_OT_load_images,
    MOSPLAT_OT_run_inference,
    MOSPLAT_OT_update_visualization,
    MOSPLAT_OT_export_gaussian_splat,
    MOSPLAT_OT_clear_scene,
    MOSPLAT_PT_main_panel,
    MOSPLAT_PT_parameters_panel,
    MOSPLAT_PT_export_panel,
)


def register():
    print("MOSPLAT Blender addon registration starting...")
    
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mosplat_props = bpy.props.PointerProperty(type=MOSPLATProperties)
    
    print("MOSPLAT Blender addon registration completed. Use the MOSPLAT panel to install dependencies.")


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError as e:
            print(f"Error during unregistration of {cls.__name__}: {e}")
    
    try:
        del bpy.types.Scene.mosplat_props
    except AttributeError as e:
        print(f"Error during unregistration of Scene.mosplat_props: {e}")
    
    print("MOSPLAT Blender addon unregistration completed")
