"""
Defines the operators for the MOSPLAT Blender integration, including:
- Installing pip packages
- Installing VGGT model
- Loading images
- Running VGGT inference
- Updating visualization
- Exporting to Gaussian splatting format
"""

import os
import sys
import subprocess
import threading
import importlib
import shutil
import bpy
from bpy.types import Operator
from bpy.props import StringProperty

from .vggt_interface import VGGTInterface, VGGTPredictions
from . import constants
from . import helpers
from . import install


# Prediction mode constants
PREDICTION_MODE_POINTMAP = "POINTMAP"
PREDICTION_MODE_DEPTHMAP = "DEPTHMAP_CAMERA"

# Global storage for MOSPLAT data between operator calls
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


class MOSPLAT_OT_install_packages(Operator):
    """Install required pip packages for MOSPLAT."""

    bl_idname = "mosplat.install_packages"
    bl_label = "Install Dependencies"
    bl_description = (
        "Install required Python packages and clone VGGT GitHub repository."
    )
    bl_options = {"REGISTER"}

    _timer = None
    _thread = None

    def modal(self, context, event):
        props = context.scene.mosplat_props

        if event.type == "TIMER":
            # Check if installation thread is complete
            if self._thread is not None and not self._thread.is_alive():
                # Installation complete
                context.window_manager.event_timer_remove(self._timer)
                props.installation_in_progress = False

                # Check if installation was successful
                if props.packages_installed:
                    props.installation_message = "All packages installed successfully!"
                    self.report({"INFO"}, "Package installation completed successfully")
                else:
                    props.installation_message = (
                        "Package installation failed. Check console for details."
                    )
                    self.report({"ERROR"}, "Package installation failed")

                # Redraw panel
                for area in context.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()

                return {"FINISHED"}

        return {"PASS_THROUGH"}

    def execute(self, context):
        props = context.scene.mosplat_props

        if props.installation_in_progress:
            self.report({"WARNING"}, "Installation already in progress")
            return {"CANCELLED"}

        # Mark installation as in progress
        props.installation_in_progress = True
        props.installation_message = "Starting package installation..."
        props.installation_progress = 10.0

        # Start installation in background thread
        self._thread = threading.Thread(
            target=self._install_packages_thread, args=(context,), daemon=True
        )
        self._thread.start()

        # Set up modal timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.25, window=context.window)
        wm.modal_handler_add(self)

        return {"RUNNING_MODAL"}

    def _install_packages_thread(self, context):
        """Background thread for package installation and VGGT repository setup."""
        props = context.scene.mosplat_props

        try:
            print(
                f"Blender's Python executable located at {install.get_blender_python_path()}."
            )
            # Ensure pip is available
            props.installation_message = "Ensuring pip is installed..."
            install.ensure_pip()

            props.installation_progress = 20.0

            # Get modules path
            modules_path = helpers.resolve_script_file_path(constants.ADDON_SUBPATH)
            print(f"Adding modules path at {modules_path} to PATH.")
            install.append_to_path(modules_path)

            for _, pkg in enumerate(install.REQUIRED_PACKAGES):
                # Check if already installed
                print(f"Checking {pkg.module_name}...")
                installed = install.ensure_package_installed(pkg.module_name)

                # Install if needed
                if installed:
                    print(f"{pkg.module_name} already successfully installed.")
                else:
                    try:
                        install.install_package(
                            pkg.module_name, pkg.pip_spec, modules_path
                        )
                    except Exception as e:
                        props.packages_installed = False
                        props.installation_in_progress = False
                        raise

                props.installation_progress += 5.0

            print("All packages installed. Setting up VGGT repository...")

            # Install VGGT repository
            try:
                # Check if already installed
                print("Checking for existing VGGT installation...")

                vggt_already_installed = install.ensure_package_installed("vggt")

                if vggt_already_installed:
                    print("vggt already successfully installed")
                else:
                    # Import git module
                    print("Preparing to clone VGGT repository...")

                    if not install.ensure_package_installed("git"):
                        print(f"Error using installed GitPython: {e}")
                        props.packages_installed = False
                        props.installation_in_progress = False
                        return
                    
                    git = importlib.import_module("git")

                    # Clone VGGT repository
                    vggt_repo_dir = os.path.join(
                        modules_path, constants.VGGT_REPO_SUBPATH
                    )

                    props.installation_message = f"Cloning VGGT repository..."
                    print("Cloning VGGT repository...")

                    try:
                        if os.path.isdir(vggt_repo_dir):
                            shutil.rmtree(vggt_repo_dir)
                    except Exception as e:
                        print(f"Error deleting previous installation of VGGT: {e}")
                        raise

                    try:
                        git.Repo.clone_from(
                            "https://github.com/facebookresearch/vggt.git",
                            vggt_repo_dir,
                            multi_options=[
                                "--no-tags",  # disable Git packfiles
                                "--depth=1",
                                "--recurse-submodules",
                            ],
                        )
                        print("Clone success.")
                    except Exception as e:
                        props.installation_message = f"Failed to clone VGGT repository"
                        print(f"Error cloning VGGT from GitHub: {e}")
                        props.packages_installed = False
                        props.installation_in_progress = False
                        raise

                    props.installation_progress += 10.0

                    # Install as Python module
                    props.installation_message = "Installing VGGT as Python module..."
                    print("Installing VGGT as Python module...")

                    try:
                        subprocess.check_call(
                            [
                                install.get_blender_python_path(),
                                "-m",
                                "pip",
                                "install",
                                "--force-reinstall",
                                "--quiet",
                                "--upgrade",
                                "-e",
                                vggt_repo_dir,
                            ]
                        )
                        install.append_to_path(vggt_repo_dir) # append to path
                    except subprocess.CalledProcessError as e:
                        props.installation_message = (
                            "Failed to install VGGT as Python module"
                        )
                        print(f"Failed to install vggt as Python module. Error: {e}")
                        props.packages_installed = False
                        props.installation_in_progress = False
                        return
                    except Exception as e:
                        props.installation_message = f"Failed to install VGGT: {str(e)}"
                        print(f"Error installing VGGT: {e}")
                        props.packages_installed = False
                        props.installation_in_progress = False
                        return

            except Exception as e:
                props.installation_message = (
                    f"Error setting up VGGT repository: {str(e)}"
                )
                print(f"VGGT repository setup error: {e}")
                props.packages_installed = False
                props.installation_in_progress = False
                return

            # All dependencies installed
            props.installation_progress = 100.0
            props.installation_message = "All dependencies installed successfully!"
            props.packages_installed = True

        except Exception as e:
            props.installation_message = f"Installation error: {str(e)}"
            print(f"Package installation error: {e}")
            props.packages_installed = False
            props.installation_in_progress = False


class MOSPLAT_OT_install_model(Operator):
    """Install VGGT model weights from HuggingFace."""

    bl_idname = "mosplat.install_vggt_model"
    bl_label = "Install VGGT Model"
    bl_description = "Install VGGT model weights from HuggingFace"
    bl_options = {"REGISTER"}

    _timer = None
    _thread = None

    def modal(self, context, event):
        props = context.scene.mosplat_props

        if event.type == "TIMER":
            # Check if installation thread is complete
            if self._thread is not None and not self._thread.is_alive():
                # Installation complete
                context.window_manager.event_timer_remove(self._timer)
                props.installation_in_progress = False

                # Check if installation was successful
                if props.vggt_model_installed:
                    props.installation_message = "VGGT model installed successfully!"
                    self.report(
                        {"INFO"}, "VGGT model installation completed successfully"
                    )
                else:
                    props.installation_message = (
                        "VGGT model installation failed. Check console for details."
                    )
                    self.report({"ERROR"}, "VGGT model installation failed")

                # Redraw panel
                for area in context.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()

                return {"FINISHED"}

        return {"PASS_THROUGH"}

    def execute(self, context):
        props = context.scene.mosplat_props

        if props.installation_in_progress:
            self.report({"WARNING"}, "Installation already in progress")
            return {"CANCELLED"}

        if not props.packages_installed:
            self.report({"ERROR"}, "Please install dependencies first")
            return {"CANCELLED"}

        # Mark installation as in progress
        props.installation_in_progress = True
        props.installation_message = "Starting VGGT model installation..."
        props.installation_progress = 0.0

        # Start installation in background thread
        self._thread = threading.Thread(
            target=self._install_model_thread, args=(context,), daemon=True
        )
        self._thread.start()

        # Set up modal timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.25, window=context.window)
        wm.modal_handler_add(self)

        return {"RUNNING_MODAL"}

    def _install_model_thread(self, context):
        """Background thread for downloading VGGT model weights from HuggingFace."""
        props = context.scene.mosplat_props

        try:
            props.installation_message = "Preparing to install VGGT model..."
            props.installation_progress = 10.0

            # Import required modules
            try:
                import torch
                from vggt.models.vggt import VGGT
            except ImportError as e:
                props.installation_message = (
                    "VGGT module not found. Please install dependencies first."
                )
                print(f"Import error: {e}")
                props.vggt_model_installed = False
                props.installation_in_progress = False
                return

            props.installation_progress = 20.0

            # Get model path and cache directory from properties
            model_path = props.model_path if props.model_path else "facebook/VGGT-1B"

            props.installation_message = f"Installing model from {model_path}..."
            print(f"Installing model from {model_path}...")
            props.installation_progress = 30.0

            # Download and initialize the model (this triggers HuggingFace download)
            try:
                # Get the VGGT interface
                interface = get_vggt_interface()
                props.installation_progress = 40.0

                # Configure the interface
                cache_dir = (
                    props.cache_directory
                    if os.path.isdir(props.cache_directory)
                    else os.path.expanduser(".cache/vggt/")
                )
                interface.initialize_model(props.model_path, cache_dir)

                props.installation_progress = 90.0
                props.installation_message = (
                    "Model installed successfully. Verifying..."
                )

            except Exception as e:
                props.installation_message = f"Failed to install model from HuggingFace"
                print(f"Error installing model: {e}")
                props.vggt_model_installed = False
                props.installation_in_progress = False
                return

            # Complete
            props.installation_progress = 100.0
            props.installation_message = "VGGT model installed successfully!"
            print("VGGT model installed successfully!")
            props.vggt_model_installed = True

        except Exception as e:
            props.installation_message = f"Model installation error: {str(e)}"
            print(f"VGGT model installation error: {e}")
            props.vggt_model_installed = False
            props.installation_in_progress = False


class MOSPLAT_OT_load_images(Operator):
    """Load images from the specified directory for VGGT processing."""

    bl_idname = "mosplat.load_images"
    bl_label = "Load Images"
    bl_description = "Load images from the specified directory"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.mosplat_props

        if not props.images_directory:
            self.report({"ERROR"}, "Please specify an images directory")
            return {"CANCELLED"}

        images_dir = bpy.path.abspath(props.images_directory)

        if not os.path.isdir(images_dir):
            self.report({"ERROR"}, f"Directory not found: {images_dir}")
            return {"CANCELLED"}

        # Find image files in the directory
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = []

        for filename in sorted(os.listdir(images_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_files.append(filename)

        if not image_files:
            self.report({"ERROR"}, f"No image files found in: {images_dir}")
            return {"CANCELLED"}

        # Update properties
        props.num_cameras = len(image_files)

        self.report({"INFO"}, f"Found {len(image_files)} images")
        return {"FINISHED"}


class MOSPLAT_OT_run_inference(Operator):
    """Run VGGT inference on the loaded images."""

    bl_idname = "mosplat.run_inference"
    bl_label = "Run VGGT Inference"
    bl_description = "Run VGGT model inference on the loaded images"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.mosplat_props

        if not props.images_directory:
            self.report({"ERROR"}, "Please specify an images directory")
            return {"CANCELLED"}

        images_dir = bpy.path.abspath(props.images_directory)

        if not os.path.isdir(images_dir):
            self.report({"ERROR"}, f"Directory not found: {images_dir}")
            return {"CANCELLED"}

        try:
            # Get the VGGT interface
            interface = get_vggt_interface()

            # Run inference
            self.report({"INFO"}, "Running VGGT inference... This may take a moment.")
            predictions = interface.run_inference(images_dir)

            # Store predictions
            set_predictions(predictions)

            # Update properties
            props.is_loaded = True
            props.num_cameras = predictions.num_cameras
            props.num_points = predictions.get_total_points()

            print("Model inference complete and predictions stored")

            # Create initial visualization
            self._create_visualization(context, predictions, props)

            print("Point cloud creation completion")

            self.report(
                {"INFO"},
                f"VGGT inference complete. Generated {props.num_points:,} points.",
            )
            return {"FINISHED"}

        except ImportError as e:
            self.report({"ERROR"}, f"Missing dependencies: {e}")
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"VGGT inference failed: {e}")
            return {"CANCELLED"}

    def _create_visualization(self, context, predictions, props):
        """Create the initial Blender visualization from predictions."""
        from .visualization import create_point_cloud, create_cameras

        # Get filtered points based on current settings
        prediction_mode = (
            PREDICTION_MODE_POINTMAP
            if props.prediction_mode == "POINTMAP"
            else PREDICTION_MODE_DEPTHMAP
        )
        points, colors, confidence = predictions.get_filtered_points(
            conf_thres=props.conf_thres,
            mask_black_bg=props.mask_black_bg,
            mask_white_bg=props.mask_white_bg,
            prediction_mode=prediction_mode,
            frame_filter=props.camera_filter,
        )

        print(f"POINTS shape {points.shape} and datatype {points.dtype}.")
        print(f"COLORS shape {colors.shape} and datatype {colors.dtype}.")
        print(f"CONFIDENCE shape {confidence.shape} and datatype {confidence.dtype}.")

        # Create point cloud
        create_point_cloud(
            points, colors, name="MOSPLAT_PointCloud", point_size=confidence
        )

        # Create cameras if enabled
        if props.show_cameras:
            create_cameras(
                predictions.extrinsic,
                name_prefix="MOSPLAT_Camera",
                scale=props.camera_scale,
            )


class MOSPLAT_OT_update_visualization(Operator):
    """Update the visualization based on current parameter settings."""

    bl_idname = "mosplat.update_visualization"
    bl_label = "Update Visualization"
    bl_description = "Update the 3D visualization with current parameter settings"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.mosplat_props.is_loaded

    def execute(self, context):
        props = context.scene.mosplat_props
        predictions = get_predictions()

        if predictions is None:
            self.report(
                {"ERROR"}, "No VGGT predictions available. Run inference first."
            )
            return {"CANCELLED"}

        try:
            from .visualization import (
                remove_mosplat_objects,
                create_point_cloud,
                create_cameras,
            )

            # Remove existing MOSPLAT objects
            remove_mosplat_objects()

            # Get filtered points based on current settings
            prediction_mode = (
                PREDICTION_MODE_POINTMAP
                if props.prediction_mode == "POINTMAP"
                else PREDICTION_MODE_DEPTHMAP
            )
            points, colors, confidence = predictions.get_filtered_points(
                conf_thres=props.conf_thres,
                mask_black_bg=props.mask_black_bg,
                mask_white_bg=props.mask_white_bg,
                prediction_mode=prediction_mode,
                frame_filter=props.frame_filter,
            )

            # Update point count
            props.num_points = len(points)

            # Create point cloud
            create_point_cloud(
                points, colors, name="MOSPLAT_PointCloud", point_size=props.point_size
            )

            # Create cameras if enabled
            if props.show_cameras:
                create_cameras(
                    predictions.extrinsic,
                    name_prefix="MOSPLAT_Camera",
                    scale=props.camera_scale,
                )

            self.report(
                {"INFO"}, f"Visualization updated with {props.num_points:,} points"
            )
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to update visualization: {e}")
            return {"CANCELLED"}


class MOSPLAT_OT_export_gaussian_splat(Operator):
    """Export MOSPLAT data to Gaussian splatting-compatible PLY format."""

    bl_idname = "mosplat.export_gaussian_splat"
    bl_label = "Export Gaussian Splat"
    bl_description = "Export point cloud to Gaussian splatting-compatible PLY format"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.mosplat_props.is_loaded

    def execute(self, context):
        props = context.scene.mosplat_props
        predictions = get_predictions()

        if predictions is None:
            self.report(
                {"ERROR"}, "No VGGT predictions available. Run inference first."
            )
            return {"CANCELLED"}

        export_path = bpy.path.abspath(props.export_path)

        if not export_path:
            self.report({"ERROR"}, "Please specify an export path")
            return {"CANCELLED"}

        try:
            from .export import export_gaussian_splat_ply

            # Get filtered points based on current settings
            prediction_mode = (
                PREDICTION_MODE_POINTMAP
                if props.prediction_mode == "POINTMAP"
                else PREDICTION_MODE_DEPTHMAP
            )
            points, colors, confidence = predictions.get_filtered_points(
                conf_thres=props.conf_thres,
                mask_black_bg=props.mask_black_bg,
                mask_white_bg=props.mask_white_bg,
                prediction_mode=prediction_mode,
                frame_filter=props.frame_filter,
            )

            # Export to PLY
            export_gaussian_splat_ply(
                export_path,
                points,
                colors=colors if props.include_colors else None,
                confidence=confidence if props.include_confidence else None,
            )

            self.report({"INFO"}, f"Exported {len(points):,} points to {export_path}")
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Export failed: {e}")
            return {"CANCELLED"}


class MOSPLAT_OT_clear_scene(Operator):
    """Clear all MOSPLAT data and objects from the scene."""

    bl_idname = "mosplat.clear_scene"
    bl_label = "Clear MOSPLAT Data"
    bl_description = "Remove all MOSPLAT objects and clear stored data"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.mosplat_props

        try:
            from .visualization import remove_mosplat_objects

            # Remove MOSPLAT objects from scene
            remove_mosplat_objects()

            # Clear predictions
            clear_predictions()

            # Reset properties
            props.is_loaded = False
            props.num_cameras = 0
            props.num_points = 0

            self.report({"INFO"}, "MOSPLAT data cleared")
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to clear scene: {e}")
            return {"CANCELLED"}
