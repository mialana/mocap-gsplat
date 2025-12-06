# VGGT Blender Addon

A Blender addon for integrating the [VGGT (Visual Geometry Grounded Transformer)](https://github.com/facebookresearch/vggt) model with support for parameter tuning and Gaussian splatting export.

## Features

- **VGGT Model Integration**: Load and process images through the VGGT model directly within Blender
- **Parameter Tuning**: Real-time control over visualization parameters matching the VGGT Gradio demo:
  - Confidence threshold filtering
  - Background masking (black/white)
  - Sky segmentation filtering
  - Prediction mode selection (Depthmap/Camera vs Pointmap branch)
- **3D Visualization**: Point cloud and camera visualization in Blender's 3D viewport
- **Gaussian Splatting Export**: Export to PLY format compatible with Gaussian splatting pipelines

## Requirements

### Blender Version
- Blender 3.6.0 or later

### Python Dependencies
The addon requires the following Python packages to be available in Blender's Python environment:

- PyTorch (with CUDA support recommended)
- NumPy
- VGGT model package

### Setting Up Dependencies

1. **Install PyTorch in Blender's Python**:
   ```bash
   # Find Blender's Python executable (example path)
   /path/to/blender/3.6/python/bin/python -m pip install torch torchvision
   ```

2. **Install VGGT Dependencies**:
   ```bash
   /path/to/blender/3.6/python/bin/python -m pip install huggingface_hub einops safetensors
   ```

3. **Set up VGGT package**:
   Ensure the VGGT repository is accessible. The addon will look for VGGT in the standard Python path or you can configure the path.

## Installation

### Method 1: Install as Blender Addon (Recommended)

1. Package the addon:
   ```bash
   cd src/blender_addon
   zip -r vggt_blender_addon.zip .
   ```

2. In Blender:
   - Go to Edit > Preferences > Add-ons
   - Click "Install..."
   - Select the `vggt_blender_addon.zip` file
   - Enable the addon "VGGT Integration"

### Method 2: Development Installation

1. Create a symlink in Blender's addons folder:
   ```bash
   ln -s /path/to/mocap-gsplat/src/blender_addon ~/.config/blender/3.6/scripts/addons/vggt_integration
   ```

2. In Blender, enable the addon in Preferences > Add-ons

## Usage

### Basic Workflow

1. **Open the VGGT Panel**: In the 3D Viewport, press `N` to open the sidebar, then select the "VGGT" tab.

2. **Configure Model** (Optional):
   - Set the model path (default: `facebook/VGGT-1B` for HuggingFace model)
   - Set cache directory for model weights

3. **Load Images**:
   - Set the "Images Directory" to a folder containing your input images
   - Click "Load Images" to scan the directory

4. **Run Inference**:
   - Click "Run VGGT Inference" to process the images
   - Wait for the model to generate predictions (this may take a moment)

5. **Adjust Parameters** (Expand "Parameter Tuning" panel):
   - **Prediction Mode**: Choose between Depthmap/Camera branch or Pointmap branch
   - **Confidence Threshold**: Adjust to filter out low-confidence points (0-100%)
   - **Frame Filter**: Select specific frames or show all
   - **Filtering Options**: Enable black/white background or sky masking
   - Click "Update Visualization" to apply changes

6. **Export** (Expand "Gaussian Splatting Export" panel):
   - Set the export path
   - Choose whether to include confidence scores and colors
   - Click "Export PLY" to save the point cloud

### Parameter Descriptions

| Parameter | Description | Default |
|-----------|-------------|---------|
| Confidence Threshold | Percentage of low-confidence points to filter out | 50% |
| Prediction Mode | Depthmap/Camera uses depth and pose estimation; Pointmap uses direct regression | Depthmap/Camera |
| Filter Black Background | Remove points where RGB sum < 16 | Off |
| Filter White Background | Remove points where all RGB > 240 | Off |
| Filter Sky | Apply sky segmentation to remove sky points | Off |
| Show Cameras | Display camera frustums in the scene | On |

### Export Format

The PLY export is compatible with Gaussian splatting pipelines and includes:

- **Required**: 3D point positions (x, y, z)
- **Optional**: RGB colors (red, green, blue as uint8)
- **Optional**: Confidence scores (float)

## API Reference

### VGGTPredictions Class

The core data class for storing VGGT model outputs:

```python
from vggt_interface import VGGTPredictions

predictions = VGGTPredictions(
    world_points=...,           # Nx3 pointmap regression output
    world_points_conf=...,      # Nx1 confidence for pointmap
    world_points_from_depth=..., # Nx3 points from depth unprojection
    depth_conf=...,             # Nx1 confidence for depth
    images=...,                 # SxHxWx3 input images
    extrinsic=...,              # Sx3x4 camera extrinsic matrices
    intrinsic=...,              # Sx3x3 camera intrinsic matrices
)

# Get filtered points with parameters
points, colors, conf = predictions.get_filtered_points(
    conf_thres=50.0,
    mask_black_bg=False,
    mask_white_bg=False,
    prediction_mode='DEPTHMAP_CAMERA',
    frame_filter='ALL'
)
```

### Export Functions

```python
from export import export_gaussian_splat_ply, export_cameras_json

# Export point cloud to PLY
export_gaussian_splat_ply(
    filepath="output.ply",
    points=points,
    colors=colors,
    confidence=conf
)

# Export camera parameters to JSON
export_cameras_json(
    filepath="cameras.json",
    extrinsic_matrices=extrinsics,
    intrinsic_matrices=intrinsics,
    image_width=1920,
    image_height=1080
)
```

## Troubleshooting

### Common Issues

1. **"Missing dependencies" error**:
   - Ensure PyTorch and VGGT are installed in Blender's Python environment
   - Check that the VGGT submodule is initialized: `git submodule update --init`

2. **CUDA out of memory**:
   - Reduce the number of input images
   - Use a machine with more GPU memory
   - The model requires approximately 8GB+ GPU memory for inference

3. **Slow visualization**:
   - Reduce point count using a higher confidence threshold
   - Decrease point size for faster rendering

4. **No points visible**:
   - Check that confidence threshold isn't too high
   - Verify that masking options aren't filtering all points
   - Ensure images were properly loaded and processed

### Debug Mode

To enable verbose logging, set the `VGGT_DEBUG` environment variable:
```bash
VGGT_DEBUG=1 blender
```

## License

This addon is part of the mocap-gsplat project. See the main repository for license information.

## Credits

- VGGT model by [Meta AI Research](https://github.com/facebookresearch/vggt)
- 3D Gaussian Splatting by [INRIA](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
