# 3D Gaussian Splatting for Motion Capture Data

This is a repository that was created with the intent of building a **gaussian splatting** pipeline for 3D Motion Capture data.

To elaborate, motion capture data is often exported from **sparse view directions**, say 4-8 cameras directed towards an area of focus. This differs from the usual creation process of gaussian splatting, which would be taken from a single, moving camera, and **Structure-from-Motion** techniques would use the continuous video data to interpolate the structure (as the concept's name connotates).

## Requirements

- Blender 4.2+
- FFmpeg with shared library build
  - This project utilizes the new [https://github.com/meta-pytorch/torchcodec](module) 




## Installation

### Pre-packaged Releases

Visit the [Releases](https://github.com/mialana/mocap-gsplat/releases) page of this repository, where ZIP files corresponding to Windows, Linux, and MacOS

### From source
