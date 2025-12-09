
"""
Provides functions for exporting MOSPLAT data to Gaussian splatting-compatible formats.
The primary export format is PLY, which is widely supported by Gaussian splatting implementations.
"""

import json
import os
import struct

import numpy as np
from typing import Optional


def export_gaussian_splat_ply(
    filepath: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    scales: Optional[np.ndarray] = None,
    rotations: Optional[np.ndarray] = None,
    opacities: Optional[np.ndarray] = None,
    spherical_harmonics: Optional[np.ndarray] = None
) -> None:
    """
    Export point cloud data to a Gaussian splatting-compatible PLY file.
    
    This function exports 3D points with optional attributes that are commonly
    used in Gaussian splatting pipelines. The PLY format follows the conventions
    used by 3D Gaussian Splatting implementations.
    
    Args:
        filepath: Output PLY file path
        points: Nx3 array of 3D point coordinates
        colors: Optional Nx3 array of RGB colors (0-255 uint8)
        confidence: Optional Nx1 array of confidence scores
        normals: Optional Nx3 array of normal vectors
        scales: Optional Nx3 array of scale values for Gaussians
        rotations: Optional Nx4 array of quaternion rotations (wxyz)
        opacities: Optional Nx1 array of opacity values
        spherical_harmonics: Optional Nx(SH_dim) array of spherical harmonics coefficients
    """
    if len(points) == 0:
        raise ValueError("Cannot export empty point cloud")
    
    # Ensure output directory exists (handle case where filepath is just a filename)
    filepath_abs = os.path.abspath(filepath)
    filepath_dir = os.path.dirname(filepath_abs)
    # Only create directory if there's a meaningful directory path
    if filepath_dir and filepath_dir != filepath_abs:
        os.makedirs(filepath_dir, exist_ok=True)
    
    num_points = len(points)
    
    # Build property list for PLY header
    properties = [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
    ]
    
    # Add optional properties
    if normals is not None:
        properties.extend([
            ("nx", "float"),
            ("ny", "float"),
            ("nz", "float"),
        ])
    
    if colors is not None:
        properties.extend([
            ("red", "uchar"),
            ("green", "uchar"),
            ("blue", "uchar"),
        ])
    
    if confidence is not None:
        properties.append(("confidence", "float"))
    
    if opacities is not None:
        properties.append(("opacity", "float"))
    
    if scales is not None:
        properties.extend([
            ("scale_0", "float"),
            ("scale_1", "float"),
            ("scale_2", "float"),
        ])
    
    if rotations is not None:
        properties.extend([
            ("rot_0", "float"),
            ("rot_1", "float"),
            ("rot_2", "float"),
            ("rot_3", "float"),
        ])
    
    # Add spherical harmonics coefficients if provided
    if spherical_harmonics is not None:
        sh_dim = spherical_harmonics.shape[1] if spherical_harmonics.ndim > 1 else 1
        # For DC component
        properties.extend([
            ("f_dc_0", "float"),
            ("f_dc_1", "float"),
            ("f_dc_2", "float"),
        ])
        # For rest of SH coefficients (if more than DC)
        if sh_dim > 3:
            for i in range(sh_dim - 3):
                properties.append((f"f_rest_{i}", "float"))
    
    # Write PLY file
    with open(filepath, 'wb') as f:
        # Write header
        header = _create_ply_header(num_points, properties)
        f.write(header.encode('ascii'))
        
        # Write vertex data
        for i in range(num_points):
            # Position (always included)
            f.write(struct.pack('<fff', points[i, 0], points[i, 1], points[i, 2]))
            
            # Normal (optional)
            if normals is not None:
                f.write(struct.pack('<fff', normals[i, 0], normals[i, 1], normals[i, 2]))
            
            # Color (optional)
            if colors is not None:
                r, g, b = colors[i].astype(np.uint8)
                f.write(struct.pack('<BBB', r, g, b))
            
            # Confidence (optional)
            if confidence is not None:
                conf = float(confidence[i] if confidence.ndim == 1 else confidence[i, 0])
                f.write(struct.pack('<f', conf))
            
            # Opacity (optional)
            if opacities is not None:
                opacity = float(opacities[i] if opacities.ndim == 1 else opacities[i, 0])
                f.write(struct.pack('<f', opacity))
            
            # Scale (optional)
            if scales is not None:
                f.write(struct.pack('<fff', scales[i, 0], scales[i, 1], scales[i, 2]))
            
            # Rotation quaternion (optional)
            if rotations is not None:
                f.write(struct.pack('<ffff', 
                    rotations[i, 0], rotations[i, 1], 
                    rotations[i, 2], rotations[i, 3]))
            
            # Spherical harmonics (optional)
            if spherical_harmonics is not None:
                sh = spherical_harmonics[i]
                for val in sh[:3]:  # DC components
                    f.write(struct.pack('<f', val))
                if sh_dim > 3:  # Rest components
                    for val in sh[3:]:
                        f.write(struct.pack('<f', val))


def _create_ply_header(num_vertices: int, properties: list) -> str:
    """
    Create a PLY header string.
    
    Args:
        num_vertices: Number of vertices
        properties: List of (name, type) tuples
        
    Returns:
        PLY header string
    """
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Generated by MOSPLAT Blender Addon",
        "comment Gaussian Splatting compatible format",
        f"element vertex {num_vertices}",
    ]
    
    # Map Python types to PLY types
    type_map = {
        "float": "float",
        "uchar": "uchar",
        "int": "int",
    }
    
    for prop_name, prop_type in properties:
        ply_type = type_map.get(prop_type, "float")
        header_lines.append(f"property {ply_type} {prop_name}")
    
    header_lines.append("end_header\n")
    
    return '\n'.join(header_lines)


def export_cameras_json(
    filepath: str,
    extrinsic_matrices: np.ndarray,
    intrinsic_matrices: np.ndarray,
    image_width: int,
    image_height: int,
    image_names: Optional[list] = None
) -> None:
    """
    Export camera parameters to a JSON file compatible with Gaussian splatting pipelines.
    
    Args:
        filepath: Output JSON file path
        extrinsic_matrices: Sx3x4 array of camera extrinsic matrices
        intrinsic_matrices: Sx3x3 array of camera intrinsic matrices
        image_width: Width of input images
        image_height: Height of input images
        image_names: Optional list of image filenames
    """
    num_cameras = len(extrinsic_matrices)
    
    cameras = []
    for i in range(num_cameras):
        # Extract intrinsic parameters
        K = intrinsic_matrices[i]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Convert extrinsic to 4x4 and compute camera-to-world
        extrinsic_4x4 = np.eye(4)
        extrinsic_4x4[:3, :4] = extrinsic_matrices[i]
        camera_to_world = np.linalg.inv(extrinsic_4x4)
        
        # Extract rotation and translation
        R = camera_to_world[:3, :3]
        T = camera_to_world[:3, 3]
        
        camera_data = {
            "id": i,
            "img_name": image_names[i] if image_names else f"frame_{i:06d}.png",
            "width": image_width,
            "height": image_height,
            "position": T.tolist(),
            "rotation": R.tolist(),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }
        cameras.append(camera_data)
    
    with open(filepath, 'w') as f:
        json.dump(cameras, f, indent=2)


def export_colmap_format(
    output_dir: str,
    points: np.ndarray,
    colors: np.ndarray,
    extrinsic_matrices: np.ndarray,
    intrinsic_matrices: np.ndarray,
    image_width: int,
    image_height: int,
    image_names: Optional[list] = None
) -> None:
    """
    Export data in COLMAP-compatible format for use with Gaussian splatting pipelines.
    
    This creates the sparse reconstruction format expected by many Gaussian splatting
    implementations (e.g., 3D Gaussian Splatting).
    
    Args:
        output_dir: Output directory for COLMAP files
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
        extrinsic_matrices: Sx3x4 array of camera extrinsic matrices
        intrinsic_matrices: Sx3x3 array of camera intrinsic matrices
        image_width: Width of input images
        image_height: Height of input images
        image_names: Optional list of image filenames
    """
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    
    num_cameras = len(extrinsic_matrices)
    
    # Write cameras.txt (camera intrinsics)
    with open(os.path.join(sparse_dir, "cameras.txt"), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {num_cameras}\n")
        
        for i in range(num_cameras):
            K = intrinsic_matrices[i]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            # Using PINHOLE model
            f.write(f"{i+1} PINHOLE {image_width} {image_height} {fx} {fy} {cx} {cy}\n")
    
    # Write images.txt (camera extrinsics and image associations)
    with open(os.path.join(sparse_dir, "images.txt"), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {num_cameras}\n")
        
        for i in range(num_cameras):
            # COLMAP uses world-to-camera (extrinsic) format directly
            # The extrinsic matrix transforms world coordinates to camera coordinates
            R = extrinsic_matrices[i][:3, :3]
            T = extrinsic_matrices[i][:3, 3]
            
            # Convert rotation matrix to quaternion (for world-to-camera transform)
            quat = rotation_matrix_to_quaternion(R)
            
            img_name = image_names[i] if image_names else f"frame_{i:06d}.png"
            f.write(f"{i+1} {quat[0]} {quat[1]} {quat[2]} {quat[3]} {T[0]} {T[1]} {T[2]} {i+1} {img_name}\n")
            f.write("\n")  # Empty line for POINTS2D
    
    # Write points3D.txt
    with open(os.path.join(sparse_dir, "points3D.txt"), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        
        for i, (pt, col) in enumerate(zip(points, colors)):
            r, g, b = col.astype(int)
            f.write(f"{i+1} {pt[0]} {pt[1]} {pt[2]} {r} {g} {b} 0.0\n")


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion (w, x, y, z).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as (w, x, y, z) array
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])
