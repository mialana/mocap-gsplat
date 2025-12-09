"""
Provides functions for creating and managing 3D visualizations in Blender
from VGGT predictions.
"""

import colorsys

import bpy
import mathutils
import numpy as np
from typing import Optional, List


# Prefix for all MOSPLAT-generated objects
MOSPLAT_OBJECT_PREFIX = "MOSPLAT_"


def remove_vggt_objects():
    """
    Remove all MOSPLAT-generated objects from the Blender scene.

    This includes point clouds, cameras, and any other objects
    with the MOSPLAT prefix.
    """
    # Deselect all objects first
    bpy.ops.object.select_all(action="DESELECT")

    # Find and select all MOSPLAT objects
    objects_to_remove = []
    for obj in bpy.data.objects:
        if obj.name.startswith(MOSPLAT_OBJECT_PREFIX):
            objects_to_remove.append(obj)

    # Remove objects
    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Clean up orphan meshes and materials
    for mesh in bpy.data.meshes:
        if mesh.name.startswith(MOSPLAT_OBJECT_PREFIX) and mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    for mat in bpy.data.materials:
        if mat.name.startswith(MOSPLAT_OBJECT_PREFIX) and mat.users == 0:
            bpy.data.materials.remove(mat)


def create_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    name: str = "MOSPLAT_PointCloud",
    point_size: float = 0.01,
) -> bpy.types.Object:
    """
    Create a point cloud mesh in Blender from 3D points.

    Args:
        points: Nx3 array of 3D point coordinates
        colors: Optional Nx3 array of RGB colors (0-255)
        name: Name for the Blender object
        point_size: Size of individual points (for geometry nodes)

    Returns:
        The created Blender mesh object
    """

    num_points = points.shape[0]

    mesh = bpy.data.meshes.new(name) # Create mesh from vertices only
    obj = bpy.data.objects.new(name, mesh) # Create object from mesh

    if len(points) > 0:
        mesh.vertices.add(num_points)
        mesh.vertices.foreach_set("co", points.ravel())

        bpy.context.collection.objects.link(obj)

    mesh.update()
    mesh.validate()

    return obj


def _add_point_cloud_geometry_nodes(obj: bpy.types.Object, point_size: float):
    """
    Add geometry nodes modifier to render point cloud as visible spheres.

    Args:
        obj: The Blender object to modify
        point_size: Size of the point spheres
    """
    # Create geometry nodes modifier
    modifier = obj.modifiers.new(name="PointCloudDisplay", type="NODES")

    # Create new node tree
    node_tree = bpy.data.node_groups.new(
        name=f"{MOSPLAT_OBJECT_PREFIX}PointCloudNodes", type="GeometryNodeTree"
    )
    modifier.node_group = node_tree

    # Create nodes
    nodes = node_tree.nodes
    links = node_tree.links

    # Group input
    group_input = nodes.new("NodeGroupInput")
    group_input.location = (-300, 0)

    # Add input socket for geometry
    node_tree.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )

    # Mesh to Points node (convert vertices to points)
    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.location = (-100, 0)
    mesh_to_points.inputs["Radius"].default_value = point_size

    # Instance on Points node
    instance_on_points = nodes.new("GeometryNodeInstanceOnPoints")
    instance_on_points.location = (200, 0)

    # UV Sphere for point visualization
    uv_sphere = nodes.new("GeometryNodeMeshUVSphere")
    uv_sphere.location = (0, -150)
    uv_sphere.inputs["Segments"].default_value = 8
    uv_sphere.inputs["Rings"].default_value = 6
    uv_sphere.inputs["Radius"].default_value = point_size

    # Realize instances
    realize = nodes.new("GeometryNodeRealizeInstances")
    realize.location = (400, 0)

    # Group output
    group_output = nodes.new("NodeGroupOutput")
    group_output.location = (600, 0)

    # Add output socket for geometry
    node_tree.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    # Connect nodes
    links.new(group_input.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], instance_on_points.inputs["Points"])
    links.new(uv_sphere.outputs["Mesh"], instance_on_points.inputs["Instance"])
    links.new(instance_on_points.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], group_output.inputs["Geometry"])


def _create_point_cloud_material(obj: bpy.types.Object, name: str):
    """
    Create a material that uses vertex colors for the point cloud.

    Args:
        obj: The Blender object
        name: Base name for the material
    """
    mat_name = f"{name}_Material"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create nodes
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    # Try to get vertex color attribute
    attr = nodes.new("ShaderNodeAttribute")
    attr.location = (-300, 0)
    attr.attribute_name = "color"

    # Connect nodes
    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def create_cameras(
    extrinsic_matrices: np.ndarray, name_prefix: str = "MOSPLAT_Camera", scale: float = 0.1
) -> List[bpy.types.Object]:
    """
    Create camera objects in Blender from extrinsic matrices.

    Args:
        extrinsic_matrices: Sx3x4 array of camera extrinsic matrices (world-to-camera)
        name_prefix: Prefix for camera object names
        scale: Scale factor for camera visualization

    Returns:
        List of created camera objects
    """
    cameras = []
    num_cameras = len(extrinsic_matrices)

    for i, extrinsic in enumerate(extrinsic_matrices):
        # Convert extrinsic to 4x4 matrix
        extrinsic_4x4 = np.eye(4)
        extrinsic_4x4[:3, :4] = extrinsic

        # Compute camera-to-world transform (inverse of extrinsic)
        camera_to_world = np.linalg.inv(extrinsic_4x4)

        # Create camera data
        cam_data = bpy.data.cameras.new(name=f"{name_prefix}_{i}_Data")
        cam_data.display_size = scale

        # Create camera object
        cam_obj = bpy.data.objects.new(name=f"{name_prefix}_{i}", object_data=cam_data)

        # Set camera transform
        # Blender cameras look down -Z, so we need to apply a rotation
        blender_to_opencv = mathutils.Matrix(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )

        # Convert numpy array to Blender matrix
        transform_matrix = mathutils.Matrix(camera_to_world.tolist())

        # Apply coordinate system conversion
        cam_obj.matrix_world = transform_matrix @ blender_to_opencv

        # Link to scene
        bpy.context.collection.objects.link(cam_obj)

        # Set camera color based on index (for viewport display)
        hue = i / num_cameras
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        cam_obj.color = (*rgb, 1.0)

        cameras.append(cam_obj)

    return cameras


def get_mosplat_point_cloud() -> Optional[bpy.types.Object]:
    """
    Get the MOSPLAT point cloud object from the scene.

    Returns:
        The point cloud object if found, None otherwise
    """
    for obj in bpy.data.objects:
        if obj.name.startswith(f"{MOSPLAT_OBJECT_PREFIX}PointCloud"):
            return obj
    return None


def get_vggt_cameras() -> List[bpy.types.Object]:
    """
    Get all MOSPLAT camera objects from the scene.

    Returns:
        List of camera objects
    """
    cameras = []
    for obj in bpy.data.objects:
        if obj.name.startswith(f"{MOSPLAT_OBJECT_PREFIX}Camera") and obj.type == "CAMERA":
            cameras.append(obj)
    return sorted(cameras, key=lambda x: x.name)
