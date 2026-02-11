from __future__ import annotations

from typing import Tuple

from operators.base_ot import MosplatOperatorBase


class Mosplat_OT_install_pointcloud_preview(MosplatOperatorBase):
    @classmethod
    def _contexted_poll(cls, pkg):
        props = pkg.props
        cls._poll_error_msg_list.extend(props.is_valid_media_directory_poll_result)
        cls._poll_error_msg_list.extend(props.frame_range_poll_result(pkg.prefs))

        return len(cls._poll_error_msg_list) == 0

    def _contexted_invoke(self, pkg, event):
        prefs = pkg.prefs
        props = pkg.props

        self._frame_range: Tuple[int, int] = props.frame_range_
        self._exported_file_formatter: str = props.exported_file_formatter(prefs)

        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):

        return "RUNNING_MODAL"


import os
from pathlib import Path

import bpy
import mathutils

# ---------------- CONFIG ----------------

ROOT_DIR = (
    Path.home() / "Desktop" / "caroline_shot" / "caroline_shot_OUTPUT"
)  # <-- change this
PLAYER_NAME = "PointCloudPlayer"
FRAME_OFFSET = 0  # adjust if needed

# ----------------------------------------


def frame_to_path(frame: int) -> str:
    f = frame + FRAME_OFFSET
    folder = f"frame_{f:04d}"
    return ROOT_DIR / folder / "pointcloud.ply"


def ensure_player():
    obj = bpy.data.objects.get(PLAYER_NAME)
    if obj is None:
        mesh = bpy.data.meshes.new("pc_mesh")
        obj = bpy.data.objects.new(PLAYER_NAME, mesh)
        bpy.context.scene.collection.objects.link(obj)
    return obj


def setup_geometry_nodes(obj):
    # Create GN modifier if missing
    mod = obj.modifiers.get("PointCloudGN")
    if mod is None:
        mod = obj.modifiers.new("PointCloudGN", type="NODES")

    if mod.node_group is None:
        node_group = bpy.data.node_groups.new("PointCloudGNGroup", "GeometryNodeTree")
        mod.node_group = node_group
    else:
        node_group = mod.node_group

    nodes = node_group.nodes
    links = node_group.links

    nodes.clear()

    # Nodes
    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    setattr(mesh_to_points.inputs["Radius"], "default_value", 0.15)
    set_material = nodes.new("GeometryNodeSetMaterial")

    # Layout (optional, aesthetic)
    input_node.location = (-400, 0)
    mesh_to_points.location = (-100, 0)
    set_material.location = (150, 0)
    output_node.location = (400, 0)

    if not node_group.interface.items_tree:
        # Add geometry socket
        node_group.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        node_group.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

    # Create or get material
    mat = bpy.data.materials.get("PointCloudMaterial")
    if mat is None:
        mat = bpy.data.materials.new("PointCloudMaterial")
        mat.use_nodes = True

    set_material.inputs["Material"].default_value = mat

    # Links
    links.new(input_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])

    print("Geometry Nodes setup complete.")


def import_ply_mesh(filepath):
    print("importing...")
    print(filepath)

    print("exists")

    before = set(bpy.data.objects)

    bpy.ops.wm.ply_import(filepath=str(filepath))

    after = set(bpy.data.objects)
    new_obj = list(after - before)[0]
    print(new_obj.data.color_attributes.keys())

    mesh = new_obj.data.copy()

    transform = mathutils.Matrix(
        (
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, -1, 0, 0),
            (0, 0, 0, 1),
        )
    )

    scale = mathutils.Matrix.Scale(100.0, 4)

    mesh.transform(scale @ transform)
    mesh.update()

    bpy.data.objects.remove(new_obj, do_unlink=True)
    print("complete")

    return mesh


def on_frame_change(scene):
    obj = bpy.data.objects.get(PLAYER_NAME)
    if obj is None:
        return

    path = frame_to_path(scene.frame_current)
    mesh = import_ply_mesh(path)
    if mesh is None:
        return

    old_mesh = obj.data
    obj.data = mesh

    # Remove old mesh datablock to prevent memory growth
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def setup_material():
    mat = bpy.data.materials.get("PointCloudMaterial")
    if mat is None:
        mat = bpy.data.materials.new("PointCloudMaterial")

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"  # change if needed

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")

    attr.location = (-400, 0)
    bsdf.location = (-100, 0)
    output.location = (200, 0)

    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    print("Material setup complete.")


def install():
    obj = ensure_player()
    setup_geometry_nodes(obj)
    setup_material()

    # prevent duplicate handler
    for h in bpy.app.handlers.frame_change_pre:
        if h.__name__ == "on_frame_change":
            bpy.app.handlers.frame_change_pre.remove(h)

    bpy.app.handlers.frame_change_pre.append(on_frame_change)

    print("PLY sequence player installed.")


install()
