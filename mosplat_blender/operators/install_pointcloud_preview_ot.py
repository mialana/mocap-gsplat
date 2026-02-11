from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Optional, Set, Tuple, cast

from infrastructure.mixins import CtxPackage
from infrastructure.schemas import SavedTensorFileName, UnexpectedError
from operators.base_ot import MosplatOperatorBase

if TYPE_CHECKING:
    from bpy.types import (
        Material,
        Mesh,
        Modifier,
        Node,
        NodeLinks,
        Nodes,
        NodeTree,
        Object,
        Scene,
    )

MESH_NAME = "pc_mesh"
PLAYER_NAME = "PointCloudPlaybackManager"

PLY_FILE_FORMATTER = None


class Mosplat_OT_install_pointcloud_preview(MosplatOperatorBase):
    @classmethod
    def _contexted_poll(cls, pkg):
        props = pkg.props
        cls._poll_error_msg_list.extend(props.is_valid_media_directory_poll_result)
        cls._poll_error_msg_list.extend(props.frame_range_poll_result(pkg.prefs))

        return len(cls._poll_error_msg_list) == 0

    def _contexted_invoke(self, pkg, event):
        global PLY_FILE_FORMATTER

        prefs = pkg.prefs
        props = pkg.props

        self._frame_range: Tuple[int, int] = props.frame_range_
        self._exported_file_formatter: str = props.exported_file_formatter(prefs)

        PLY_FILE_FORMATTER = partial(
            self._exported_file_formatter.format,
            file_name=SavedTensorFileName.POINTCLOUD,
            file_ext="ply",
        )

        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):

        self.install(pkg)

        return "RUNNING_MODAL"

    def install(self, pkg):
        import bpy

        self.ensure_player(pkg)
        self.setup_geometry_nodes()
        self.setup_material()

        # prevent duplicate registration of handler
        for handler in bpy.app.handlers.frame_change_pre:
            if handler.__name__ == "on_frame_change":
                bpy.app.handlers.frame_change_pre.remove(handler)

        bpy.app.handlers.frame_change_pre.append(on_frame_change)

        self.logger.debug("Installation complete.")

    def ensure_player(self, pkg: CtxPackage):
        import bpy

        scene = pkg.context.scene
        assert scene
        obj: Optional[Object] = bpy.data.objects.get(PLAYER_NAME)
        if obj is None:
            mesh: Mesh = bpy.data.meshes.new(MESH_NAME)
            obj = bpy.data.objects.new(PLAYER_NAME, mesh)
            scene.collection.objects.link(obj)

        self._obj: Object = obj

        self.logger.debug("Player ensured.")

    def setup_geometry_nodes(self):
        import bpy

        # Create GN modifier if missing
        mod: Modifier = self._obj.modifiers.get(
            "PointCloudGN"
        ) or self._obj.modifiers.new("PointCloudGN", type="NODES")

        if hasattr(mod, "node_group") and getattr(mod, "node_group") is None:
            node_group: NodeTree = bpy.data.node_groups.new(
                "PointCloudGNGroup", "GeometryNodeTree"
            )
            setattr(mod, "node_group", node_group)
        else:
            node_group = getattr(mod, "node_group")

        nodes: Nodes = node_group.nodes
        links: NodeLinks = node_group.links

        nodes.clear()

        input_node: Node = nodes.new("NodeGroupInput")
        output_node: Node = nodes.new("NodeGroupOutput")
        mesh_to_points: Node = nodes.new("GeometryNodeMeshToPoints")
        setattr(mesh_to_points.inputs["Radius"], "default_value", 0.15)
        set_material: Node = nodes.new("GeometryNodeSetMaterial")

        input_node.location = (-400, 0)
        mesh_to_points.location = (-100, 0)
        set_material.location = (150, 0)
        output_node.location = (400, 0)

        assert node_group.interface

        if not node_group.interface.items_tree:
            # add geometry socket
            node_group.interface.new_socket(
                name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
            )
            node_group.interface.new_socket(
                name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
            )

        # create or get material
        mat: Material = bpy.data.materials.get(
            "PointCloudMaterial"
        ) or bpy.data.materials.new("PointCloudMaterial")
        mat.use_nodes = True

        setattr(set_material.inputs["Material"], "default_value", mat)

        # links
        links.new(input_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
        links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
        links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])

        self.logger.debug("Geometry Nodes setup complete.")

    def setup_material(self):
        import bpy

        mat: Material = bpy.data.materials.get(
            "PointCloudMaterial"
        ) or bpy.data.materials.new("PointCloudMaterial")

        assert mat.node_tree

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        attr: Node = nodes.new("ShaderNodeAttribute")
        setattr(attr, "attribute_name", "Col")

        bsdf: Node = nodes.new("ShaderNodeBsdfPrincipled")
        output: Node = nodes.new("ShaderNodeOutputMaterial")

        attr.location = (-400, 0)
        bsdf.location = (-100, 0)
        output.location = (200, 0)

        links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        self.logger.debug("Material setup complete.")


def on_frame_change(scene: Scene):
    import bpy

    obj: Optional[Object] = bpy.data.objects.get(PLAYER_NAME)
    if obj is None:
        return

    mesh = import_ply_mesh_for_frame(scene.frame_current)
    if mesh is None:
        return

    old_mesh: Mesh = cast(Mesh, obj.data)
    obj.data = mesh

    # remove old mesh datablock to prevent RAM memory growth
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def import_ply_mesh_for_frame(curr_frame: int) -> Mesh:
    import bpy
    import mathutils

    if not PLY_FILE_FORMATTER:
        raise UnexpectedError(f"Global PLY file formatter string no longer in scope.")

    ply_filepath: str = PLY_FILE_FORMATTER(frame_idx=curr_frame)

    before: Set[Object] = set(bpy.data.objects)

    bpy.ops.wm.ply_import(filepath=ply_filepath)

    after: Set[Object] = set(bpy.data.objects)
    created_obj: Object = list(after - before)[0]

    assert created_obj.data

    new_mesh: Mesh = cast(Mesh, created_obj.data.copy())

    PLY_TRANSFORM_MATRIX = mathutils.Matrix(
        (
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, -1, 0, 0),
            (0, 0, 0, 1),
        )
    )

    PLY_SCALE_MATRIX = mathutils.Matrix.Scale(100.0, 4)

    new_mesh.transform(PLY_SCALE_MATRIX @ PLY_TRANSFORM_MATRIX)
    new_mesh.update()

    bpy.data.objects.remove(created_obj, do_unlink=True)

    return new_mesh
