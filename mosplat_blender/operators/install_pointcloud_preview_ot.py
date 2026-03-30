from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set, Tuple, cast

from ..infrastructure.dl_ops import PointCloudTensors, load_safetensors
from ..infrastructure.macros import is_path_accessible
from ..infrastructure.mixins import CtxPackage
from ..infrastructure.schemas import (
    ExportedFileFormatterPartial,
    ExportedFileName,
    UserFacingError,
)
from ..interfaces.logging_interface import LoggingInterface
from .base_ot import MosplatOperatorBase

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
    from mathutils import Matrix

MESH_NAME = "pc_mesh"
PLAYER_NAME = "PointCloudPlaybackManager"
GN_MODIFIER_NAME = "PointCloudGN"
GN_TREE_NAME = "PointCloudGNGroup"
MATERIAL_NAME = "PointCloudMaterial"
DEFAULT_POINT_RADIUS = 0.1
CAMERA_COLLECTION_NAME = "pc_cameras"
CAMERA_NAME_FORMATTER = "pc_camera_{cam_id}"
CAMERA_DISPLAY_SIZE = 0.15

logger = LoggingInterface.configure_logger_instance(__name__)


class Mosplat_OT_install_point_cloud_preview(MosplatOperatorBase):
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

        self._ply_file_formatter = ExportedFileName.to_formatter(
            self._exported_file_formatter,
            file_name=ExportedFileName.POINT_CLOUD,
            file_ext="ply",
        )

        self._pct_file_formatter = ExportedFileName.to_formatter(
            self._exported_file_formatter,
            file_name=ExportedFileName.POINT_CLOUD_TENSORS,
            file_ext="safetensors",
        )

        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):

        self.install(pkg)

        scene = pkg.context.scene
        if scene:
            mosplat_on_frame_change_pc(
                scene,
                ply_file_formatter=self._ply_file_formatter,
                pct_file_formatter=self._pct_file_formatter,
            )  # initial execution

        return "RUNNING_MODAL"

    def install(self, pkg):
        import bpy

        self.ensure_player(pkg)
        self.setup_geometry_nodes()
        self.setup_material()

        # prevent duplicate registration of handler
        for idx, handler in enumerate(bpy.app.handlers.frame_change_pre):
            name = getattr(handler, "__name__", None) or (
                handler.func.__name__ if isinstance(handler, partial) else ""
            )
            if name == mosplat_on_frame_change_pc.__name__:
                bpy.app.handlers.frame_change_pre.pop(idx)

        bpy.app.handlers.frame_change_pre.append(
            partial(
                mosplat_on_frame_change_pc,
                ply_file_formatter=self._ply_file_formatter,
                pct_file_formatter=self._pct_file_formatter,
            )
        )

        self.logger.debug("Installation complete.")

    def ensure_player(self, pkg: CtxPackage):
        import bpy

        scene = pkg.context.scene
        assert scene
        assert scene.collection

        obj = bpy.data.objects.get(PLAYER_NAME)
        if obj is None:
            mesh: Mesh = bpy.data.meshes.new(MESH_NAME)
            obj = bpy.data.objects.new(PLAYER_NAME, mesh)
            scene.collection.objects.link(obj)

        self._obj: Object = obj

        self.logger.debug("Player ensured.")

    def setup_geometry_nodes(self):
        import bpy

        # create geometry node modifier if missing
        mod: Modifier = self._obj.modifiers.get(
            GN_MODIFIER_NAME
        ) or self._obj.modifiers.new(GN_MODIFIER_NAME, type="NODES")
        assert isinstance(mod, bpy.types.NodesModifier)

        if mod.node_group is None:
            node_group: NodeTree = bpy.data.node_groups.new(
                GN_TREE_NAME, "GeometryNodeTree"
            )
            mod.node_group = node_group
        else:
            node_group = mod.node_group

        assert isinstance(node_group, bpy.types.GeometryNodeTree)

        nodes: Nodes = node_group.nodes
        links: NodeLinks = node_group.links

        nodes.clear()

        input_node: Node = nodes.new("NodeGroupInput")
        output_node: Node = nodes.new("NodeGroupOutput")
        set_material: Node = nodes.new("GeometryNodeSetMaterial")

        mesh_to_points: Node = nodes.new("GeometryNodeMeshToPoints")

        radius_input = mesh_to_points.inputs["Radius"]
        assert isinstance(radius_input, bpy.types.NodeSocketFloatDistance)
        radius_input.default_value = DEFAULT_POINT_RADIUS

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
        mat: Material = bpy.data.materials.get(MATERIAL_NAME) or bpy.data.materials.new(
            MATERIAL_NAME
        )
        mat.use_nodes = True

        mat_input = set_material.inputs["Material"]
        assert isinstance(mat_input, bpy.types.NodeSocketMaterial)

        mat_input.default_value = mat

        # links
        links.new(input_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
        links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
        links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])

        self.logger.debug("Geometry Nodes setup complete.")

    def setup_material(self):
        import bpy

        mat: Material = bpy.data.materials.get(MATERIAL_NAME) or bpy.data.materials.new(
            MATERIAL_NAME
        )

        assert mat.node_tree

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        attr: Node = nodes.new("ShaderNodeAttribute")
        assert isinstance(attr, bpy.types.ShaderNodeAttribute)
        attr.attribute_name = "Col"

        bsdf: Node = nodes.new("ShaderNodeBsdfPrincipled")
        output: Node = nodes.new("ShaderNodeOutputMaterial")

        links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        self.logger.debug("Material setup complete.")


# prefix function name to prevent any collision with other addons' handlers
def mosplat_on_frame_change_pc(
    scene: Scene,
    *,
    ply_file_formatter: ExportedFileFormatterPartial,
    pct_file_formatter: ExportedFileFormatterPartial,
):
    import bpy
    import mathutils

    obj: Optional[Object] = bpy.data.objects.get(PLAYER_NAME)
    if obj is None:
        return

    TRANSFORM_MATRIX = mathutils.Matrix(
        (
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, -1, 0, 0),
            (0, 0, 0, 1),
        )
    )
    SCALE_MATRIX = mathutils.Matrix.Scale(100.0, 4)

    mesh = import_ply_mesh_for_frame(
        scene.frame_current,
        TRANSFORM_MATRIX,
        SCALE_MATRIX,
        ply_file_formatter=ply_file_formatter,
    )
    if mesh is None:
        return

    import_cameras_for_frame(
        scene.frame_current,
        TRANSFORM_MATRIX,
        SCALE_MATRIX,
        pct_file_formatter=pct_file_formatter,
    )

    old_mesh: Mesh = cast(bpy.types.Mesh, obj.data)
    obj.data = mesh

    # remove old mesh datablock to prevent RAM memory growth
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def import_ply_mesh_for_frame(
    curr_frame: int,
    transform_mat: Matrix,
    scale_mat: Matrix,
    *,
    ply_file_formatter: ExportedFileFormatterPartial,
) -> Optional[Mesh]:
    import bpy

    ply_filepath: str = ply_file_formatter(frame_idx=curr_frame)

    if not is_path_accessible(Path(ply_filepath)):
        return None

    before: Set[Object] = set(bpy.data.objects)

    bpy.ops.wm.ply_import(filepath=ply_filepath)

    after: Set[Object] = set(bpy.data.objects)
    created_obj: Object = list(after - before)[0]

    assert created_obj.data

    new_mesh: Mesh = cast(bpy.types.Mesh, created_obj.data.copy())

    new_mesh.transform(scale_mat @ transform_mat)
    new_mesh.update()

    bpy.data.objects.remove(created_obj, do_unlink=True)

    return new_mesh


def import_cameras_for_frame(
    curr_frame: int,
    transform_mat: Matrix,
    scale_mat: Matrix,
    *,
    pct_file_formatter: ExportedFileFormatterPartial,
):
    import mathutils

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pct_filepath: Path = Path(pct_file_formatter(frame_idx=curr_frame))
    pct_anno_map = PointCloudTensors.annotation_map()

    try:
        pct_dict = load_safetensors(pct_filepath, device, None, pct_anno_map)
    except Exception as e:
        msg = UserFacingError.make_msg(
            f"Unable to load safetensor data from '{pct_filepath}' to extract camera data for frame '{curr_frame}'.",
            e,
        )
        logger.error(msg)
        return

    pct: PointCloudTensors = PointCloudTensors.from_dict(pct_dict)

    extrinsic = pct.extrinsic
    intrinsic = pct.intrinsic
    num_cams = extrinsic.shape[0]

    for cam_id in range(num_cams):
        cam_obj = ensure_camera(cam_id)

        extri = extrinsic[cam_id].tolist()  # move to a cpu list
        intri = intrinsic[cam_id].tolist()

        world_cam = mathutils.Matrix(
            (
                (extri[0][0], extri[0][1], extri[0][2], extri[0][3]),
                (extri[1][0], extri[1][1], extri[1][2], extri[1][3]),
                (extri[2][0], extri[2][1], extri[2][2], extri[2][3]),
                (0, 0, 0, 1),
            )
        )
        cam_world = world_cam.inverted()
        cam_world = scale_mat @ transform_mat @ cam_world

        cam_obj.matrix_world = cam_world  # assign transform to camera

        cam_obj.data.display_size = CAMERA_DISPLAY_SIZE


def ensure_camera(cam_id: int):
    import bpy

    name = CAMERA_NAME_FORMATTER.format(cam_id=cam_id)
    obj = bpy.data.objects.get(name)

    if obj is None:
        cam_data = bpy.data.cameras.new(name)
        obj = bpy.data.objects.new(name, cam_data)
        if not bpy.context.scene or not bpy.context.scene.collection:
            logger.warning(f"Unable to link camera '{cam_id}' to scene")
        else:
            collection = ensure_camera_collection(CAMERA_COLLECTION_NAME)
            collection.objects.link(obj)

    return obj


def ensure_camera_collection(name: str):
    import bpy

    collection = bpy.data.collections.get(name)

    if collection is None:
        collection = bpy.data.collections.new(name)
        if not bpy.context.scene or not bpy.context.scene.collection:
            logger.warning("Unable to link camera collection to scene")
        else:
            bpy.context.scene.collection.children.link(collection)

    return collection
