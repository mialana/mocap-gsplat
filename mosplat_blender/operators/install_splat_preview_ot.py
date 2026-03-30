from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set, Tuple, cast

from ..infrastructure.constants import SPLAT_RENDER_HELPER_SCENE
from ..infrastructure.macros import is_path_accessible
from ..infrastructure.mixins import CtxPackage
from ..infrastructure.schemas import DeveloperError, ExportedFileName, UnexpectedError
from ..interfaces.logging_interface import LoggingInterface
from .base_ot import MosplatOperatorBase

if TYPE_CHECKING:
    from bpy.types import Material, Mesh, Modifier, NodeGroup, Object, Scene

logger = LoggingInterface.configure_logger_instance(__name__)

SH_C0 = 0.28209479177387814

MESH_NAME = "splat_mesh"
PLAYER_NAME = "SplatPlaybackManager"
GN_MODIFIER_NAME = "SplatGN"
GN_TREE_NAME = "SplatGNGroup"
MATERIAL_NAME = "SplatMaterial"
DEFAULT_POINT_RADIUS = 0.1

PLY_FILE_FORMATTER = None

SPLAT_ATTRIBUTES = [
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_rest_0",
]

# categories that should be imported from helper scene
IMPORT_CATEGORIES = ["node_groups", "materials"]


class Mosplat_OT_install_splat_preview(MosplatOperatorBase):
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
        self._increment_ply_file: bool = bool(props.config_accessor.increment_ply_file)

        PLY_FILE_FORMATTER = ExportedFileName.to_formatter(
            self._exported_file_formatter,
            file_name=ExportedFileName.SPLAT,
            file_ext="ply",
        )

        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):

        self.install(pkg)

        return "RUNNING_MODAL"

    def install(self, pkg):
        import bpy

        append_helper_scene()
        self.ensure_player(pkg)

        # prevent duplicate registration of handler
        for idx, handler in enumerate(bpy.app.handlers.frame_change_pre):
            if handler.__name__ == on_frame_change.__name__:
                bpy.app.handlers.frame_change_pre.pop(idx)

        bpy.app.handlers.frame_change_pre.append(on_frame_change)

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


def on_frame_change(scene: Scene):
    import bpy

    obj: Optional[Object] = bpy.data.objects.get(PLAYER_NAME)
    if obj is None:
        return

    mesh = import_ply_mesh_for_frame(scene.frame_current)
    if mesh is None:
        return

    old_mesh: Mesh = cast(bpy.types.Mesh, obj.data)
    obj.data = mesh

    # remove old mesh datablock to prevent RAM memory growth
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def import_ply_mesh_for_frame(curr_frame: int) -> Optional[Mesh]:
    import bpy

    if not PLY_FILE_FORMATTER:
        raise UnexpectedError(f"Global PLY file formatter string no longer in scope.")

    ply_filepath: Path = Path(PLY_FILE_FORMATTER(frame_idx=curr_frame))
    if not is_path_accessible(ply_filepath):
        stem = ply_filepath.stem
        dir = ply_filepath.parent
        ext = ply_filepath.suffix

        # sort alphanumerically, choose last. i.e. has largest index
        files = sorted(list(dir.glob(f"{stem}.*{ext}")))
        if not files or not is_path_accessible(ply_filepath := files[-1]):
            return None

    before: Set[Object] = set(bpy.data.objects)

    bpy.ops.wm.ply_import(filepath=str(ply_filepath))

    after: Set[Object] = set(bpy.data.objects)
    created_obj: Object = list(after - before)[0]

    return process_imported_ply(created_obj)


def parse_attribute(pc: Mesh, attr_name: str):
    import numpy as np

    attr = pc.attributes[attr_name]

    arr = np.empty(len(attr.data), dtype=np.float32)
    attr.data.foreach_get("value", arr)  # `C` backend function
    return arr


def process_imported_ply(obj: Object) -> Optional[Mesh]:
    import bpy

    import numpy as np

    assert obj.data and isinstance(obj.data, bpy.types.Mesh)

    pc: bpy.types.Mesh = obj.data

    for attr in SPLAT_ATTRIBUTES:
        if attr not in pc.attributes:
            logger.warning(
                f"'{attr}' attribute missing from imported splat file. Will not continue."
            )
            return

    point_count = len(pc.vertices)

    f_dc_0 = parse_attribute(pc, "f_dc_0")
    f_dc_1 = parse_attribute(pc, "f_dc_1")
    f_dc_2 = parse_attribute(pc, "f_dc_2")
    opacity = parse_attribute(pc, "opacity")

    if not (len(f_dc_0) == len(f_dc_1) == len(f_dc_2) == len(opacity) == point_count):
        logger.warning(
            f"Attribute data lengths are not uniform. Expected count is '{point_count}'."
        )

    rgb = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)
    rgb = rgb * SH_C0 + 0.5

    alpha = 1 / (1 + np.exp(-opacity))  # sigmoid

    # clamp values
    rgb = np.clip(rgb, 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # build rgba
    rgba = np.concatenate([rgb, alpha[:, None]], axis=1)

    assert len(rgba) == point_count

    if "Col" in pc.attributes:
        pc.attributes.remove(pc.attributes["Col"])

    # replace with new attribute
    col_attr = pc.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    col_attr.data.foreach_set("color", rgba.ravel())

    pc.color_attributes.active_color = col_attr

    render_mod = apply_modifier_from_node_group("KIRI_3DGS_Render_GN", obj)
    sorter_mod = apply_modifier_from_node_group("KIRI_3DGS_Sorter_GN", obj)
    color_adjust_mod = apply_modifier_from_node_group(
        "KIRI_3DGS_Adjust_Colour_And_Material", obj
    )
    merger_mod = apply_modifier_from_node_group("KIRI_3DGS_Write F_DC_And_Merge", obj)
    # TODO: set properties of modifiers

    apply_material("KIRI_3DGS_Render_Material", obj)

    return


def apply_modifier_from_node_group(name: str, obj: Object) -> Modifier:
    if name in obj.modifiers:
        return obj.modifiers[name]  # early return if already exists

    ng: NodeGroup = ensure_import(name, "node_groups")

    mod = obj.modifiers.new(name, type="NODES")
    mod.node_group = ng

    return mod


def apply_material(name: str, obj: Object) -> Material:
    import bpy

    mesh = obj.data

    assert mesh and isinstance(mesh, bpy.types.Mesh)

    mat: Material
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
    else:
        while len(obj.material_slots) > 0:
            obj.active_material_index = 0
            try:
                bpy.ops.object.material_slot_remove()
            except RuntimeError as e:
                logger.error(
                    f"Could not clear materials from object '{obj.name}'", str(e)
                )
                continue  # try to continue execution for now
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True

    if name not in mesh.materials:
        mesh.materials.append(mat)

    return mat


def ensure_import(name: str, category: str):
    """if node groups are deleted at some point, dynamically import"""
    import bpy

    if name in getattr(bpy.data, category):
        return getattr(bpy.data, category)[name]

    with bpy.data.libraries.load(str(SPLAT_RENDER_HELPER_SCENE), link=False) as (
        data_from,
        data_to,
    ):
        if name not in getattr(data_from, category):
            raise DeveloperError(
                f"Target '{name}' in category '{category}' not found in helper scene."
            )

        setattr(data_to, category, [name])

    return getattr(bpy.data, category)[name]


def append_helper_scene():
    """append all node groups from helper scene"""
    import bpy

    for lib in bpy.data.libraries:
        abs_path = Path(bpy.path.abspath(lib.filepath)).resolve()
        if abs_path == SPLAT_RENDER_HELPER_SCENE:
            logger.debug(f"'{SPLAT_RENDER_HELPER_SCENE}' already loaded.")
            return

    with bpy.data.libraries.load(str(SPLAT_RENDER_HELPER_SCENE), link=False) as (
        data_from,
        data_to,
    ):
        for category in IMPORT_CATEGORIES:
            existing = getattr(bpy.data, category).keys()
            imported = [
                name for name in getattr(data_from, category) if name not in existing
            ]
            setattr(data_to, category, imported)

            logger.debug(f"Imported '{len(imported)}' '{category}'.")
