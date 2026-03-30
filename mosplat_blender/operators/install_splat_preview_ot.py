from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set, Tuple, cast

from ..infrastructure.constants import (
    SPLAT_ATTRIBUTES,
    SPLAT_PLAYER_OBJ_NAME,
    SPLAT_RENDER_HELPER_SCENE,
)
from ..infrastructure.identifiers import OperatorIDEnum
from ..infrastructure.macros import is_path_accessible
from ..infrastructure.mixins import CtxPackage
from ..infrastructure.schemas import ExportedFileName, UnexpectedError
from ..interfaces.logging_interface import LoggingInterface
from .base_ot import MosplatOperatorBase

if TYPE_CHECKING:
    from bpy.types import Mesh, Object, Scene

logger = LoggingInterface.configure_logger_instance(__name__)

SPLAT_PLAYER_MESH_NAME = "splat_mesh"

SH_C0 = 0.28209479177387814

UNIFORM_SCALE = 100.0

PLY_FILE_FORMATTER = None

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

        scene = pkg.context.scene
        if scene:
            mosplat_on_frame_change_splat(scene)  # initial execution

        return "RUNNING_MODAL"

    def install(self, pkg):
        import bpy

        append_helper_scene()
        self.ensure_player(pkg)

        # prevent duplicate registration of handler
        for idx, handler in enumerate(bpy.app.handlers.frame_change_pre):
            name = getattr(handler, "__name__", None) or (
                handler.func.__name__ if isinstance(handler, partial) else ""
            )
            if name == mosplat_on_frame_change_splat.__name__:
                bpy.app.handlers.frame_change_pre.pop(idx)

        bpy.app.handlers.frame_change_pre.append(mosplat_on_frame_change_splat)

        self.logger.debug("Installation complete.")

    def ensure_player(self, pkg: CtxPackage):
        import bpy

        scene = pkg.context.scene
        assert scene
        assert scene.collection

        obj = bpy.data.objects.get(SPLAT_PLAYER_OBJ_NAME)
        if obj is None:
            mesh: Mesh = bpy.data.meshes.new(SPLAT_PLAYER_MESH_NAME)
            obj = bpy.data.objects.new(SPLAT_PLAYER_OBJ_NAME, mesh)
            scene.collection.objects.link(obj)

        apply_rendering()

        self.logger.debug("Player ensured.")


# prefix function name to prevent any collision with other addons' handlers
def mosplat_on_frame_change_splat(scene: Scene):
    import bpy

    obj: Optional[Object] = bpy.data.objects.get(SPLAT_PLAYER_OBJ_NAME)
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

    apply_rendering()


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


def process_imported_ply(obj: Object) -> Optional[Mesh]:
    import bpy

    assert obj.data and isinstance(obj.data, bpy.types.Mesh)

    mesh: bpy.types.Mesh = obj.data.copy()
    bpy.data.objects.remove(obj, do_unlink=True)

    for attr in SPLAT_ATTRIBUTES:
        if attr not in mesh.attributes:
            logger.warning(
                f"'{attr}' attribute missing from imported splat file. Will not continue."
            )
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            return

    transform_mesh(mesh)
    handle_mesh_color_from_sh(mesh)

    return mesh


def handle_mesh_color_from_sh(mesh: Mesh):
    import numpy as np

    point_count = len(mesh.vertices)

    f_dc_0 = parse_attribute(mesh, "f_dc_0")
    f_dc_1 = parse_attribute(mesh, "f_dc_1")
    f_dc_2 = parse_attribute(mesh, "f_dc_2")
    opacity = parse_attribute(mesh, "opacity")

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

    if "Col" in mesh.attributes:
        mesh.attributes.remove(mesh.attributes["Col"])

    # replace with new attribute
    col_attr = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    col_attr.data.foreach_set("color", rgba.ravel())

    mesh.color_attributes.active_color = col_attr


def transform_mesh(mesh: Mesh):
    import math

    import mathutils

    import numpy as np

    # extract vert positions
    verts = mesh.vertices
    coords = np.array([v.co[:] for v in verts], dtype=np.float32)

    # center mesh by subtracting the centroid of the vertices
    centroid = coords.mean(axis=0)
    coords -= centroid

    # y-up to z-up
    rot = mathutils.Matrix.Rotation(-math.pi / 2, 4, "X")
    coords = np.array([rot @ mathutils.Vector(c) for c in coords], dtype=np.float32)

    # scale by 100
    coords *= UNIFORM_SCALE
    UNIFORM_LOG_SCALE = np.log(UNIFORM_SCALE)

    for attr_name in ("scale_0", "scale_1", "scale_2"):
        if attr_name in mesh.attributes:
            attr = mesh.attributes[attr_name]

            data = np.empty(len(attr.data), dtype=np.float32)
            attr.data.foreach_get("value", data)

            # apply same uniform scale to splat scales, which are log-encoded
            data += UNIFORM_LOG_SCALE

            attr.data.foreach_set("value", data)

    # write back onto mesh
    for i, v in enumerate(verts):
        v.co = coords[i]


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


def parse_attribute(pc: Mesh, attr_name: str):
    import numpy as np

    attr = pc.attributes[attr_name]

    arr = np.empty(len(attr.data), dtype=np.float32)
    attr.data.foreach_get("value", arr)  # `C` backend function
    return arr


def apply_rendering():
    try:
        OperatorIDEnum.run(OperatorIDEnum.APPLY_SPLAT_RENDER_MODE)
    except Exception as e:
        msg = UnexpectedError.make_msg(
            f"Error occurred while applying splat rendering mode.", e
        )
        logger.error(msg)
