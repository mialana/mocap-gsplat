from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from ..infrastructure.constants import (
    KIRI_LITERALS as KIRI,
    SPLAT_PLAYER_OBJ_NAME,
    SPLAT_RENDER_HELPER_SCENE,
)
from ..infrastructure.schemas import DeveloperError, SplatRenderMode, UnexpectedError
from ..interfaces.logging_interface import LoggingInterface
from .base_ot import MosplatOperatorBase

if TYPE_CHECKING:
    from bpy.types import Material, Modifier, NodeGroup, Object


logger = LoggingInterface.configure_logger_instance(__name__)


class ObjectDataPayload(NamedTuple):
    render_mod: Modifier
    sorter_mod: Modifier
    color_adjust_mod: Modifier
    merger_mod: Modifier
    mat: Material


class Mosplat_OT_apply_splat_render_mode(MosplatOperatorBase):
    @classmethod
    def _contexted_poll(cls, pkg):
        import bpy

        props = pkg.props
        cls._poll_error_msg_list.extend(props.is_valid_media_directory_poll_result)
        cls._poll_error_msg_list.extend(props.frame_range_poll_result(pkg.prefs))

        obj = bpy.data.objects.get(SPLAT_PLAYER_OBJ_NAME)
        if obj is None:
            cls._poll_error_msg_list.append(
                f"There is no valid '{SPLAT_PLAYER_OBJ_NAME}' object in the scene. Try changing the animation frame."
            )

        return len(cls._poll_error_msg_list) == 0

    def _contexted_execute(self, pkg):
        import bpy

        mode: SplatRenderMode = SplatRenderMode.from_variable_name(
            pkg.props.splat_render_mode
        )

        obj = bpy.data.objects.get(SPLAT_PLAYER_OBJ_NAME)

        render_mod = apply_modifier_from_node_group(KIRI["render_mod"], obj)
        sorter_mod = apply_modifier_from_node_group(KIRI["sorter_mod"], obj)
        color_adjust_mod = apply_modifier_from_node_group(KIRI["color_adjust_mod"], obj)
        merger_mod = apply_modifier_from_node_group(KIRI["merger_mod"], obj)

        mat = apply_material(KIRI["mat"], obj)
        payload = ObjectDataPayload(
            render_mod, sorter_mod, color_adjust_mod, merger_mod, mat
        )

        if mode == SplatRenderMode.POINTCLOUD:
            apply_pointcloud_render_mode(obj, payload)
        else:
            apply_gaussian_render_mode(obj, payload)

        obj.update_tag(refresh={"OBJECT", "DATA"})

        if bpy.context and bpy.context.screen:
            for area in bpy.context.screen.areas:
                area.tag_redraw()

        self.logger.info(f"Applied '{mode.name}' splat render mode.")

        return "FINISHED"


def apply_pointcloud_render_mode(obj: Object, payload: ObjectDataPayload):
    render_mod, sorter_mod, color_adjust_mod, merger_mod, mat = payload

    kiri_pg = getattr(obj, KIRI["property_group"])
    kiri_pg.update_mode = "Show As Point Cloud"

    render_mod[KIRI["update_mode_socket"]] = 2  # corresponds to 'Show as Point Cloud'
    render_mod[KIRI["point_radius_socket"]] = 0.001  # TODO: expose as property
    render_mod[KIRI["material_socket"]] = mat
    render_mod.show_viewport = True

    color_adjust_mod.show_viewport = True

    merger_mod.show_viewport = False
    merger_mod.show_render = False

    sorter_mod.show_viewport = False
    sorter_mod.show_render = False

    material_render_method = mat.surface_render_method

    logger.debug(f"Material render method: '{material_render_method}'")

    sorter_mod.show_viewport = material_render_method == "BLENDED"
    sorter_mod.show_render = material_render_method == "BLENDED"


def apply_gaussian_render_mode(obj: Object, payload: ObjectDataPayload):
    import bpy

    render_mod, sorter_mod, color_adjust_mod, merger_mod, mat = payload

    render_mod[KIRI["update_mode_socket"]] = 0
    render_mod.show_viewport = True
    color_adjust_mod.show_viewport = True
    merger_mod.show_viewport = False
    merger_mod.show_render = False
    sorter_mod.show_viewport = False
    sorter_mod.show_render = False

    kiri_pg = getattr(obj, KIRI["property_group"])
    kiri_pg.update_mode = "Enable Camera Updates"

    material_render_method = mat.surface_render_method

    logger.debug(f"Material render method: '{material_render_method}'")

    render_mod[KIRI["material_socket"]] = mat

    assert bpy.context.view_layer

    bpy.context.view_layer.objects.active = obj

    kiri_operators = getattr(bpy.ops, "sna")
    op = getattr(kiri_operators, KIRI["align_to_view_operator"])
    op()

    bpy.context.view_layer.objects.active = None


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
                msg = UnexpectedError.make_msg(
                    f"Could not clear materials from object '{obj.name}'", e
                )
                logger.error(msg)
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
