from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, Any

from ...infrastructure.constants import OperatorIDEnum, PanelIDEnum

from .base_pt import MosplatPanelBase, PanelPollReqs

if TYPE_CHECKING:
    from ..properties import Mosplat_PG_MediaProcessStatus
else:
    Mosplat_PG_MediaProcessStatus: TypeAlias = Any


class Mosplat_PT_Preprocess(MosplatPanelBase):
    bl_idname = PanelIDEnum.PREPROCESS
    bl_description = "Holds operations for preprocessing Mosplat data"

    bl_parent_id = PanelIDEnum.MAIN

    __poll_reqs__ = {PanelPollReqs.PROPS}

    def draw_with_layout(self, context, layout):
        column = layout.column()

        props = self.props(context)

        column.operator(OperatorIDEnum.INITIALIZE_MODEL)

        column.separator()

        box = column.box()
        box.row().label(text=props.get_prop_name("current_media_dir"))
        box.row().prop(props, "current_media_dir", text="")

        if props.current_media_io_metadata.media_process_statuses:
            media_box = box.box()
            media_box.label(
                text=props.current_media_io_metadata.get_prop_name(
                    "media_process_statuses"
                )
            )
            media_box.alert = (
                not props.current_media_io_metadata.do_media_durations_all_match
            )
            for item in props.current_media_io_metadata.media_process_statuses:
                media: Mosplat_PG_MediaProcessStatus = item
                item_basename = Path(media.filepath).name
                row = media_box.row()
                row.label(text=item_basename, icon="FILE_MOVIE")
                sub = row.row()
                sub.alignment = "RIGHT"
                sub.label(text=str(media.frame_count))

        box.row().prop(props, "current_frame_range")

        column.row().operator(OperatorIDEnum.RUN_INFERENCE)
