from ...infrastructure.identifiers import OperatorIDEnum
from .base_pt import MosplatPanelBase


class Mosplat_PT_data_inference(MosplatPanelBase):
    @classmethod
    def _contexted_poll(cls, pkg) -> bool:
        props = pkg.props
        return (
            len(props.is_valid_media_directory_poll_result) == 0
            and len(props.frame_range_poll_result(pkg.prefs)) == 0
            and pkg.props.was_frame_range_extracted
        )

    def draw_with_layout(self, pkg, layout):
        props = pkg.props
        column = layout.column()

        init_model_box = column.box()
        init_model_box.row().operator(OperatorIDEnum.INITIALIZE_MODEL)
        progress = props.progress_accessor
        prog_curr: int = progress.current
        prog_total: int = progress.total
        if prog_curr > 0 and prog_total > 0:
            init_model_box.row().progress(factor=(float(prog_curr) / float(prog_total)))

        options = props.options_accessor
        prop_box = layout.box()
        prop_box.prop(options, options._meta.inference_mode.id)
        prop_box.prop(options, options._meta.confidence_percentile.id)

        column.row().operator(OperatorIDEnum.RUN_INFERENCE)
