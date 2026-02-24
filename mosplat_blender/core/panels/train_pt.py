from ...infrastructure.identifiers import OperatorIDEnum
from .base_pt import MosplatPanelBase


class Mosplat_PT_train(MosplatPanelBase):
    @classmethod
    def _contexted_poll(cls, pkg) -> bool:
        import torch

        props = pkg.props
        return (
            len(props.is_valid_media_directory_poll_result) == 0
            and len(props.frame_range_poll_result(pkg.prefs)) == 0
            and pkg.props.was_frame_range_extracted
            and pkg.props.was_frame_range_preprocessed
            and pkg.props.ran_inference_on_frame_range
            and torch.cuda.is_available()
        )

    def draw_with_layout(self, pkg, layout):
        props = pkg.props
        config = props.config_accessor

        layout.prop(config, config._meta.steps.id)
        layout.prop(config, config._meta.lr.id)
        layout.prop(config, config._meta.sh_degree.id)

        row = layout.row()
        row.enabled = False
        row.prop(config, config._meta.scene_size.id)

        layout.prop(config, config._meta.alpha_weight.id)
        layout.prop(config, config._meta.depth_weight.id)
        layout.prop(config, config._meta.save_ply_interval.id)

        layout.row().operator(OperatorIDEnum.TRAIN_GAUSSIAN_SPLATS)
