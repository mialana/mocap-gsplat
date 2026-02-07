from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, List, NamedTuple, Tuple, cast

from infrastructure.constants import PREPROCESS_MEDIA_SCRIPT_FUNCTION_NAME
from infrastructure.macros import (
    get_required_function,
    import_module_from_path_dynamic,
)
from infrastructure.schemas import (
    DeveloperError,
    FrameNPZStructure,
    MediaIODataset,
    NPZNameToPathLookup,
    SavedNPZName,
    UserFacingError,
)
from operators.base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    preprocess_fn: Callable
    npz_file_lookup: NPZNameToPathLookup
    media_files: List[Path]
    dataset_as_dc: MediaIODataset


class Mosplat_OT_run_preprocess_script(
    MosplatOperatorBase[Tuple[str, str], ProcessKwargs],
):
    @classmethod
    def _contexted_poll(cls, pkg) -> bool:
        props = pkg.props
        cls._poll_error_msg_list.extend(props.is_valid_media_directory_poll_result)
        cls._poll_error_msg_list.extend(props.frame_range_poll_result(pkg.prefs))

        return len(cls._poll_error_msg_list) == 0

    def _contexted_invoke(self, pkg, event):
        prefs = pkg.prefs
        props = pkg.props

        self._media_files: List[Path] = props.media_files(prefs)
        self._preprocess_script = prefs.preprocess_media_script_filepath
        self._npz_file_lookup: NPZNameToPathLookup = (
            props.generate_npz_filepaths_for_frame_range(
                prefs, [SavedNPZName.RAW, SavedNPZName.PREPROCESSED], [True, False]
            )
        )

        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        try:
            self._module = import_module_from_path_dynamic(self._preprocess_script)
            self._preprocess_fn = get_required_function(
                self._module, PREPROCESS_MEDIA_SCRIPT_FUNCTION_NAME
            )
            self.logger.info(
                f"Found and imported function '{self._preprocess_fn.__qualname__}' from module '{self._module.__name__}'."
            )
        except ImportError as e:
            msg = UserFacingError.make_msg(
                f"Cannot import selected preprocess script: '${self._preprocess_script}'",
                e,
            )
            self.logger.error(msg)
            return "FINISHED"
        except (AttributeError, TypeError) as e:
            msg = UserFacingError.make_msg(
                f"Cannot import required `${PREPROCESS_MEDIA_SCRIPT_FUNCTION_NAME}` function from selected preprocess script: '${self._preprocess_script}'",
                e,
            )
            self.logger.error(msg)
            return "FINISHED"

        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(
                preprocess_fn=self._preprocess_fn,
                npz_file_lookup=self._npz_file_lookup,
                media_files=self._media_files,
                dataset_as_dc=self.data,
            ),
        )

        self._sync_to_props(pkg.props)

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg, event, next):
        status, msg = next
        if status == "error":
            self.logger.error(msg)
        elif status == "warning":  # if sent a warning via queue
            self.logger.warning(msg)
        elif status == "done":
            self.logger.info(msg)
        else:
            self.logger.debug(msg)

        if status == "done" or status == "error":
            return "FINISHED"  # error still needs finished as blender data was modified
        else:
            return "RUNNING_MODAL"

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        import numpy as np
        from numpy.lib import npyio

        fn = pwargs.preprocess_fn
        # as strings and `Counter` collection type
        media_files_counter = Counter(pwargs.media_files)

        raw_npz_files = pwargs.npz_file_lookup[SavedNPZName.RAW]
        preprocessed_npz_files = pwargs.npz_file_lookup[SavedNPZName.PREPROCESSED]
        for idx, files in enumerate(zip(raw_npz_files, preprocessed_npz_files)):
            if cancel_event.is_set():
                return

            raw, preprocessed = files
            try:
                npz_file: npyio.NpzFile = np.load(raw, mmap_mode=None)
                npz_structure = FrameNPZStructure(**dict(npz_file.items()))
                # converts to native int
                frame_idx: int = npz_structure.get("frame")[0]
                if frame_idx != idx:
                    raise DeveloperError(
                        "Saved frame index should match iteration index.\n"
                        f"Expected Index: '{idx}'\nExtracted Index: '{frame_idx}'"
                    )
                media_files: List[Path] = [
                    Path(s) for s in npz_structure.get("media_files")
                ]
                if media_files_counter != Counter(media_files):
                    raise ValueError(
                        f"Media files did not match."
                        f"\nExpected Files: '{pwargs.media_files}'"
                        f"\nExtracted Files: '{media_files}'"
                    )
            except (OSError, ValueError, TypeError) as e:
                msg = UserFacingError.make_msg(
                    f"Data used to create NPZ file did not match currrent state of media directory or NPZ files are corrupted. Behavior is unpredictable. Delete the NPZ files and re-extract frame data to clean up data state.",
                    e,
                )
                queue.put(("error", msg))
                break
            try:
                processed_data = fn(frame_idx, media_files, npz_file.get("data"))
                npz_structure["data"] = processed_data
                np.savez(preprocessed, **cast(dict, npz_structure))
                queue.put(("update", f"Finished processing frame '{frame_idx}'"))
            except Exception as e:
                msg = UserFacingError.make_msg(
                    f"Could not run script on frame '{frame_idx}'", e
                )
                queue.put(("error", msg))
                break


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_run_preprocess_script._operator_subprocess(*args, **kwargs)
