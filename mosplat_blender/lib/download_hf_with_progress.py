"""
this script exists to be ran as a subprocess for downloading a model from huggingface.
this is due to the size of the downloads being multiple gigabytes.
running as a subprocess allows complete separation of concerns,
and the process can even be completely killed if necessary.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, Type, TypedDict

if "--" in sys.argv:  # parse original subprocess call
    ARGV = sys.argv[sys.argv.index("--") + 1 :]
else:
    ARGV = []


@dataclass
class SubprocessPayload:
    status: str = ""
    current: int = -1
    total: int = -1
    msg: str = ""

    def overwrite(self, *, status, current, total, msg):
        self.status = status
        self.current = current
        self.total = total
        self.msg = msg

    def values_tuple(self) -> Tuple[str, int, int, str]:
        return tuple(asdict(self).values())


def make_tqdm_class():
    from huggingface_hub.utils.tqdm import tqdm

    class ProgressTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            self._payload = SubprocessPayload()
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)

        def display(self, *args, **kwargs):

            if self.total:
                self._payload.overwrite(
                    status="progress",
                    current=int(float(self.n) / 100.0),  # convert from bytes to mb
                    total=int(float(self.total) / 100.0),
                    msg="",
                )
                self.sp(str(self._payload.values_tuple()))

        def refresh(self, *args, **kwargs) -> None:
            super().refresh(*args, **kwargs)

    return ProgressTqdm


class DownloadArgs(TypedDict):
    repo_id: str
    filename: str
    cache_dir: str


def main():
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("cache_dir", type=Path)

    args = p.parse_args(ARGV)
    repo_id, cache_dir = args.repo_id, args.cache_dir

    dl_args = DownloadArgs(
        repo_id=repo_id, filename=SAFETENSORS_SINGLE_FILE, cache_dir=str(cache_dir)
    )

    try:
        hf_hub_download(**dl_args, tqdm_class=make_tqdm_class())

        print(
            str(SubprocessPayload("ok", msg=str(json.dumps(dl_args))).values_tuple()),
            flush=True,
        )  # send download args
    except Exception:
        # ensure download args still send
        print(
            str(
                SubprocessPayload(
                    "warning", msg=str(json.dumps(dl_args))
                ).values_tuple()
            ),
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
