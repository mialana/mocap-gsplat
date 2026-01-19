from pathlib import Path
from typing import final, TYPE_CHECKING, ClassVar, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from ..infrastructure.decorators import no_instantiate
from ..infrastructure.mixins import MosplatLogClassMixin
from ..infrastructure.constants import MOSPLAT_MEDIA_METADATA_FILENAME

if TYPE_CHECKING:
    from ..core.preferences import Mosplat_AP_Global
    from ..core.properties import Mosplat_PG_Global
else:
    Mosplat_AP_Global: TypeAlias = Any
    Mosplat_PG_Global: TypeAlias = Any


@dataclass(frozen=True)
class MosplatAppliedPreprocessScript:
    date_last_applied: str = datetime.now().isoformat()
    script_path: str = ""

    @staticmethod
    def now(script_path: str) -> "MosplatAppliedPreprocessScript":
        return MosplatAppliedPreprocessScript(
            script_path=script_path,
            date_last_applied=datetime.now().isoformat(),
        )


@dataclass
class MosplatProcessedFrameRange:
    start_frame: int
    end_frame: int
    applied_preprocess_scripts: Set[MosplatAppliedPreprocessScript]


@dataclass
class MosplatMediaMetadata:
    base_directory: str = ""
    is_valid: bool = False
    collective_frame_count: int = -1
    media_files: Set[str] = set()
    processed_frame_ranges: Set[MosplatProcessedFrameRange] = set()

    def toJSON(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        as_dict = asdict(self)

        with path.open("w", encoding="utf-8") as f:
            json.dump(as_dict, f, sort_keys=True, indent=4)

    def fromJSON(self, path: Path):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.__init__(**data)

    def try_normalize_frame_ranges(self):
        """combine overlapping frame ranges if they have had the same preprocess scripts applied to them"""


@final
@no_instantiate
class MosplatMediaIOInterface(MosplatLogClassMixin):
    metadata: ClassVar[Union[MosplatMediaMetadata, None]] = None
    data_output_dir: ClassVar[Union[Path, None]] = None

    @classmethod
    def set_data_output_dir(cls, path: Path):
        cls.data_output_dir = path
        cls._find_or_create_metadata_json()

    @classmethod
    def set_metadata_base_directory(cls, dirpath: Path):
        cls.metadata = MosplatMediaMetadata(base_directory=str(dirpath))
        cls._find_or_create_metadata_json()

    @classmethod
    def set_metadata_media_files(cls, files: Set[str]):
        if not cls.metadata:
            raise RuntimeError(f"Not set: `{cls.metadata=}`")
        cls.metadata.media_files = files

    @classmethod
    def _metadata_path(cls) -> Path:
        if not cls.data_output_dir:
            raise RuntimeError(f"Not set: `{cls.data_output_dir=}`")
        return cls.data_output_dir.joinpath(MOSPLAT_MEDIA_METADATA_FILENAME)

    @classmethod
    def _find_or_create_metadata_json(cls):
        pass

    @classmethod
    def apply_preprocess_script(cls, script_path: Path) -> bool:
        return False

    @classmethod
    def extract_frame_range(cls, frame_range):
        pass
