from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

from semantic_digital_twin.adapters.grasp_clutter_6d_dataset.exceptions import (
    GraspClutter6DArchiveLayoutError,
    GraspClutter6DSceneNotFoundError,
)
from semantic_digital_twin.adapters.grasp_clutter_6d_dataset.schema import (
    GraspClutter6DScene,
)

try:
    import huggingface_hub
except ImportError:
    logger.warning(
        "huggingface_hub not installed. `GraspClutter6DDatasetLoader` downloads will not "
        "work. Install it with `pip install huggingface_hub`."
    )
    huggingface_hub: Optional[ModuleType] = None

try:
    import py7zr
except ImportError:
    logger.warning(
        "py7zr not installed. `GraspClutter6DDatasetLoader` archive extraction will not "
        "work. Install it with `pip install py7zr`."
    )
    py7zr: Optional[ModuleType] = None


class GraspClutter6DModelVariant(StrEnum):
    """
    The dataset's model archives: millimeter- and meter-unit point-cloud (`.ply`) and mesh
    (`.obj`) variants, plus a simplified/watertight `EVAL` variant used for pose-error
    metrics. See :meth:`GraspClutter6DDatasetLoader.download_models`.
    """

    MODELS = "models"
    EVAL = "models_eval"
    METERS = "models_m"
    OBJ = "models_obj"
    OBJ_EVAL = "models_obj_eval"
    OBJ_METERS = "models_obj_m"


class GraspClutter6DObjectSet(StrEnum):
    """
    The two object catalogs GraspClutter6D's scenes are built from. See
    :meth:`GraspClutter6DDatasetLoader.available_scene_ids`.
    """

    GRASP = "grasp"
    """The dataset's own 200 novel objects."""

    YCBV = "ycbv"
    """The standard YCB-Video objects."""


class GraspClutter6DSplit(StrEnum):
    """
    The dataset's train/test split. See
    :meth:`GraspClutter6DDatasetLoader.available_scene_ids`.
    """

    TRAIN = "train"
    TEST = "test"


@dataclass
class GraspClutter6DDatasetLoader:
    """
    Loader for scenes of the GraspClutter6D dataset
    (https://huggingface.co/datasets/GraspClutter6D/GraspClutter6D), a dataset of 1000
    densely cluttered bin/shelf/table scenes (~14 objects/scene, 200 object models plus
    the standard YCB-Video objects), annotated with per-frame 6D object poses in the BOP
    dataset format (https://github.com/thodan/bop_toolkit).

    .. important::
        Unlike this package's other dataset loaders (e.g.
        :class:`~semantic_digital_twin.adapters.sage_10k_dataset.loader.Sage10kDatasetLoader`),
        GraspClutter6D does not store scenes as separate repository files - all 1000
        scenes are packed into one combined, 5-volume, ~203 GB `scenes.7z` archive, so
        there is no way to fetch a single scene without downloading and extracting that
        entire archive first. :meth:`download_split_info` and :meth:`download_models` are
        comparatively small (KB-GB) and safe to call freely; :meth:`download_scenes` is
        not.

    .. note::
        Requires the ``huggingface_hub`` and ``py7zr`` packages.
    """

    directory: Path = field(
        default_factory=lambda: Path.home() / "graspclutter6d-dataset"
    )
    """The directory archives are downloaded to and extracted in."""

    token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN"))
    """
    The Hugging Face access token used to download the dataset.

    The dataset is public, so this is only needed to raise Hugging Face's anonymous-
    access rate limit.
    """

    repository_id: str = "GraspClutter6D/GraspClutter6D"
    """The Hugging Face dataset repository ID."""

    _scenes_archive_parts: Tuple[str, ...] = field(
        default=(
            "scenes.7z.001",
            "scenes.7z.002",
            "scenes.7z.003",
            "scenes.7z.004",
            "scenes.7z.005",
        ),
        init=False,
        repr=False,
    )
    """
    The 5 volumes of the dataset's one combined scenes archive (~203 GB total), split
    because Hugging Face Hub caps individual file size.
    """

    def _download_archive(self, filename: str) -> Path:
        """
        :param filename: The archive's path within the repository, e.g.
            ``"models_eval.7z"``.
        :return: The path the archive was downloaded to. Returns early without a network
            call if it is already there.
        """
        return Path(
            huggingface_hub.hf_hub_download(
                repo_id=self.repository_id,
                repo_type="dataset",
                filename=filename,
                token=self.token,
                local_dir=self.directory,
            )
        )

    @staticmethod
    def _extract_archive(
        archive_path: Path, extraction_root: Path, expected_name: str
    ) -> Path:
        """
        Extract a (possibly multi-volume) 7z archive into `extraction_root`, returning
        the subdirectory the archive itself creates.

        Every one of this dataset's archives is packed with one top-level folder
        matching the archive's own name (confirmed for `split_info.7z` and
        `models_eval.7z`; assumed by the same naming convention for the others, and
        checked below rather than trusted blindly) - so this extracts straight into
        `extraction_root` instead of a pre-named subdirectory, which would otherwise
        double that path segment (e.g. `models_eval/models_eval/obj_000001.ply`).

        :param archive_path: The archive's first (or only) volume.
        :param extraction_root: The parent directory to extract into.
        :param expected_name: The top-level folder name the archive is expected to
            create.
        :raises GraspClutter6DArchiveLayoutError: if extraction does not produce
            `extraction_root / expected_name`.
        :return: `extraction_root / expected_name`. Returns early without extracting if
            this already exists.
        """
        expected_directory = extraction_root / expected_name
        if expected_directory.exists():
            return expected_directory
        if py7zr is None:
            raise ImportError(
                "py7zr is required to extract GraspClutter6D archives. Install it with "
                "`pip install py7zr`, or the semantic_digital_twin 'datasets' extra."
            )
        extraction_root.mkdir(parents=True, exist_ok=True)
        try:
            with py7zr.SevenZipFile(str(archive_path), mode="r") as archive:
                archive.extractall(path=str(extraction_root))
        except Exception:
            shutil.rmtree(expected_directory, ignore_errors=True)
            raise
        if not expected_directory.is_dir():
            raise GraspClutter6DArchiveLayoutError(
                archive_path=archive_path, expected_directory=expected_directory
            )
        return expected_directory

    def download_split_info(self) -> Path:
        """
        Download and extract `split_info.7z` (~12 KB): the train/test scene id lists used
        by :meth:`available_scene_ids`, and an `obj_ids_per_scene.json` used by
        :meth:`object_ids_for_scene`.

        :return: The extracted `split_info` directory.
        """
        archive_path = self._download_archive("split_info.7z")
        return self._extract_archive(archive_path, self.directory, "split_info")

    def download_models(
        self, variant: GraspClutter6DModelVariant = GraspClutter6DModelVariant.EVAL
    ) -> Path:
        """
        Download and extract one of the dataset's model archives.

        :param variant: Defaults to :attr:`~GraspClutter6DModelVariant.EVAL` (~87 MB,
            simplified/watertight meshes) as the cheapest variant to fetch; pass
            :attr:`~GraspClutter6DModelVariant.MODELS` for the full-detail millimeter-
            unit meshes, or :attr:`~GraspClutter6DModelVariant.METERS` for the meter-unit
            variant (pair this with ``mesh_unit_scale=1.0`` on
            :meth:`~semantic_digital_twin.adapters.grasp_clutter_6d_dataset.schema.GraspClutter6DScene.create_world`).
        :return: The extracted directory, containing `obj_%06d.ply`/`.obj` files and
            `models_info.json`.
        """
        archive_path = self._download_archive(f"{variant.value}.7z")
        return self._extract_archive(archive_path, self.directory, variant.value)

    def download_scenes(self) -> Path:
        """
        Download and extract the entire ~203 GB, 5-part `scenes.7z` archive.

        .. warning::
            There is no way to fetch a single scene from this dataset - see the class
            docstring. This downloads and extracts all 1000 scenes.

        :return: The extracted `scenes` directory.
        """
        expected_directory = self.directory / "scenes"
        if expected_directory.exists():
            return expected_directory
        part_paths = [
            self._download_archive(name) for name in self._scenes_archive_parts
        ]
        return self._extract_archive(part_paths[0], self.directory, "scenes")

    def available_scene_ids(
        self,
        object_set: GraspClutter6DObjectSet = GraspClutter6DObjectSet.GRASP,
        split: GraspClutter6DSplit = GraspClutter6DSplit.TRAIN,
    ) -> Tuple[str, ...]:
        """
        List the scene ids of one split, without downloading any scene data.

        :param object_set: Which object catalog the scenes are built from.
        :param split: Which split to list.
        :return: The scene ids of that split, e.g. ``("000005", "000009", ...)``.
        """
        split_info_directory = self.download_split_info()
        file_path = (
            split_info_directory / f"{object_set.value}_{split.value}_scene_ids.json"
        )
        return tuple(json.loads(file_path.read_text()))

    def object_ids_for_scene(self, scene_id: str) -> Tuple[int, ...]:
        """
        :param scene_id: A scene id, e.g. ``"000005"``.
        :return: The object ids present anywhere in that scene, from
            `split_info/obj_ids_per_scene.json`.

        .. note::
            That file keys scenes by their plain (non-zero-padded) integer index rather
            than the zero-padded id used everywhere else in the dataset (including
            :meth:`available_scene_ids`); this method converts between the two under the
            assumption that they correspond 1:1 in numeric order, which matches every
            scene id observed so far but is not documented by the dataset itself.
        """
        split_info_directory = self.download_split_info()
        file_path = split_info_directory / "obj_ids_per_scene.json"
        data = json.loads(file_path.read_text())
        return tuple(data[str(int(scene_id))])

    def load_scene(self, scene_id: str) -> GraspClutter6DScene:
        """
        Load one scene's annotations from the already-downloaded/extracted scenes
        directory (see :meth:`download_scenes`).

        Since this dataset's internal folder nesting under `scenes/` (e.g. by split
        and/or object set) is not documented anywhere fetchable without downloading the
        entire ~203 GB archive, this searches for it by name instead of assuming a fixed
        path.

        :param scene_id: The scene id to load, e.g. ``"000005"``.
        :raises GraspClutter6DSceneNotFoundError: if `scene_id` does not resolve to
            exactly one folder under the scenes directory.
        :return: The parsed scene.
        """
        scenes_directory = self.download_scenes()
        candidates = [
            path
            for path in scenes_directory.rglob(scene_id)
            if path.is_dir() and (path / "scene_camera.json").is_file()
        ]
        if len(candidates) != 1:
            raise GraspClutter6DSceneNotFoundError(
                scene_id=scene_id,
                scenes_directory=scenes_directory,
                candidates=tuple(candidates),
            )
        return GraspClutter6DScene.from_directory(
            scene_id=scene_id, directory=candidates[0]
        )
