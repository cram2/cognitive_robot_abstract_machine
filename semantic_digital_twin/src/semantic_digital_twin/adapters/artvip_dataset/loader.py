from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from typing_extensions import Optional

from semantic_digital_twin.adapters.artvip_dataset.exceptions import (
    ArtVipMainStageFileAmbiguousError,
    ArtVipObjectNotFoundError,
)
from semantic_digital_twin.adapters.artvip_dataset.schema import (
    ArtVipCategory,
    ArtVipObject,
)
from semantic_digital_twin.adapters.usd.parser import USDParser
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)

logger = logging.getLogger(__name__)

try:
    import pxr  # noqa: F401
except ImportError:
    logger.warning(
        "usd-core is required for ArtVIP dataset loading. "
        "Please install it using 'pip install usd-core'"
    )


@dataclass
class ArtVipDatasetLoader:
    """
    Loader for professionally modelled, articulated digital-twin objects from the ArtVIP
    dataset (https://x-humanoid-artvip.github.io/), including a dedicated IKEA furniture
    category.

    ArtVIP's meshes are clean, hand-authored CAD, decomposed into rigid links connected
    by real USD Physics joints
    (``UsdPhysics.FixedJoint``/``RevoluteJoint``/``PrismaticJoint``), each carrying an
    authored axis, position, and limits. This loader downloads one object's files from
    its Hugging Face repository and parses its main USD stage with
    :class:`~semantic_digital_twin.adapters.usd.parser.USDParser`.

    .. note::
        Requires the ``usd-core`` package (``pxr``) to read the object's USD stage.
    """

    directory: Path = field(default_factory=lambda: Path.home() / "artvip-dataset")
    """
    The directory object files are downloaded to.
    """

    token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN"))
    """
    The Hugging Face access token used to download the dataset.

    The dataset is public, so this is only needed to raise Hugging Face's anonymous-
    access rate limit.
    """

    repository_id: str = "x-humanoid-robomind/ArtVIP"
    """
    The Hugging Face dataset repository ID.
    """

    _repository_files: Optional[tuple[str, ...]] = field(
        default=None, init=False, repr=False
    )
    """
    Every file path in the dataset repository, cached here after the first listing so
    that loading many objects (e.g. sweeping the whole catalog) only lists the
    repository's ~5000+ files once instead of once per :meth:`available_objects`/
    :meth:`load` call.
    """

    def available_objects(self, category: ArtVipCategory) -> tuple[str, ...]:
        """
        :param category: The category to list objects of.
        :return: Every object name in ``category`` - the path (relative to the
            category, ``/``-separated) of each directory that directly contains a
            top-level USD stage file. Most objects are one path segment deep; some
            categories nest an extra subcategory level (e.g. a name of
            ``"refrigerator/fridge_01"``), in which case the returned name includes
            that segment too.
        """
        prefix = f"Articulated_objects/{category.value}/"
        return self._object_names(self._list_repository_files(), prefix)

    @staticmethod
    def _object_names(files: tuple[str, ...], prefix: str) -> tuple[str, ...]:
        """
        :param files: Every file path in the dataset repository.
        :param prefix: The category's path prefix, e.g.
            ``"Articulated_objects/small_furniture/"``.
        :return: The name of every object directory directly under ``prefix``, found by
            looking for a top-level (not nested under a ``resource`` directory) ``.usd``
            file rather than assuming a fixed nesting depth or file-naming convention.
        """
        names = set()
        for file_path in files:
            if not file_path.startswith(prefix):
                continue
            segments = file_path[len(prefix) :].split("/")
            if len(segments) < 2 or "resource" in segments[:-1]:
                continue
            if not segments[-1].endswith(".usd"):
                continue
            names.add("/".join(segments[:-1]))
        return tuple(sorted(names))

    def _list_repository_files(self) -> tuple[str, ...]:
        """
        :return: Every file path in the dataset repository, listing the repository only
            on the first call and reusing that listing afterward.
        """
        if self._repository_files is None:
            self._repository_files = tuple(
                HfApi(token=self.token).list_repo_files(
                    self.repository_id, repo_type="dataset"
                )
            )
        return self._repository_files

    def load(self, category: ArtVipCategory, name: str) -> ArtVipObject:
        """
        Load one object as a World, with joints for its links.

        :param category: The object's category.
        :param name: The object's folder name, e.g.
            ``"EKET_Cabinet_with_door_brown_walnut_effect_35x35x35cm"``.
        :return: The loaded object.
        :raises ArtVipObjectNotFoundError: if no dataset entry matches category and name.
        """
        main_usd_path = self._download_object_if_not_exists(category, name)
        world = USDParser.from_file(str(main_usd_path), prefix=name).parse()
        with world.modify_world():
            world.add_semantic_annotation(
                NaturalLanguageWithTypeDescription(
                    root=world.root, description=name, type_description=category.value
                )
            )
        return ArtVipObject(world=world, category=category, name=name)

    def _download_object_if_not_exists(
        self, category: ArtVipCategory, name: str
    ) -> Path:
        """
        Download every file of one object if not already present, preserving the
        dataset's own relative file layout so the object's USD stage's relative
        references resolve.

        :param category: The object's category.
        :param name: The object's folder name.
        :return: The path to the object's main USD file.
        :raises ArtVipObjectNotFoundError: if no dataset entry matches category and
            name.
        :raises ArtVipMainStageFileAmbiguousError: if the object's directory does not
            contain exactly one top-level USD file to treat as its main stage.
        """
        prefix = f"Articulated_objects/{category.value}/{name}/"
        object_files = [
            file_path
            for file_path in self._list_repository_files()
            if file_path.startswith(prefix)
        ]
        if not object_files:
            raise ArtVipObjectNotFoundError(category=category, name=name)

        main_relative_path = self._main_stage_file(
            object_files, prefix, category=category, name=name
        )

        main_usd_path = None
        for file_path in object_files:
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=self.repository_id,
                    repo_type="dataset",
                    filename=file_path,
                    token=self.token,
                    local_dir=self.directory,
                )
            )
            if file_path == main_relative_path:
                main_usd_path = downloaded_path

        logger.info(
            f"Downloaded ArtVIP object {category.value}/{name} to {self.directory}"
        )
        return main_usd_path

    @staticmethod
    def _main_stage_file(
        object_files: list[str],
        prefix: str,
        *,
        category: ArtVipCategory,
        name: str,
    ) -> str:
        """
        Pick out an object's main USD stage file: the one ``.usd`` file directly inside
        its directory (not one of its ``resource/`` reference files). Objects do not
        agree on a single naming convention - some use ``model_<name>.usd``, others
        reuse the object's own name or an unrelated one (e.g.
        ``cabinet_1/cabinet_1.usd``, ``Collected_AirFryer1/AirFryer1.usd``) - so the
        directory position, not the file name, is what identifies it.

        :param object_files: Every file path under the object's directory.
        :param prefix: The object's directory path prefix.
        :param category: The object's category, for the error if this is ambiguous.
        :param name: The object's name, for the error if this is ambiguous.
        :return: The main USD file's path.
        :raises ArtVipMainStageFileAmbiguousError: if there is not exactly one such
            file.
        """
        candidates = [
            file_path
            for file_path in object_files
            if "/" not in file_path[len(prefix) :] and file_path.endswith(".usd")
        ]
        if len(candidates) != 1:
            raise ArtVipMainStageFileAmbiguousError(
                category=category, name=name, candidates=tuple(candidates)
            )
        return candidates[0]
