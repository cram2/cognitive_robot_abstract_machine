from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

MENAGERIE_ASSET_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie"
    "/{revision}/franka_emika_panda/assets/{filename}"
)
"""
Template for one Panda mesh in the ``mujoco_menagerie`` repository.
"""


@dataclass
class PandaMeshAssets:
    """
    The Panda meshes a scene needs, downloaded from ``mujoco_menagerie`` on
    first use.

    The meshes are several tens of megabytes, so they are fetched on demand
    rather than committed alongside the scene that references them.
    """

    scene: Path
    """
    The MJCF scene whose mesh references decide what has to be downloaded.
    """

    revision: str = "main"
    """
    Git revision of ``mujoco_menagerie`` to download from.

    Pin this to a commit to keep a scene reproducible against upstream changes.
    """

    timeout: float = 60.0
    """
    Seconds to allow for a single mesh download.
    """

    session: requests.Session = field(default_factory=requests.Session)
    """
    Connection pool reused across the individual mesh downloads.
    """

    @property
    def directory(self) -> Path:
        """
        Where the meshes belong, taken from the scene's own ``meshdir`` so the
        two cannot disagree about it.
        """
        compiler = ElementTree.parse(self.scene).getroot().find("compiler")
        mesh_directory = compiler.get("meshdir")
        return self.scene.parent / mesh_directory

    def required_filenames(self) -> list[str]:
        """
        The mesh files the scene refers to, in the order it declares them.
        """
        root = ElementTree.parse(self.scene).getroot()
        filenames = [mesh.get("file") for mesh in root.iter("mesh")]
        return sorted({filename for filename in filenames if filename})

    def download_if_missing(self) -> Path:
        """
        Download whichever of the scene's meshes are not present yet.

        :return: The directory holding the meshes.
        """
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)

        missing = [
            filename
            for filename in self.required_filenames()
            if not (directory / filename).exists()
        ]
        if not missing:
            return directory

        logger.info(
            "Downloading %d Panda meshes from mujoco_menagerie@%s into %s",
            len(missing),
            self.revision,
            directory,
        )
        for filename in missing:
            url = MENAGERIE_ASSET_URL.format(
                revision=self.revision, filename=filename
            )
            response = self.session.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            # Written to a temporary name first so an interrupted download
            # cannot leave a truncated mesh that later runs take for complete.
            partial = directory / f"{filename}.partial"
            with partial.open("wb") as mesh_file:
                for chunk in response.iter_content(chunk_size=8192):
                    mesh_file.write(chunk)
            partial.rename(directory / filename)

        return directory
