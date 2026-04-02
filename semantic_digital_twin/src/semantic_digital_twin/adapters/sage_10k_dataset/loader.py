from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

import requests

from semantic_digital_twin.adapters.sage_10k_dataset.schema import Sage10kScene


@dataclass
class Sage10kDatasetLoader:
    """
    Loader for scenes from the Sage10k dataset.
    This loader currently does not load Windows of walls.
    """

    directory: Path = field(default_factory=lambda: Path.home() / "sage-10k-scenes")
    """
    The directory where the scene should be downloaded to.
    """

    def _download_scene_if_not_exists(self, scene_url: str) -> Path:
        """
        Download the scene from the Sage10k dataset and unzip it.
        Returns early if a directory with the requested scene already exists.

        :return: The path to the unzipped scene.
        """
        self.directory.mkdir(parents=True, exist_ok=True)

        filename = Path(urlparse(scene_url).path).name
        zipped_scene = self.directory / filename
        extraction_directory = self.directory / zipped_scene.stem

        # return early if the scene exists already
        if extraction_directory.exists():
            return extraction_directory

        # download the scene
        with requests.get(scene_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with zipped_scene.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        # unzip the scene
        extraction_directory.mkdir(parents=True, exist_ok=True)
        with ZipFile(zipped_scene, "r") as zip_ref:
            zip_ref.extractall(extraction_directory)

        os.remove(str(zipped_scene))

        return extraction_directory

    def _parse_json(self, extracted_dir: Path) -> Sage10kScene:
        """
        Parses the extracted directory to locate and load a specific JSON file, ensuring there is
        exactly one valid file matching the naming pattern. Load the JSON into a Sage10kScene object.

        :param extracted_dir: The directory containing the extracted files to be parsed.
        :return: A Sage10kScene object created from the parsed JSON content. The object's
            `directory_path` attribute is also updated to the given `extracted_dir`.
        """
        json_files = list(extracted_dir.glob("layout_*.json"))
        if not json_files:
            raise ValueError(f"JSON file not found in {extracted_dir}")
        elif len(json_files) > 1:
            raise ValueError(f"Multiple JSON files found in {extracted_dir}")
        json_file = json_files[0]

        raw_json = json_file.read_text()
        json_dict = json.loads(raw_json)
        result = Sage10kScene._from_json(json_dict)
        result.directory = extracted_dir

        return result

    def _delete_assets(self, extracted_dir: Path):
        """
        Delete the assets of a scene.
        Use this when you only want to fetch all layout JSONS.

        :param extracted_dir: The directory containing the extracted scene.
        """
        objects = extracted_dir / "objects"
        preview = extracted_dir / "preview"
        materials = extracted_dir / "materials"

        for directory in [objects, preview, materials]:
            shutil.rmtree(directory)

    def create_scene(self, scene_url: str) -> Sage10kScene:
        """
        Create a scene from the given URL by downloading it and loading it into the memory.

        :param scene_url: The URL of the scene to be loaded.
        :return: The Sage10kScene object.
        """
        unzipped_scene = self._download_scene_if_not_exists(scene_url)
        scene = self._parse_json(unzipped_scene)
        return scene

    @classmethod
    def available_scenes(
        cls, repository: str = "nvidia/SAGE-10k", folder_path: str = "scenes"
    ) -> list[str]:
        """
        Use this to select random scenes from the dataset.
        Requires the extra requirement huggingface_hu.

        :param repository: The repo id of the dataset.
        :param folder_path: The path to the folder containing the scenes in the repository.
        :return: A list of all possible URLs to the scenes in the dataset.
        """
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()

        # Hugging Face filesystem paths follow the format: datasets/repo_id/path
        full_path = f"datasets/{repository}/{folder_path}"

        # List all files (glob '**/*' to get nested files if needed)
        files = fs.glob(f"{full_path}/**/*")

        # Filter out directories and convert to "resolve" URLs
        base_url = f"https://huggingface.co/datasets/{repository}/resolve/main"

        urls = []
        for file in files:
            # fs.glob returns paths like 'datasets/nvidia/SAGE-10k/scenes/file.ext'
            # We need to strip the 'datasets/repo_id/' prefix
            relative_path = file.replace(f"datasets/{repository}/", "")
            urls.append(f"{base_url}/{relative_path}")

        return urls
