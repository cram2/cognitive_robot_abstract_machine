import logging
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import field
from pathlib import Path
from xml.etree import ElementTree as ET

import jinja2
from typing_extensions import List

from krrood.utils import recursive_subclasses
from semantic_digital_twin.adapters.package_resolver import FileUriResolver
from semantic_digital_twin.adapters.partnet_mobility_dataset.generated_semantic_annotations import *  # type: ignore
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

try:
    import sapien
except ImportError:
    logger.warn(
        "Sapien library is required for Partnet Mobility dataset loading. "
        "Please install it using 'pip install -U 'sapien>=3.0.0b1'"
    )

SAPIEN_ACCESS_TOKEN_ENVIRONMENT_VARIABLE_NAME = "SAPIEN_ACCESS_TOKEN"


@dataclass
class PartNetMobilityDatasetLoader:
    """
    Loader for articulated assets from the PartNet-Mobility dataset (https://sapien.ucsd.edu/browse).

    For this to work out of the box, the environment variable SAPIEN_ACCESS_TOKEN must be set,
    and you have to install sapien.

    The URDF files provided by sapien are missing some information. This loader also adds the missing information.
    The semantics are also added to the world, utilizing the PartNet-Mobility dataset semantics and the subclasses of
    PartNetLabel.
    """

    token: str = field(
        default_factory=lambda: os.environ[
            SAPIEN_ACCESS_TOKEN_ENVIRONMENT_VARIABLE_NAME
        ]
    )
    """
    The token to use for communication with the partnet server.
    """

    directory: Path = field(
        default_factory=lambda: Path.home() / "partnet-mobility-dataset"
    )
    """
    The directory where to save the downloaded URDF files into.
    """

    default_effort_for_limit_tags: float = 100.0
    """
    The default effort value to use for limit tags in the URDF files.
    """

    default_velocity_for_limit_tags: float = 100.0
    """
    The default velocity value to use for limit tags in the URDF files.
    """

    def load(self, model_id: int = 179) -> World:
        """
        Load a world given the model id.

        .. note::
            The PartNet model ids are distributed without any structure.
            Use the browser of PartNet (https://sapien.ucsd.edu/browse) to find the model id of the model you want to
            load. Or download the entire dataset and figure it out with `available_model_ids`.


        :param model_id: The id of the model to load.
        :return: The loaded world.
        """
        urdf_file = sapien.asset.download_partnet_mobility(
            model_id=model_id, token=self.token, directory=self.directory
        )
        self._add_missing_information_to_limit_tags(file_path=urdf_file)
        world = URDFParser.from_file(
            file_path=urdf_file,
            path_resolver=FileUriResolver(
                base_directory=self.directory / str(model_id)
            ),
        ).parse()
        self._apply_semantics_to_world(world, model_id)
        return world

    def _apply_semantics_to_world(self, world: World, model_id: int):
        """
        Apply the semantics of a model to the world by creating semantic annotations.

        :param world: The world to apply the semantics to.
        :param model_id: The id of the model to apply the semantics from.
        """
        semantics_file = self.directory / str(model_id) / "semantics.txt"

        link_labels = {}

        with semantics_file.open("r") as f:
            lines = f.readlines()

        for line in lines:
            link_name, _, label = line.strip().split(" ")
            link_labels[link_name] = label

        for link_name, label in link_labels.items():
            body = world.get_body_by_name(link_name)
            [part_net_semantic_annotations_class] = [
                cls for cls in recursive_subclasses(PartNetLabel) if label in cls.labels
            ]
            semantic_annotation = part_net_semantic_annotations_class(root=body)

            with world.modify_world():
                world.add_semantic_annotation(semantic_annotation)

    def _add_missing_information_to_limit_tags(self, file_path: str):
        """
        Add the missing information to all limit tags in a URDF file.

        :param file_path: Path to the URDF file.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        for limit in root.findall(".//limit"):
            limit.set("effort", str(self.default_effort_for_limit_tags))
            limit.set("velocity", str(self.default_velocity_for_limit_tags))
        tree.write(file_path)

    def _create_python_file_with_semantic_annotations_from_dataset(self):
        """
        Write all semantic annotations classes from the dataset to a Python file.
        This requires the entire dataset to be located in the specified directory.
        Only call this when you want to generate/update the `generated_semantic_annotations.py` file.
        """

        labels = defaultdict(set)  # a dict of class names to sets of labels

        # collect all labels from all semantics.txt files
        for directory in self.directory.glob("*"):
            semantics_file = directory / "semantics.txt"

            with semantics_file.open("r") as f:
                lines = f.readlines()

            for line in lines:
                _, _, label = line.strip().split(" ")
                labels[self.class_name_from_label(label)].add(label)

        # generate class descriptions
        class_descriptions = [
            PartNetLabelClassDescription(class_name=class_name, labels=labels)
            for class_name, labels in labels.items()
        ]

        # generate the semantic annotations file
        package_directory = os.path.join(os.path.dirname(__file__))
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(package_directory),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Load the template
        template = env.get_template("partnet_annotations_template.jinja")

        # Render the template
        output = template.render(
            class_descriptions=class_descriptions,
        )

        with open(
            os.path.join(package_directory, "generated_semantic_annotations.py"), "w"
        ) as f:

            # Write the output to the file
            f.write(output)

        # format the output with black
        command = [sys.executable, "-m", "black", str(f.name)]
        subprocess.run(command, capture_output=True, text=True, check=True)

    @classmethod
    def class_name_from_label(cls, label: str) -> str:
        # remove _body
        label = label.replace("_body", "")

        # convert to upper camel case
        label = label.title().replace("_", "")
        return label

    @property
    def available_model_ids(self) -> List[int]:
        """
        :return: A list of all downloaded models ids.
        """
        return [int(directory.name) for directory in self.directory.glob("*")]


@dataclass
class PartNetLabelClassDescription:
    """
    Class description for a PartNet label.
    """

    class_name: str
    """
    Class name of the label.
    """

    labels: Set[str]
    """
    Set of labels for the class.
    """
