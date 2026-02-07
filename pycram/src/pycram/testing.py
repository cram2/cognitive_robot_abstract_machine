import logging
import os
import time
import unittest
from copy import deepcopy

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive
from .datastructures.dataclasses import Context
from .plan import Plan

logger = logging.getLogger(__name__)

try:
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
except ImportError:
    logger.info(
        "Could not import VizMarkerPublisher. This is probably because you are not running ROS."
    )


def setup_world() -> World:
    logger.setLevel(logging.DEBUG)

    pr2_sem_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "robots",
            "pr2_calibrated_with_ft.urdf",
        )
    ).parse()
    apartment_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "worlds",
            "apartment.urdf",
        )
    ).parse()
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "milk.stl"
        )
    ).parse()
    cereal_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "objects",
            "breakfast_cereal.stl",
        )
    ).parse()
    # apartment_world.merge_world(pr2_sem_world)
    apartment_world.merge_world(milk_world)
    apartment_world.merge_world(cereal_world)

    with apartment_world.modify_world():
        pr2_root = pr2_sem_world.get_body_by_name("base_footprint")
        apartment_root = apartment_world.root
        c_root_bf = OmniDrive.create_with_dofs(
            parent=apartment_root, child=pr2_root, world=apartment_world
        )
        apartment_world.merge_world(pr2_sem_world, c_root_bf)
        c_root_bf.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.5, 0)

    apartment_world.get_body_by_name("milk.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=apartment_world.root
        )
    )
    apartment_world.get_body_by_name(
        "breakfast_cereal.stl"
    ).parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2.37, 1.8, 1.05, reference_frame=apartment_world.root
    )
    milk_view = Milk(root=apartment_world.get_body_by_name("milk.stl"))
    with apartment_world.modify_world():
        apartment_world.add_semantic_annotation(milk_view)

    return apartment_world


class SemanticWorldTestCase(unittest.TestCase):
    world: World

    @classmethod
    def setUpClass(cls):
        cls.pr2_sem_world = URDFParser(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "resources",
                "robots",
                "pr2_calibrated_with_ft.urdf",
            )
        ).parse()
        cls.apartment_world = URDFParser(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "resources",
                "worlds",
                "apartment.urdf",
            )
        ).parse()
        cls.apartment_world.merge_world(cls.pr2_sem_world)
