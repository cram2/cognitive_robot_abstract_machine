"""
The world SegMind's tests watch.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

RESTING_ON_THE_TABLE = (-1.7, 0.0, 0.93)
"""
Where the milk stands resting on the apartment's second box.
"""

WHERE_THE_MILK_STOOD = (-1.7, 0.0, 1.07)
"""
Where the milk stands when the apartment is built, which a test puts it back to.
"""


@pytest.fixture
def milk_in_the_apartment(_simple_apartment_setup):
    """
    The apartment with its milk and boxes, the milk put back where it stood afterwards.
    """
    world = _simple_apartment_setup
    milk = world.get_body_by_name("milk.stl")
    yield world, milk, world.get_body_by_name("box")
    x, y, z = WHERE_THE_MILK_STOOD
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x, y, z, yaw=np.pi, reference_frame=milk.parent_connection.parent
    )


@pytest.fixture
def milk_annotated_in_the_apartment(milk_in_the_apartment):
    """
    The apartment with its milk annotated as :class:`Milk`, the annotation removed
    afterwards.
    """
    world, milk, box = milk_in_the_apartment
    annotation = Milk(root=milk)
    with world.modify_world():
        world.add_semantic_annotation(annotation)
    yield world, annotation, box
    with world.modify_world():
        world.remove_semantic_annotation(annotation)
