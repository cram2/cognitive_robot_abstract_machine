"""
Tests for a simulator property's dictionary form, under the simulator's own names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.adapters.multi_sim import ContactCategories, MujocoGeom
from semantic_digital_twin.mixin import (
    SimulatorAdditionalProperty,
    SimulatorAttributeName,
)


@dataclass
class PropertyWithOneRenamedField(SimulatorAdditionalProperty):
    """
    A property of which one field is spelled differently by the simulator.
    """

    spelled_alike: int = 1

    spelled_differently: int = field(
        default=2, metadata=SimulatorAttributeName("diff").as_dict()
    )


def test_fields_appear_under_the_simulators_name_where_one_is_declared():
    assert PropertyWithOneRenamedField().to_dict() == {"spelled_alike": 1, "diff": 2}


def test_a_geom_carries_its_masks_under_mujocos_names():
    geom = MujocoGeom(
        contact_type=ContactCategories(2), contact_affinity=ContactCategories.DEFAULT
    )

    assert geom.to_dict() == {"contype": 2, "conaffinity": 1}


def test_contact_categories_combine_and_keep_unnamed_bits():
    both = ContactCategories.DEFAULT | ContactCategories(4)

    assert both == 5
    assert ContactCategories.DEFAULT in both
    assert ContactCategories(4) in both
    assert ContactCategories(8) not in both
