"""
Tests for where the bullet world demo puts its place setting: each object starts resting
on something and is laid resting on the table, so what watches the run sees it taken up
and put down.
"""

from __future__ import annotations

import importlib.util
from enum import StrEnum
from pathlib import Path

from semantic_digital_twin.api import WorldSpecification
from semantic_digital_twin.reasoning.predicates import SupportedBy
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world_description.connections import FixedConnection

DEMO_PATH = (
    Path(__file__).resolve().parents[2]
    / "coraplex"
    / "demos"
    / "coraplex_bullet_world_demo"
    / "demo.py"
)
"""
The bullet world demo, which lives outside any package and is loaded from its file.
"""


class ApartmentSurface(StrEnum):
    """
    The apartment's bodies the place setting rests on.
    """

    COUNTER = "island_countertop"
    SPOON_DRAWER = "cabinet10_drawer_top"
    TABLE = "table_area_main"


def test_the_place_setting_rests_where_it_starts_and_where_it_is_laid():
    specification = importlib.util.spec_from_file_location(
        "bullet_world_demo", DEMO_PATH
    )
    demo = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(demo)
    demonstration = demo.BulletWorldDemonstration(used_robot=PR2)
    world = WorldSpecification.from_urdf(
        demo.SceneFile.APARTMENT.path
    ).to_domain_object()
    milk, bowl, spoon = demonstration.milk, demonstration.bowl, demonstration.spoon
    for placed_object in demonstration.place_setting:
        placed_object.spawn(world)
    counter = world.get_body_by_name(ApartmentSurface.COUNTER)
    spoon_drawer = world.get_body_by_name(ApartmentSurface.SPOON_DRAWER)
    table = world.get_body_by_name(ApartmentSurface.TABLE)

    assert SupportedBy(milk.annotation_in(world).root, counter)()
    assert SupportedBy(bowl.annotation_in(world).root, counter)()
    assert SupportedBy(spoon.annotation_in(world).root, spoon_drawer)()

    for placed_object in demonstration.place_setting:
        body = placed_object.annotation_in(world).root
        target = placed_object.target_location(world).homogeneous_matrix
        with world.modify_world():
            world.remove_connection(body.parent_connection)
            world.add_connection(
                FixedConnection(
                    parent=world.root, child=body, parent_T_connection_expression=target
                )
            )

    assert SupportedBy(milk.annotation_in(world).root, table)()
    assert SupportedBy(bowl.annotation_in(world).root, table)()
    assert SupportedBy(spoon.annotation_in(world).root, table)()
