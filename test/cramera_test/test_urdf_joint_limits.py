"""
Native joint bounds are represented faithfully or rejected before URDF export.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest
from urdf_parser_py.urdf import URDF

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body

from cramera.onboard.world_to_urdf import UnrepresentableJointLimits, UrdfDocument


# %% native joint fixture
@pytest.fixture
def joint_world() -> World:
    """
    Build one native prismatic connection with explicit finite source limits.
    """
    world = World()
    parent = Body(name=PrefixedName("parent"))
    child = Body(name=PrefixedName("child"))
    degree = DegreeOfFreedom(
        name=PrefixedName("slider"),
        limits=DegreeOfFreedomLimits(
            lower=DerivativeMap(position=-0.5, velocity=-0.2),
            upper=DerivativeMap(position=1.5, velocity=0.2),
        ),
    )
    with world.modify_world():
        world.add_degree_of_freedom(degree)
        world.add_connection(
            PrismaticConnection(
                parent=parent, child=child, raw_dof=degree, axis=Vector3(1, 0, 0)
            )
        )
    return world


@pytest.mark.parametrize(
    "lower,upper",
    [
        (None, None),
        (None, 1.0),
        (-1.0, None),
        (float("-inf"), 1.0),
        (-1.0, float("inf")),
        (float("nan"), 1.0),
    ],
)
def test_unrepresentable_prismatic_positions_are_rejected(
    joint_world: World, tmp_path: Path, lower: float | None, upper: float | None
) -> None:
    """
    URDF never receives absent or nonfinite prismatic position bounds.

    :param joint_world: Native world containing a slider.
    :param tmp_path: Export directory.
    :param lower: Source lower position bound.
    :param upper: Source upper position bound.
    """
    connection = joint_world.get_body_by_name("child").parent_connection
    connection.raw_dof.limits.lower.position = lower
    connection.raw_dof.limits.upper.position = upper
    with pytest.raises(UnrepresentableJointLimits) as failure:
        UrdfDocument.of_world(joint_world, "slider", str(tmp_path), "slider")
    assert failure.value.joint_name == str(connection.name)
    assert not (tmp_path / "slider.urdf").exists()


def test_finite_limits_round_trip_through_standard_urdf(
    joint_world: World, tmp_path: Path
) -> None:
    """
    The export preserves the native joint's actual bounds and velocity.

    :param joint_world: Native world containing a slider.
    :param tmp_path: Export directory.
    """
    connection = joint_world.get_body_by_name("child").parent_connection
    report = UrdfDocument.of_world(joint_world, "slider", str(tmp_path), "slider")
    joint = URDF.from_xml_file(report.urdf).joints[0]
    assert joint.limit.lower == connection.dof.limits.lower.position
    assert joint.limit.upper == connection.dof.limits.upper.position
    assert joint.limit.velocity == connection.dof.limits.upper.velocity


@pytest.mark.parametrize("partial", [False, True])
def test_revolute_limits_distinguish_continuous_from_partial(
    joint_world: World, tmp_path: Path, partial: bool
) -> None:
    """
    Only truly unbounded rotation is represented as a continuous URDF joint.

    :param joint_world: Native world whose slider is replaced with a rotary joint.
    :param tmp_path: Export directory.
    :param partial: Whether a one-sided position bound must be rejected.
    """
    previous = joint_world.get_body_by_name("child").parent_connection
    previous.raw_dof.limits.lower.position = None
    previous.raw_dof.limits.upper.position = 1.0 if partial else None
    with joint_world.modify_world():
        joint_world.remove_connection(previous)
        joint_world.add_connection(
            RevoluteConnection(
                parent=previous.parent,
                child=previous.child,
                raw_dof=previous.raw_dof,
                axis=Vector3(0, 0, 1),
            )
        )
    if partial:
        with pytest.raises(UnrepresentableJointLimits):
            UrdfDocument.of_world(joint_world, "rotary", str(tmp_path), "rotary")
    else:
        report = UrdfDocument.of_world(joint_world, "rotary", str(tmp_path), "rotary")
        assert ElementTree.parse(report.urdf).find("joint").get("type") == "continuous"
