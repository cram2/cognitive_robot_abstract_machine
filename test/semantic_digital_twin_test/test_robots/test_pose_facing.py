import importlib
import pkgutil

import numpy as np
import pytest
from typing_extensions import List, Tuple, Type

import semantic_digital_twin.robots as robots_package
from krrood.entity_query_language.factories import (
    an,
    and_,
    distinct,
    entity,
    symbolic_function,
    variable_from,
)
from krrood.utils import get_generic_type_parameters, recursive_subclasses
from semantic_digital_twin.adapters.package_resolver import CompositePathResolver
from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import AbstractRobot, MobileBase
from semantic_digital_twin.spatial_types.spatial_types import Pose, Vector3
from semantic_digital_twin.world import World

# %% the forward axes the declared robots use


def axis_components(axis: Vector3) -> Tuple[float, float, float]:
    """
    The three components of ``axis``.

    :class:`~semantic_digital_twin.spatial_types.spatial_types.Vector3` compares by
    identity and cannot be hashed, so an axis needs this form to be grouped or matched
    by the direction it points in.
    """
    x, y, z = axis.to_np().flatten()[:3]
    return float(x), float(y), float(z)


def mobile_base_type(robot_type: Type[AbstractRobot]) -> Type[MobileBase]:
    """
    The mobile base ``robot_type`` stands on.
    """
    return get_generic_type_parameters(robot_type, HasMobileBase)[0]


@symbolic_function
def forward_axis_components(
    robot_type: Type[AbstractRobot],
) -> Tuple[float, float, float]:
    """
    The direction the robot's base calls its front.
    """
    return axis_components(mobile_base_type(robot_type).forward_axis)


@symbolic_function
def points_along(robot_type: Type[AbstractRobot], axis: Vector3) -> bool:
    """
    Whether the robot's base calls ``axis`` its front.
    """
    return axis_components(
        mobile_base_type(robot_type).forward_axis
    ) == axis_components(axis)


@symbolic_function
def description_is_available(robot_type: Type[AbstractRobot]) -> bool:
    """
    Whether this machine carries the description needed to spawn the robot.

    Which descriptions are installed varies, and a robot that cannot be spawned cannot
    stand in for the axis it declares.
    """
    try:
        CompositePathResolver().resolve(robot_type.get_ros_file_path())
    except (ParsingError, NotImplementedError):
        return False
    return True


def robot_types_with_a_mobile_base() -> List[Type[AbstractRobot]]:
    """
    Every robot this repository declares that stands on a mobile base.

    The robot modules are imported first, because a class nothing has imported is a
    subclass nobody can find. Sorting keeps which robot stands in for a shared axis from
    depending on the order something else happened to import them in.
    """
    for module in pkgutil.iter_modules(robots_package.__path__):
        importlib.import_module(f"{robots_package.__name__}.{module.name}")
    return sorted(
        (
            robot_type
            for robot_type in recursive_subclasses(AbstractRobot)
            if issubclass(robot_type, HasMobileBase)
        ),
        key=lambda declared: declared.__name__,
    )


ROBOT_TYPES_WITH_A_MOBILE_BASE = robot_types_with_a_mobile_base()
"""
Every robot that drives on a mobile base, whether or not its description is installed.
"""

ROBOTS_WITH_DISTINCT_FORWARD_AXES = distinct(
    an(
        entity(candidate := variable_from(ROBOT_TYPES_WITH_A_MOBILE_BASE)).where(
            description_is_available(candidate)
        )
    ),
    forward_axis_components(candidate),
).tolist()
"""
One spawnable robot per forward axis in use, so each axis is exercised once rather than
once per robot that happens to share it.
"""

X_FORWARD_ROBOTS = an(
    entity(x_forward_candidate := variable_from(ROBOT_TYPES_WITH_A_MOBILE_BASE)).where(
        and_(
            description_is_available(x_forward_candidate),
            points_along(x_forward_candidate, Vector3.X()),
        )
    )
).tolist()
"""
The spawnable robots whose base already calls the x-axis its front.
"""


def spawn(robot_type: Type[AbstractRobot]) -> MobileBase:
    """
    The mobile base of ``robot_type``, alone in a world of its own.
    """
    world = World.create_with_root_body("root")
    robot = RobotSpecification(semantic_annotation_type=robot_type).spawn(world)
    return robot.mobile_base


def robot_name(robot_type: Type[AbstractRobot]) -> str:
    """
    Name a parametrized case after the robot it runs against.
    """
    return robot_type.__name__


@pytest.fixture()
def x_forward_robot() -> Type[AbstractRobot]:
    """
    A robot whose front is the x-axis, which needs no correction to face a heading.
    """
    if not X_FORWARD_ROBOTS:
        pytest.skip("no robot with an x-forward base can be spawned here")
    return X_FORWARD_ROBOTS[0]


def test_a_robot_is_selected_for_every_forward_axis_that_can_be_spawned():
    """
    The parametrized cases below stand in for every axis the declared robots use, which
    only holds while the selection misses none of them.
    """
    axes_in_use = {
        forward_axis_components(robot_type)
        for robot_type in ROBOT_TYPES_WITH_A_MOBILE_BASE
        if description_is_available(robot_type)
    }

    assert axes_in_use
    assert {
        forward_axis_components(robot_type)
        for robot_type in ROBOTS_WITH_DISTINCT_FORWARD_AXES
    } == axes_in_use


# %% the forward axis ends up pointing along the heading


@pytest.mark.parametrize(
    "robot_type", ROBOTS_WITH_DISTINCT_FORWARD_AXES, ids=robot_name
)
@pytest.mark.parametrize("heading_yaw", [0.0, np.pi / 2, -np.pi / 2, np.pi, 0.3])
def test_forward_axis_points_along_the_heading(
    robot_type: Type[AbstractRobot], heading_yaw: float
):
    mobile_base = spawn(robot_type)
    heading = Pose.from_xyz_rpy(
        1.3, 2.0, 0.0, yaw=heading_yaw, reference_frame=mobile_base.root._world.root
    )

    base_pose = mobile_base.pose_facing(heading)

    world_V_forward = base_pose.to_rotation_matrix() @ mobile_base.forward_axis
    np.testing.assert_allclose(
        world_V_forward.to_np()[:3].flatten(),
        [np.cos(heading_yaw), np.sin(heading_yaw), 0.0],
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "robot_type", ROBOTS_WITH_DISTINCT_FORWARD_AXES, ids=robot_name
)
def test_the_position_is_the_headings_own(robot_type: Type[AbstractRobot]):
    mobile_base = spawn(robot_type)
    heading = Pose.from_xyz_rpy(
        1.3, 2.0, 0.0, yaw=0.3, reference_frame=mobile_base.root._world.root
    )

    base_pose = mobile_base.pose_facing(heading)

    np.testing.assert_allclose(
        base_pose.to_position().to_np(), heading.to_position().to_np(), atol=1e-9
    )
    assert base_pose.reference_frame is heading.reference_frame


# %% a base whose front is already the x-axis needs no correction


def test_an_x_forward_base_takes_the_heading_unchanged(
    x_forward_robot: Type[AbstractRobot],
):
    mobile_base = spawn(x_forward_robot)
    heading = Pose.from_xyz_rpy(
        1.3, 2.0, 0.0, yaw=0.3, reference_frame=mobile_base.root._world.root
    )

    np.testing.assert_allclose(
        mobile_base.pose_facing(heading).to_np(), heading.to_np(), atol=1e-9
    )
