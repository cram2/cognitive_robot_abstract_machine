"""
Tests for exporting worlds whose entities share names to MuJoCo, which identifies every
body, joint and actuator by a name unique among its kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import mujoco
import pytest

from ...pytest_environment import runs_in_continuous_integration

from semantic_digital_twin.adapters.multi_sim import (
    MujocoBuilder,
    MujocoCamera,
    MujocoSim,
    SimulatorElementNames,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import ServoGains
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, PositionServo

# %% scenes


def box_body(name: PrefixedName, x: float = 0.0, z: float = 0.0) -> Body:
    """
    :param name: The name of the body.
    :param x: How far in front of the body's origin the box sits.
    :param z: How far above the body's origin the box sits.
    :return: A body holding one small box, colliding and visible.
    """
    body = Body(name=name)
    geometry = ShapeCollection(
        [
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=x, z=z, reference_frame=body
                ),
                scale=Scale(0.1, 0.02, 0.02),
            )
        ],
        reference_frame=body,
    )
    body.collision, body.visual = geometry, geometry
    return body


@dataclass
class Pendulum:
    """
    A servoed hinge with a second hinge mimicking it.
    """

    arm: Body
    """
    The body the servoed hinge moves.
    """

    hinge: RevoluteConnection
    """
    The servoed hinge.
    """

    mirrored_hinge: RevoluteConnection
    """
    The hinge following :attr:`hinge` with a negative multiplier.
    """

    servo: PositionServo
    """
    The servo driving :attr:`hinge`.
    """


@dataclass
class TwinPendulumWorld:
    """
    Two pendulums whose bodies, joints and servos carry the same names.
    """

    world: World
    """
    The world holding both pendulums.
    """

    pendulums: list[Pendulum]
    """
    The two pendulums, in the order they were added.
    """


def add_pendulum(world: World, root: Body, sideways_offset: float) -> Pendulum:
    """
    :param world: The world to add the pendulum to.
    :param root: The body the pendulum hangs from.
    :param sideways_offset: How far the pendulum hangs to the side of the root, so
        pendulums never touch each other.
    :return: The added pendulum, named the same as every other pendulum.
    """
    base = Body(name=PrefixedName("base"))
    # the two arms hang at different heights so they never touch each other
    arm = box_body(PrefixedName("arm"), x=0.1)
    mirrored_arm = box_body(PrefixedName("mirrored_arm"), x=0.1, z=0.1)
    degree_of_freedom = DegreeOfFreedom(
        name=PrefixedName("hinge"),
        limits=DegreeOfFreedomLimits.from_position_range_and_speed(
            lower_position=-1.0, upper_position=1.0
        ),
    )
    servo = PositionServo(
        name=PrefixedName("hinge_servo"),
        gains=ServoGains(stiffness=50.0, damping=5.0, torque_limit=2.0),
    )
    servo.add_dof(degree_of_freedom)
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=root,
                child=base,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=sideways_offset, reference_frame=root
                ),
            )
        )
        world.add_degree_of_freedom(degree_of_freedom)
        hinge = RevoluteConnection(
            name=PrefixedName("hinge"),
            parent=base,
            child=arm,
            axis=Vector3.Z(reference_frame=arm),
            raw_dof=degree_of_freedom,
        )
        world.add_connection(hinge)
        mirrored_hinge = RevoluteConnection(
            name=PrefixedName("mirrored_hinge"),
            parent=base,
            child=mirrored_arm,
            axis=Vector3.Z(reference_frame=mirrored_arm),
            raw_dof=degree_of_freedom,
            multiplier=-1.0,
        )
        world.add_connection(mirrored_hinge)
        world.add_actuator(servo)
    return Pendulum(arm=arm, hinge=hinge, mirrored_hinge=mirrored_hinge, servo=servo)


@pytest.fixture
def twin_pendulum_world() -> TwinPendulumWorld:
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_body(root)
    pendulums = [
        add_pendulum(world, root, sideways_offset) for sideways_offset in (-0.5, 0.5)
    ]
    return TwinPendulumWorld(world=world, pendulums=pendulums)


@pytest.fixture
def built_twin_pendulums(twin_pendulum_world, tmp_path) -> MujocoBuilder:
    builder = MujocoBuilder()
    builder.build_world(
        world=twin_pendulum_world.world, file_path=str(tmp_path / "scene.xml")
    )
    return builder


# %% assigning names


def test_an_element_is_named_after_the_short_name_of_its_entity():
    handle = Body(name=PrefixedName("handle", "mug"))

    assert SimulatorElementNames().assign(handle) == handle.name.name


def test_the_second_of_two_equally_named_entities_is_told_apart_by_its_id():
    first = Body(name=PrefixedName("handle", "mug_a"))
    second = Body(name=PrefixedName("handle", "mug_b"))
    element_names = SimulatorElementNames()

    assert element_names.assign(first) == "handle"
    assert element_names.assign(second) == f"handle_{second.id.hex}"


# %% building


def test_equally_named_entities_are_each_built_under_their_own_name(
    twin_pendulum_world, built_twin_pendulums
):
    element_names = built_twin_pendulums.element_names
    pendulums = twin_pendulum_world.pendulums

    assert sorted(body.name for body in built_twin_pendulums.spec.bodies) == sorted(
        element_names.of(body) for body in twin_pendulum_world.world.bodies
    )
    assert sorted(joint.name for joint in built_twin_pendulums.spec.joints) == sorted(
        element_names.of(connection)
        for pendulum in pendulums
        for connection in (pendulum.hinge, pendulum.mirrored_hinge)
    )
    assert sorted(
        actuator.name for actuator in built_twin_pendulums.spec.actuators
    ) == sorted(element_names.of(pendulum.servo) for pendulum in pendulums)


def test_one_of_equally_named_entities_keeps_the_short_name_a_unique_one_would_get(
    twin_pendulum_world, built_twin_pendulums
):
    """
    Only entities that actually share a name are told apart, so one of them keeps the
    name any lookup by name expects.
    """
    pendulums = twin_pendulum_world.pendulums
    element_names = built_twin_pendulums.element_names
    [arm_name] = {pendulum.arm.name.name for pendulum in pendulums}
    [hinge_name] = {pendulum.hinge.name.name for pendulum in pendulums}
    [servo_name] = {pendulum.servo.name.name for pendulum in pendulums}

    assert arm_name in {element_names.of(pendulum.arm) for pendulum in pendulums}
    assert hinge_name in {element_names.of(pendulum.hinge) for pendulum in pendulums}
    assert servo_name in {element_names.of(pendulum.servo) for pendulum in pendulums}


def test_a_shape_is_built_into_the_body_of_its_own_entity(
    twin_pendulum_world, built_twin_pendulums
):
    element_names = built_twin_pendulums.element_names
    spec = built_twin_pendulums.spec

    for pendulum in twin_pendulum_world.pendulums:
        assert len(spec.body(element_names.of(pendulum.arm)).geoms) == len(
            pendulum.arm.collision.shapes
        )


def test_a_joint_is_built_into_the_body_of_its_own_child(
    twin_pendulum_world, built_twin_pendulums
):
    element_names = built_twin_pendulums.element_names
    spec = built_twin_pendulums.spec

    for pendulum in twin_pendulum_world.pendulums:
        [joint] = spec.body(element_names.of(pendulum.arm)).joints
        assert joint.name == element_names.of(pendulum.hinge)


def test_a_servo_drives_the_joint_of_its_own_degree_of_freedom(
    twin_pendulum_world, built_twin_pendulums
):
    element_names = built_twin_pendulums.element_names
    spec = built_twin_pendulums.spec

    for pendulum in twin_pendulum_world.pendulums:
        actuator = spec.actuator(element_names.of(pendulum.servo))
        assert actuator.target == element_names.of(pendulum.hinge)
        assert actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT


def test_a_mimic_joint_follows_the_joint_of_its_own_degree_of_freedom(
    twin_pendulum_world, built_twin_pendulums
):
    element_names = built_twin_pendulums.element_names

    assert sorted(
        (equality.name1, equality.name2)
        for equality in built_twin_pendulums.spec.equalities
    ) == sorted(
        (
            element_names.of(pendulum.mirrored_hinge),
            element_names.of(pendulum.hinge),
        )
        for pendulum in twin_pendulum_world.pendulums
    )


def test_a_camera_is_built_into_the_body_it_is_attached_to(
    twin_pendulum_world, tmp_path
):
    [_, second] = twin_pendulum_world.pendulums
    camera = MujocoCamera(name="arm_camera", body=second.arm)
    second.arm.add_simulator_property(camera)
    builder = MujocoBuilder()

    builder.build_world(
        world=twin_pendulum_world.world, file_path=str(tmp_path / "scene.xml")
    )

    [built_camera] = builder.spec.body(builder.element_names.of(second.arm)).cameras
    assert built_camera.name == camera.name


def test_a_second_build_names_its_model_afresh(twin_pendulum_world, tmp_path):
    builder = MujocoBuilder()
    builder.build_world(
        world=twin_pendulum_world.world, file_path=str(tmp_path / "first.xml")
    )
    first_build_names = sorted(body.name for body in builder.spec.bodies)

    builder.build_world(
        world=twin_pendulum_world.world, file_path=str(tmp_path / "second.xml")
    )

    assert sorted(body.name for body in builder.spec.bodies) == first_build_names


# %% simulating


@dataclass
class TwinFallingBoxWorld:
    """
    Two equally named boxes falling freely from different heights.
    """

    world: World
    """
    The world holding both boxes.
    """

    boxes: list[Body]
    """
    The two boxes, in the order they were added.
    """


@pytest.fixture
def twin_falling_box_world() -> TwinFallingBoxWorld:
    world = World()
    root = Body(name=PrefixedName("root"))
    boxes = [box_body(PrefixedName("box")) for _ in range(2)]
    with world.modify_world():
        world.add_body(root)
        for box, (sideways_offset, height) in zip(boxes, ((-0.5, 1.0), (0.5, 2.0))):
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=world,
                    parent=root,
                    child=box,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=sideways_offset, z=height, reference_frame=root
                    ),
                )
            )
    return TwinFallingBoxWorld(world=world, boxes=boxes)


def height_of(world: World, body: Body) -> float:
    """
    :param world: The world holding the body.
    :param body: The body to measure.
    :return: The height of the body above the root of the world.
    """
    return float(world.compute_forward_kinematics_np(world.root, body)[2, 3])


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_equally_named_boxes_each_fall_from_their_own_height(twin_falling_box_world):
    world = twin_falling_box_world.world
    boxes = twin_falling_box_world.boxes
    start_heights = [height_of(world, box) for box in boxes]
    duration = timedelta(milliseconds=300)

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        simulation.step_simulation(duration)
    finally:
        simulation.stop_simulation()

    fallen = 0.5 * 9.81 * duration.total_seconds() ** 2
    assert [height_of(world, box) for box in boxes] == pytest.approx(
        [start_height - fallen for start_height in start_heights], abs=0.01
    )


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_each_equally_named_servo_is_commanded_with_its_own_set_point(
    twin_pendulum_world,
):
    set_points = [0.5, -0.5]
    world = twin_pendulum_world.world
    pendulums = twin_pendulum_world.pendulums

    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        for pendulum, set_point in zip(pendulums, set_points):
            world.state[pendulum.hinge.raw_dof.id].position = set_point
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=2))
        reached = [
            simulation.simulator.get_joint_value(
                simulation.synchronizer.element_names.of(pendulum.hinge)
            ).result
            for pendulum in pendulums
        ]
    finally:
        simulation.stop_simulation()

    assert reached == pytest.approx(set_points, abs=0.02)
