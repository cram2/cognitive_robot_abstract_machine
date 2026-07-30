from __future__ import annotations

import json
import math
import numpy as np
import pytest
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.ros_executor import Ros2Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    ObservationStateValues,
    DefaultWeights,
    LifeCycleValues,
)
from giskardpy.motion_statechart.exceptions import (
    MissingExitSpeedError,
    MissingInflowEquationError,
    NonPositiveClearanceError,
    RootLinkNotWorldRootError,
)
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
    PouringTask,
)

from .debug_expression_helpers import debug_expression_by_name
from .single_cup_world import PourableContainer, build_single_cup_world
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.physics.equations.pouring_equations import InflowEquation
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    LiquidConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Mesh,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from dataclasses import dataclass

_JEROEN_CUP_STL = str(
    Path(files("semantic_digital_twin")).parent.parent
    / "resources"
    / "stl"
    / "jeroen_cup.stl"
)
_JEROEN_CUP_SCALE = Scale(1, 1, 1)
_TABLE_SURFACE_Z = 0.9
_POURING_TARGET_FREQUENCY = 80
_POURING_PREDICTION_HORIZON = 120
_DEFAULT_PERCEPTION_HZ: int = 10


def _pouring_context(world: World) -> MotionStatechartContext:
    """
    Builds a context whose QP runs at a high frequency over a long prediction horizon.

    The long horizon lets the linearized fill prediction span the pouring overshoot, so
    the constraint converges without a reactive damping term.
    """
    return MotionStatechartContext(
        world=world,
        qp_controller_config=QPControllerConfig(
            target_frequency=_POURING_TARGET_FREQUENCY,
            prediction_horizon=_POURING_PREDICTION_HORIZON,
        ),
    )


@pytest.fixture
def pr2_world_setup(pr2_world_copy):
    """
    Function-scoped PR2 world, suitable for tests that modify the world.
    """
    return pr2_world_copy


@pytest.fixture
def world_with_cup():
    """
    World containing a single pourable container with a tilt joint, filled to 100%.
    """
    return build_single_cup_world()


def _spawn_jeroen_cup_body(name: str) -> Body:
    """
    Create a Body with the Jeroen cup mesh geometry.
    """
    mesh = Mesh(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
        filename=_JEROEN_CUP_STL,
        scale=_JEROEN_CUP_SCALE,
    )
    return Body.from_shape_collection(
        shape_collection=ShapeCollection([mesh]), name=PrefixedName(name)
    )


@pytest.fixture(scope="function")
def tracy_pouring_world(tracy_world):
    """
    Tracy world with both arms in park position and a Jeroen cup on the table.
    """
    world = deepcopy(tracy_world)
    [tracy] = world.get_semantic_annotations_by_type(Tracy)

    left_park = tracy.left_arm.get_joint_state_by_type(StaticJointState.PARK)
    right_park = tracy.right_arm.get_joint_state_by_type(StaticJointState.PARK)
    world.set_positions_1DOF_connection(dict(left_park.items()))
    world.set_positions_1DOF_connection(dict(right_park.items()))

    table_cup_body = _spawn_jeroen_cup_body("table_cup")
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world,
                world.root,
                table_cup_body,
                name=PrefixedName("table_T_table_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.5, 0.0, _TABLE_SURFACE_Z
                ),
            )
        )

    return world, tracy


@pytest.fixture(scope="function")
def tracy_transfer_world(tracy_pouring_world):
    """
    World with a source cup attached to Tracy's left gripper and a receiving cup on the
    table, pre-coupled for liquid transfer.

    Extends :func:`tracy_pouring_world` by positioning the left gripper upright, attaching a
    source cup via a fixed connection, placing a receiving cup on the table, and coupling them
    with :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasFillLevel.receive_outflow_from`.

    :returns: ``(world, source_cup, receiving_cup, left_tool_frame)``
    """
    world, _tracy = tracy_pouring_world
    left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")

    upright_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=1,
        pos_y=0.2,
        pos_z=_TABLE_SURFACE_Z + 0.3,
        quat_z=0.5,
        quat_x=0.5,
        quat_y=0.5,
        quat_w=0.5,
        reference_frame=world.root,
    ).to_pose()

    cartesian_statechart = MotionStatechart()
    cartesian_task = CartesianPose(
        root_link=world.root, tip_link=left_tool_frame, goal_pose=upright_pose
    )
    cartesian_statechart.add_node(cartesian_task)
    cartesian_statechart.add_node(EndMotion.when_true(cartesian_task))
    cartesian_executor = Executor(
        MotionStatechartContext(world=world),
        pacer=SimulationPacer(real_time_factor=1),
    )
    cartesian_executor.compile(motion_statechart=cartesian_statechart)
    cartesian_executor.tick_until_end(timeout=1000)

    source_cup_body = _spawn_jeroen_cup_body("source_cup")
    with world.modify_world():
        world.add_body(source_cup_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=left_tool_frame,
                child=source_cup_body,
                name=PrefixedName("l_gripper_T_source_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    roll=-math.pi / 2.0, y=-0.0
                ),
            )
        )
    source_cup = PourableContainer(
        name=PrefixedName("source_cup"), root=source_cup_body
    )
    with world.modify_world():
        world.add_semantic_annotation(source_cup)
    source_cup.initialize_fill_level(
        world=world, initial_fill=1.0, outflow_rate_constant=1.0
    )

    receiving_cup_body = _spawn_jeroen_cup_body("receiving_cup")
    with world.modify_world():
        world.add_body(receiving_cup_body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world,
                world.root,
                receiving_cup_body,
                name=PrefixedName("table_T_receiving_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.0, 0.1, _TABLE_SURFACE_Z
                ),
            )
        )
    receiving_cup = PourableContainer(
        name=PrefixedName("receiving_cup"), root=receiving_cup_body
    )
    with world.modify_world():
        world.add_semantic_annotation(receiving_cup)
    receiving_cup.initialize_fill_level(
        world=world, initial_fill=0.0, outflow_rate_constant=1.0
    )

    receiving_cup.receive_outflow_from(source=source_cup, world=world)

    left_wrist_joint = world.get_connection_by_name("left_wrist_3_joint")
    world.set_positions_1DOF_connection(
        {left_wrist_joint: left_wrist_joint.position + 0.1}
    )

    return world, source_cup, receiving_cup, left_tool_frame


@dataclass
class TransferMotion:
    """
    A built liquid-transfer motion together with the entities its tasks act on.

    Bundles what a transfer test needs to run the motion and observe the cups, so the
    tests that exercise different weightings of the same motion share one construction
    path.
    """

    world: World
    """
    The world the motion is built against.
    """

    source_cup: PourableContainer
    """
    The cup being poured from.
    """

    receiving_cup: PourableContainer
    """
    The cup being poured into.
    """

    transfer_task: PouringTask
    """
    The fill-driving task, whose tick hook the clearance recorder wraps.
    """

    motion_statechart: MotionStatechart
    """
    The statechart executing the transfer.
    """

    def record_rim_clearance(self) -> list[float]:
        """
        Start recording the source-lip-above-receiver-rim clearance on every transfer
        tick.

        :return: The list the clearance samples are appended to as the motion runs.
        """
        source_lip = self.source_cup.liquid_exit_point(self.world)
        receiver_rim = (
            self.world.compose_forward_kinematics_expression(
                self.world.root, self.receiving_cup.root
            )
            @ self.receiving_cup.rim_point()
        )
        clearance_history: list[float] = []
        original_on_tick = self.transfer_task.on_tick

        def recording_on_tick(context):
            clearance_history.append(
                float(source_lip.z.evaluate()[0] - receiver_rim.z.evaluate()[0])
            )
            return original_on_tick(context)

        self.transfer_task.on_tick = recording_on_tick
        return clearance_history

    def execute(self) -> None:
        """
        Run the transfer to completion in simulation.
        """
        executor = Executor(
            _pouring_context(self.world), pacer=SimulationPacer(real_time_factor=1)
        )
        executor.compile(motion_statechart=self.motion_statechart)
        executor.tick_until_end(timeout=4000)


def _build_transfer_motion(
    tracy_transfer_world,
    minimum_clearance: float = 0.05,
    no_spill_weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
    no_spill_reference_velocity: float = 0.2,
    fill_level_tolerance: float = 0.05,
) -> TransferMotion:
    """
    Build the cup-to-cup transfer motion the transfer tests share.

    The fill-driving task starts only once the pour-aiming task observes the projectile
    landing in the receiver.  While the pour is mis-aimed the transfer gate blocks all
    flow, so driving the predicted terminal fill cannot make progress anyway; keeping
    the stiff fill constraint out of the QP during the approach also keeps the solver
    away from its iteration budget while the gate's logistic transition makes the fill
    gradient large and rapidly changing.

    :param tracy_transfer_world: The transfer world fixture value.
    :param minimum_clearance: Rim-to-rim clearance floor handed to the clearance task.
    :param no_spill_weight: Weight of the competing pour-aiming task.
    :param no_spill_reference_velocity: Reference velocity of the competing pour-aiming
        task.
    :param fill_level_tolerance: Tolerance around the fill goal handed to the fill task.
    :return: The built motion.
    """
    world, source_cup, receiving_cup, left_tool_frame = tracy_transfer_world

    transfer_task = FillByTransferTask(
        receiver=receiving_cup,
        goal_value=0.7,
        fill_level_tolerance=fill_level_tolerance,
        reference_velocity=0.03,
    )
    no_spill = KeepProjectileInReceiver(
        receiver=receiving_cup,
        source=source_cup,
        weight=no_spill_weight,
        reference_velocity=no_spill_reference_velocity,
    )
    keep_above = KeepSourceRimAboveReceiverRim(
        receiver=receiving_cup, source=source_cup, minimum_clearance=minimum_clearance
    )
    keep_plane = AlignPlanes(
        root_link=world.root,
        tip_link=left_tool_frame,
        goal_normal=Vector3.X(reference_frame=world.root),
        tip_normal=Vector3.Z(reference_frame=left_tool_frame),
    )
    motion = Parallel([transfer_task, no_spill, keep_above, keep_plane])
    transfer_task.start_condition = no_spill.observation_variable
    transfer_statechart = MotionStatechart()
    transfer_statechart.add_node(motion)
    transfer_statechart.add_node(EndMotion.when_true(motion))
    return TransferMotion(
        world=world,
        source_cup=source_cup,
        receiving_cup=receiving_cup,
        transfer_task=transfer_task,
        motion_statechart=transfer_statechart,
    )


def _tick_with_perception_correction(
    executor: Executor,
    world: World,
    fill_connection: LiquidConnection,
    sigma: float,
    perception_hz: float,
    rng: np.random.Generator,
    timeout: int = 4000,
) -> None:
    """
    Run the executor tick loop, injecting a noisy fill-level measurement at
    ``perception_hz``.

    After each tick the ODE has already advanced the fill level via
    :meth:`~semantic_digital_twin.world.World.step_physics`.  When the current tick index
    aligns with the perception period the ODE-integrated value is replaced by the true fill
    plus additive Gaussian noise, making it the QP's linearization point on the next tick.
    This models a perception pipeline that corrects the controller's fill-level belief at a
    rate lower than the control frequency.

    Cleanup (zero velocities/accelerations/jerks, node and context teardown) mirrors
    :meth:`~giskardpy.executor.Executor.tick_until_end`.

    :param executor: The executor driving the motion statechart.
    :param world: The world whose fill state is corrected by perception.
    :param fill_connection: The receiver's fill DOF to update on each perception tick.
    :param sigma: Standard deviation of additive Gaussian noise on the fill measurement.
    :param perception_hz: Frequency at which perception measurements arrive, in Hz.
    :param rng: Random number generator for reproducible noise sequences.
    :param timeout: Maximum number of control ticks before raising ``TimeoutError``.
    """
    control_hz = executor.context.qp_controller_config.target_frequency
    ticks_per_perception = max(1, round(control_hz / perception_hz))
    try:
        for tick_index in range(timeout):
            executor.tick()
            if tick_index % ticks_per_perception == 0:
                true_fill = float(fill_connection.position)
                noisy_fill = float(
                    np.clip(true_fill + rng.normal(0.0, sigma), 0.0, 1.0)
                )
                world.set_positions_1DOF_connection({fill_connection: noisy_fill})
            executor.pacer.sleep()
            if executor.motion_statechart.is_end_motion():
                return
        raise TimeoutError("Timeout reached while waiting for end of motion.")
    finally:
        state = executor.context.world.state
        state.velocities[:] = 0
        state.accelerations[:] = 0
        state.jerks[:] = 0
        executor.motion_statechart.cleanup_nodes(context=executor.context)
        executor.context.cleanup()


class TestPouringTask:
    """
    Test suite for the PouringTask in Giskardpy.
    """

    def test_pouring_task_achieves_goal(self, world_with_cup) -> None:
        """
        Test that PouringTask successfully tilts the cup and reduces fill level to the
        target value.
        """
        world, cup = world_with_cup
        goal_fill = 0.6
        tolerance = 0.05

        motion_statechart = MotionStatechart()
        pouring_task = PouringTask(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            root_link=world.root,
            tip_link=cup.root,
            goal_value=goal_fill,
            fill_level_tolerance=tolerance,
            reference_velocity=0.05,
        )
        motion_statechart.add_node(pouring_task)
        motion_statechart.add_node(EndMotion.when_true(pouring_task))

        executor = Executor(
            _pouring_context(world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        executor.compile(motion_statechart=motion_statechart)

        executor.tick_until_end(timeout=4000)

        assert pouring_task.observation_state == ObservationStateValues.TRUE
        assert cup.fill_level <= goal_fill + tolerance
        assert cup.fill_level >= goal_fill - tolerance
        assert cup.root.parent_connection.position > 0.1
        assert cup.fill_equation.symbolic_velocity(cup.fill_connection).evaluate()[
            0
        ] == pytest.approx(0.0, abs=1e-2)

    def test_build_rejects_non_world_root_link(self, world_with_cup) -> None:
        """
        The cup tilt is derived from the world root, so building with any other root
        link must fail loudly rather than silently mispredict the pour against a non-
        vertical reference.
        """
        world, cup = world_with_cup
        pouring_task = PouringTask(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            root_link=cup.root,
            tip_link=cup.root,
            goal_value=0.6,
            fill_level_tolerance=0.05,
        )
        with pytest.raises(RootLinkNotWorldRootError) as error_info:
            pouring_task.build(_pouring_context(world))

        assert error_info.value.root_link is cup.root
        assert error_info.value.world_root is world.root

    def test_on_tick_does_not_report_true_while_still_flowing(
        self, world_with_cup
    ) -> None:
        """
        A fill level inside the goal band must not count as done while liquid is still
        flowing, otherwise the motion would end mid-pour and the fill would keep
        dropping.
        """
        world, cup = world_with_cup
        goal_fill = 0.6
        pouring_task = PouringTask(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            root_link=world.root,
            tip_link=cup.root,
            goal_value=goal_fill,
            fill_level_tolerance=0.05,
        )
        context = MotionStatechartContext(world=world)
        pouring_task.build(context)

        flowing_tilt = 1.3
        world.set_positions_1DOF_connection(
            {cup.root.parent_connection: flowing_tilt, cup.fill_connection: goal_fill}
        )
        fill_rate = cup.fill_equation.symbolic_velocity(cup.fill_connection).evaluate()[
            0
        ]
        assert abs(fill_rate) > pouring_task.outflow_tolerance

        assert pouring_task.on_tick(context) is None

    def test_proactive_tilt_back(self, world_with_cup) -> None:
        """
        Verify that the linearized MPC starts reducing tilt before the fill level
        reaches the goal, demonstrating proactive rather than purely reactive control.

        The cup tilt must begin decreasing while the fill level is still strictly above
        ``goal_value + fill_level_tolerance``.
        """
        world, cup = world_with_cup
        goal_fill = 0.6
        tolerance = 0.05

        motion_statechart = MotionStatechart()
        pouring_task = PouringTask(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            root_link=world.root,
            tip_link=cup.root,
            goal_value=goal_fill,
            fill_level_tolerance=tolerance,
            reference_velocity=0.05,
        )
        motion_statechart.add_node(pouring_task)
        motion_statechart.add_node(EndMotion.when_true(pouring_task))

        tilt_history: list[float] = []
        fill_history: list[float] = []

        original_on_tick = pouring_task.on_tick

        def recording_on_tick(context):
            tilt_history.append(float(cup.root.parent_connection.position))
            fill_history.append(float(cup.fill_level))
            return original_on_tick(context)

        pouring_task.on_tick = recording_on_tick

        executor = Executor(
            _pouring_context(world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        executor.compile(motion_statechart=motion_statechart)
        executor.tick_until_end(timeout=4000)

        assert pouring_task.observation_state == ObservationStateValues.TRUE

        tilt_near_goal_start: float | None = None
        max_tilt_before_threshold = 0.0
        threshold = goal_fill + 2 * tolerance
        for tilt, fill in zip(tilt_history, fill_history):
            if fill > threshold:
                max_tilt_before_threshold = max(max_tilt_before_threshold, tilt)
            elif tilt_near_goal_start is None:
                tilt_near_goal_start = tilt
                break

        assert tilt_near_goal_start is not None, "fill level never approached goal"
        assert tilt_near_goal_start < max_tilt_before_threshold, (
            f"Expected tilt to be decreasing when fill first reached goal region "
            f"(tilt={tilt_near_goal_start:.4f} should be < "
            f"max tilt before threshold={max_tilt_before_threshold:.4f})"
        )

    def test_pr2_pouring_from_gripper(self, pr2_world_setup) -> None:
        """
        Test that PouringTask works when the cup is held by the PR2 robot.
        """
        world = pr2_world_setup
        # Create a cup setup
        gripper_frame = world.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )

        with world.modify_world():
            cup_body = Body(name=PrefixedName("cup"))
            world.add_body(cup_body)
            gripper_C_tilt = FixedConnection.create_with_dofs(
                world=world,
                parent=gripper_frame,
                child=cup_body,
                name=PrefixedName("gripper_T_cup_tilt"),
            )
            world.add_connection(gripper_C_tilt)

            _cup_height = 0.12
            _cup_half_width = 0.04
            cup_shape = Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=_cup_height / 2,
                    reference_frame=cup_body,
                ),
                scale=Scale(
                    2 * _cup_half_width,
                    2 * _cup_half_width,
                    _cup_height,
                ),
            )
            cup_body.visual = ShapeCollection(shapes=[cup_shape])
            cup_body.collision = ShapeCollection(shapes=[cup_shape])
            cup_body.collision.reference_frame = cup_body

        cup = PourableContainer(name=PrefixedName("cup"), root=cup_body)
        with world.modify_world():
            world.add_semantic_annotation(cup)

        cup.initialize_fill_level(
            world=world,
            initial_fill=1.0,
            outflow_rate_constant=1.0,
        )

        goal_fill = 0.6
        tolerance = 0.05
        motion_statechart = MotionStatechart()
        pouring_task = PouringTask(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            root_link=world.root,
            tip_link=cup_body,
            goal_value=goal_fill,
            fill_level_tolerance=tolerance,
        )
        motion_statechart.add_node(pouring_task)
        motion_statechart.add_node(EndMotion.when_true(pouring_task))

        executor = Executor(
            _pouring_context(world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        executor.compile(motion_statechart=motion_statechart)

        executor.tick_until_end(timeout=4000)

        assert pouring_task.observation_state == ObservationStateValues.TRUE
        assert cup.fill_level == pytest.approx(goal_fill, abs=tolerance)
        assert cup.fill_equation.symbolic_velocity(cup.fill_connection).evaluate()[
            0
        ] == pytest.approx(0.0, abs=1e-2)


class TestTracyPouring:
    """
    Test suite for PouringTask using the Tracy dual-arm robot.
    """

    def add_left_wrist_3_offset(self, world: World, offset: float) -> None:
        """
        Add an angular offset to the current left_wrist_3_joint position.
        """
        joint = world.get_connection_by_name("left_wrist_3_joint")
        world.set_positions_1DOF_connection({joint: joint.position + offset})

    def test_tracy_pouring(self, tracy_pouring_world) -> None:
        """
        Test that PouringTask reduces the fill level of a Jeroen cup held in Tracy's
        left gripper from 1.0 to 0.5.

        A CartesianPose task first moves the left gripper to the upright pose, then the
        cup is grasped and pouring begins.
        """
        world, tracy = tracy_pouring_world
        left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")

        upright_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=1,
            pos_y=0.2,
            pos_z=_TABLE_SURFACE_Z + 0.3,
            quat_z=0.5,
            quat_x=0.5,
            quat_y=0.5,
            quat_w=0.5,
            reference_frame=world.root,
        ).to_pose()

        cartesian_statechart = MotionStatechart()
        cartesian_task = CartesianPose(
            root_link=world.root, tip_link=left_tool_frame, goal_pose=upright_pose
        )
        cartesian_statechart.add_node(cartesian_task)
        cartesian_statechart.add_node(EndMotion.when_true(cartesian_task))

        cartesian_executor = Executor(
            MotionStatechartContext(world=world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        cartesian_executor.compile(motion_statechart=cartesian_statechart)
        cartesian_executor.tick_until_end(timeout=1000)

        grasped_cup_body = _spawn_jeroen_cup_body("grasped_cup")
        with world.modify_world():
            world.add_body(grasped_cup_body)
            world.add_connection(
                FixedConnection.create_with_dofs(
                    world=world,
                    parent=left_tool_frame,
                    child=grasped_cup_body,
                    name=PrefixedName("l_gripper_T_grasped_cup"),
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        roll=-math.pi / 2.0, y=-0.0
                    ),
                )
            )

        grasped_cup = PourableContainer(
            name=PrefixedName("grasped_cup"), root=grasped_cup_body
        )
        with world.modify_world():
            world.add_semantic_annotation(grasped_cup)
        grasped_cup.initialize_fill_level(
            world=world, initial_fill=1.0, outflow_rate_constant=1.0
        )

        self.add_left_wrist_3_offset(world, 0.1)

        assert grasped_cup.fill_level == pytest.approx(1.0)

        goal_fill = 0.8
        tolerance = 0.05

        pouring_statechart = MotionStatechart()
        pouring_task = PouringTask(
            fill_equation=grasped_cup.fill_equation,
            fill_connection=grasped_cup.fill_connection,
            root_link=world.root,
            tip_link=grasped_cup_body,
            goal_value=goal_fill,
            fill_level_tolerance=tolerance,
        )
        keep_position = CartesianPosition(
            root_link=world.root,
            tip_link=left_tool_frame,
            goal_point=Point3(reference_frame=left_tool_frame),
            weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
        )
        motion = Parallel([pouring_task, keep_position])
        pouring_statechart.add_node(motion)
        pouring_statechart.add_node(EndMotion.when_true(motion))

        pouring_executor = Executor(
            _pouring_context(world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        pouring_executor.compile(motion_statechart=pouring_statechart)
        pouring_executor.tick_until_end(timeout=4000)

        assert pouring_task.observation_state == ObservationStateValues.TRUE
        assert grasped_cup.fill_level == pytest.approx(goal_fill, abs=tolerance)
        assert grasped_cup.fill_equation.symbolic_velocity(
            grasped_cup.fill_connection
        ).evaluate()[0] == pytest.approx(0.0, abs=1e-2)


class TestTracyLiquidTransfer:
    """
    Test suite for cup-to-cup liquid transfer driven by a fill-level goal on the
    receiver.

    The commanded goal is the fill level of a *receiving* cup standing on the table; the
    only controllable degrees of freedom belong to the arm holding the *source* cup. The
    optimizer must therefore tilt the source cup so that liquid leaving it lands in the
    receiver and raises the receiver's fill level to the goal.
    """

    def test_tracy_liquid_transfer_fills_receiver(
        self, tracy_transfer_world, rclpy_node
    ) -> None:
        """
        Commanding a fill-level goal on the receiving cup makes the optimizer tilt the
        grasped source cup until the receiver reaches the goal.

        The transfer is volume conserving: while the source rim is above the receiver the
        volume the source loses equals the volume the receiver gains, so no liquid is spilled.
        """
        world, source_cup, receiving_cup, left_tool_frame = tracy_transfer_world

        assert receiving_cup.fill_level == pytest.approx(0.0)
        assert source_cup.fill_level == pytest.approx(1.0)

        goal_fill = 0.7
        tolerance = 0.05
        source_fill_before = source_cup.fill_level

        transfer_task = FillByTransferTask(
            receiver=receiving_cup,
            goal_value=goal_fill,
            fill_level_tolerance=tolerance,
            reference_velocity=0.03,
        )
        # The no-spill task keeps the liquid's projectile landing in the receiver, so the optimizer
        # repositions the gripper upstream as the source tilts and the arc reaches forward.
        no_spill = KeepProjectileInReceiver(receiver=receiving_cup, source=source_cup)
        # Keep the source cup in a tight height band above the receiver so the optimizer aims the
        # pour from a stable elevation instead of thrashing vertically while repositioning.  The
        # band is rim-to-rim, matching what the transfer gate measures: a band on the cup origins
        # lets the lip sink towards the receiver's rim as the source tilts, closing the gate the
        # fill task needs open.
        minimum_clearance = 0.2
        keep_above = KeepSourceRimAboveReceiverRim(
            receiver=receiving_cup,
            source=source_cup,
            minimum_clearance=minimum_clearance,
            clearance_band=0.02,
            weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
        )
        keep_plane = AlignPlanes(
            root_link=world.root,
            tip_link=left_tool_frame,
            goal_normal=Vector3.X(reference_frame=world.root),
            tip_normal=Vector3.Z(reference_frame=left_tool_frame),
        )
        motion = Parallel([transfer_task, no_spill, keep_above, keep_plane])
        transfer_statechart = MotionStatechart()
        transfer_statechart.add_node(motion)
        transfer_statechart.add_node(EndMotion.when_true(motion))

        gate_history: list[float] = []
        tilt_history: list[float] = []
        clearance_history: list[float] = []
        original_on_tick = transfer_task.on_tick

        def recording_on_tick(context):
            inflow_equation = transfer_task.fill_connection.inflow_equation
            gate_history.append(float(inflow_equation.gate.evaluate()[0]))
            tilt_history.append(
                float(inflow_equation.source_tilt_expression.evaluate()[0])
            )
            source_z = world.compute_forward_kinematics_np(world.root, source_cup.root)[
                2, 3
            ]
            receiver_z = world.compute_forward_kinematics_np(
                world.root, receiving_cup.root
            )[2, 3]
            clearance_history.append(float(source_z - receiver_z))
            return original_on_tick(context)

        transfer_task.on_tick = recording_on_tick

        transfer_executor = Ros2Executor(
            _pouring_context(world),
            pacer=SimulationPacer(real_time_factor=1),
            ros_node=rclpy_node,
            publish_debug_expressions=True,
        )
        transfer_executor.compile(motion_statechart=transfer_statechart)
        transfer_executor.tick_until_end(timeout=4000)

        assert transfer_task.observation_state == ObservationStateValues.TRUE
        assert receiving_cup.fill_level == pytest.approx(goal_fill, abs=tolerance)

        receiver_gain = receiving_cup.fill_level
        source_loss = source_fill_before - source_cup.fill_level
        assert source_loss > tolerance, "source cup never poured"
        assert receiver_gain == pytest.approx(source_loss, abs=tolerance), (
            "transfer must be volume conserving (equal cups): "
            f"receiver gained {receiver_gain:.3f}, source lost {source_loss:.3f}"
        )
        assert max(tilt_history) > 0.5, "the source cup never tilted to pour"
        # The source starts mis-aimed at the offset receiver, so the optimizer first swings the arc
        # onto the opening (gate closed, but the gated source does not spill meanwhile). Once the
        # pour starts the projectile must stay in the receiver — the gate must not close again.
        first_open_tick = next(
            (tick for tick, gate in enumerate(gate_history) if gate > 0.5), None
        )
        assert (
            first_open_tick is not None
        ), "the optimizer never aimed the pour into the receiver"
        assert min(clearance_history) > 0.0, (
            "the source cup dropped to or below the receiver during the pour: "
            f"minimum clearance was {min(clearance_history):.3f}"
        )


class TestRimClearanceDuringTransfer:
    """
    :class:`~giskardpy.motion_statechart.tasks.pouring.KeepSourceRimAboveReceiverRim`
    keeps the pouring cup's lip above the receiving cup's rim throughout the transfer,
    so the rims never collide however far the source tilts, and the motion statechart
    stays serializable.
    """

    def test_transfer_motion_is_json_serializable(self, tracy_transfer_world) -> None:
        """
        The transfer motion statechart serializes to JSON, as the standalone demo
        requires when it ships the goal to Giskard; a task carrying a live symbolic
        point would break this.
        """
        transfer = _build_transfer_motion(tracy_transfer_world)

        tracker = WorldEntityWithIDKwargsTracker.from_world(transfer.world)
        restored = MotionStatechart.from_json(
            json.loads(json.dumps(transfer.motion_statechart.to_json())),
            world=transfer.world,
            **tracker.create_kwargs(),
        )

        assert [node.name for node in restored.nodes] == [
            node.name for node in transfer.motion_statechart.nodes
        ]

    def test_pouring_lip_stays_above_receiver_rim(self, tracy_transfer_world):
        """
        The rim clearance stays positive for the whole pour, without a hand-tuned origin
        offset.
        """
        transfer = _build_transfer_motion(tracy_transfer_world)
        clearance_history = transfer.record_rim_clearance()

        transfer.execute()

        transfer_task = transfer.transfer_task
        assert transfer_task.observation_state == ObservationStateValues.TRUE
        assert clearance_history, "transfer never ticked"
        assert min(clearance_history) > 0.0, (
            "the pouring lip dropped to or below the receiver rim: "
            f"minimum clearance was {min(clearance_history):.3f} m"
        )


class TestFillDriveStartsOnceAimed:
    """
    The transfer motion stages its tasks: the fill-driving task stays inactive until the
    pour-aiming task observes the projectile landing in the receiver.

    While the pour is mis-aimed the transfer gate blocks all flow, so an active fill
    constraint could not make progress; it would only load the QP with the gate
    transition's stiff, rapidly changing fill gradient.
    """

    def test_fill_task_waits_for_the_aim_observation(
        self, tracy_transfer_world
    ) -> None:
        """
        The fill task is not started while the pour is mis-aimed and starts on the tick
        after the aiming task first observes the landing inside the receiver.
        """
        transfer = _build_transfer_motion(tracy_transfer_world)
        statechart = transfer.motion_statechart
        transfer_task = transfer.transfer_task
        [parallel] = [node for node in statechart.nodes if isinstance(node, Parallel)]
        [no_spill] = [
            node
            for node in parallel.nodes
            if isinstance(node, KeepProjectileInReceiver)
        ]

        executor = Executor(
            _pouring_context(transfer.world), pacer=SimulationPacer(real_time_factor=1)
        )
        executor.compile(motion_statechart=statechart)
        try:
            executor.tick()
            assert (
                statechart.observation_state[no_spill] != ObservationStateValues.TRUE
            ), "the pour must start mis-aimed for this test to be meaningful"
            assert (
                statechart.life_cycle_state[transfer_task]
                == LifeCycleValues.NOT_STARTED
            )

            aimed_tick = None
            for tick in range(1000):
                if (
                    statechart.observation_state[no_spill]
                    == ObservationStateValues.TRUE
                ):
                    aimed_tick = tick
                    break
                assert (
                    statechart.life_cycle_state[transfer_task]
                    == LifeCycleValues.NOT_STARTED
                ), f"fill task started at tick {tick} before the pour was aimed"
                executor.tick()
            assert aimed_tick is not None, "the pour never got aimed"

            executor.tick()
            assert statechart.life_cycle_state[transfer_task] == LifeCycleValues.RUNNING
        finally:
            state = executor.context.world.state
            state.velocities[:] = 0
            state.accelerations[:] = 0
            state.jerks[:] = 0
            executor.motion_statechart.cleanup_nodes(context=executor.context)
            executor.context.cleanup()


class TestClearanceBandStaysAboveTheRim:
    """
    The clearance band of
    :class:`~giskardpy.motion_statechart.tasks.pouring.KeepSourceRimAboveReceiverRim` describes
    where the pouring lip is held, and the optimizer tracks it only to within a slack tolerance.
    A band reaching down to the rim therefore lets the lip settle below the rim, so the task
    rejects one instead of accepting a physically impossible pour.
    """

    def test_band_reaching_down_to_the_rim_is_rejected(self, world_with_cup) -> None:
        """
        A zero clearance floor asks for the rims to touch, which the soft bound turns
        into the rims crossing, so building the task raises rather than accepting it.
        """
        world, cup = world_with_cup
        keep_above = KeepSourceRimAboveReceiverRim(
            receiver=cup, source=cup, minimum_clearance=0.0
        )

        with pytest.raises(NonPositiveClearanceError) as error_info:
            keep_above.build(MotionStatechartContext(world=world))

        assert error_info.value.minimum_clearance == 0.0

    def test_band_is_derived_from_the_configured_width(self, world_with_cup) -> None:
        """
        The band's upper end follows :attr:`clearance_band` above the floor, so callers
        configure one width instead of two absolute heights that could be ordered
        wrongly.
        """
        _world, cup = world_with_cup
        keep_above = KeepSourceRimAboveReceiverRim(
            receiver=cup,
            source=cup,
            minimum_clearance=0.1,
            clearance_band=0.04,
        )

        assert keep_above.maximum_clearance == pytest.approx(0.14)

    def test_rims_do_not_cross_when_aiming_outweighs_the_clearance(
        self, tracy_transfer_world
    ):
        """
        With the pour-aiming task at maximum weight, as the standalone demo configures
        it, the pouring lip still never reaches the receiver rim.
        """
        transfer = _build_transfer_motion(
            tracy_transfer_world,
            minimum_clearance=0.05,
            no_spill_weight=DefaultWeights.WEIGHT_MAXIMUM,
            no_spill_reference_velocity=0.1,
        )
        clearance_history = transfer.record_rim_clearance()

        transfer.execute()

        assert clearance_history, "transfer never ticked"
        assert min(clearance_history) > 0.0, (
            "the pouring lip reached the receiver rim: minimum clearance was "
            f"{min(clearance_history):.4f} m"
        )


def _box_cup_body(name: str, height: float = 0.1, width: float = 0.06) -> Body:
    """
    Create a box-shaped cup body whose bounding box spans ``[0, height]`` in z.
    """
    body = Body(name=PrefixedName(name))
    cup_shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=height / 2, reference_frame=body
        ),
        scale=Scale(width, width, height),
    )
    body.visual = ShapeCollection(shapes=[cup_shape])
    body.collision = ShapeCollection(shapes=[cup_shape])
    body.collision.reference_frame = body
    return body


def _rim_band_world(
    source_height_offset: float,
) -> tuple[World, PourableContainer, PourableContainer]:
    """
    Build a world with two identical upright box cups, the source raised by the given
    offset.

    Because the cups are identical and upright, the source-lip-above-receiver-rim
    clearance equals ``source_height_offset`` exactly.

    :param source_height_offset: Height of the source cup's origin above the receiver's.
    :return:``(world, source, receiver)``.
    """
    world = World()
    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        world.add_body(map_body)
    receiver_body = _box_cup_body("receiver_cup")
    source_body = _box_cup_body("source_cup")
    with world.modify_world():
        world.add_body(receiver_body)
        world.add_body(source_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=map_body,
                child=receiver_body,
                name=PrefixedName("map_T_receiver_cup"),
            )
        )
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=map_body,
                child=source_body,
                name=PrefixedName("map_T_source_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=source_height_offset
                ),
            )
        )
    receiver = PourableContainer(name=PrefixedName("receiver_cup"), root=receiver_body)
    source = PourableContainer(name=PrefixedName("source_cup"), root=source_body)
    with world.modify_world():
        world.add_semantic_annotation(receiver)
        world.add_semantic_annotation(source)
    return world, source, receiver


class TestRimClearanceBandObservation:
    """
    The observation of
    :class:`~giskardpy.motion_statechart.tasks.pouring.KeepSourceRimAboveReceiverRim` reports
    whether the source-lip-above-receiver-rim clearance lies within the configured band.
    """

    MINIMUM_CLEARANCE = 0.05
    """
    Clearance floor used by all band observation tests, in metres.
    """

    CLEARANCE_BAND = 0.05
    """
    Band width above the floor used by all band observation tests, in metres.
    """

    def _observation_at_offset(self, source_height_offset: float) -> float:
        """
        Build the task in a world with the given rim-to-rim clearance and evaluate it.
        """
        world, source, receiver = _rim_band_world(source_height_offset)
        keep_above = KeepSourceRimAboveReceiverRim(
            receiver=receiver,
            source=source,
            minimum_clearance=self.MINIMUM_CLEARANCE,
            clearance_band=self.CLEARANCE_BAND,
        )
        artifacts = keep_above.build(MotionStatechartContext(world=world))
        return float(artifacts.observation.evaluate()[0])

    def test_clearance_inside_the_band_observes_true(self) -> None:
        """
        A clearance between floor and ceiling satisfies the band.
        """
        assert self._observation_at_offset(0.07) == 1.0

    def test_clearance_above_the_band_observes_false(self) -> None:
        """
        A clearance above the ceiling violates the band.
        """
        assert self._observation_at_offset(0.2) == 0.0

    def test_clearance_below_the_band_observes_false(self) -> None:
        """
        A clearance below the floor violates the band.
        """
        assert self._observation_at_offset(0.01) == 0.0


class TestFillTaskInitializationErrors:
    """
    The transfer tasks validate their liquid coupling at build time and raise concrete
    initialization errors instead of failing deep inside the QP with opaque symptoms.
    """

    def test_fill_by_transfer_requires_inflow_coupling(self, world_with_cup) -> None:
        """
        A receiver never coupled via receive_outflow_from cannot drive a transfer.
        """
        world, cup = world_with_cup
        transfer_task = FillByTransferTask(
            receiver=cup, goal_value=0.7, fill_level_tolerance=0.05
        )

        with pytest.raises(MissingInflowEquationError):
            transfer_task.build(MotionStatechartContext(world=world))

    def test_keep_projectile_requires_inflow_coupling(self, world_with_cup) -> None:
        """
        The no-spill task needs the receiver's inflow equation to derive the gate.
        """
        world, cup = world_with_cup
        no_spill = KeepProjectileInReceiver(receiver=cup, source=cup)

        with pytest.raises(MissingInflowEquationError):
            no_spill.build(MotionStatechartContext(world=world))

    def test_keep_projectile_requires_an_exit_speed(self, world_with_cup) -> None:
        """
        A source exposing no live outflow model combined with an ungated inflow equation
        leaves no exit speed to derive the projectile from, so building must fail.
        """
        world, cup = world_with_cup
        cup.fill_connection.inflow_equation = InflowEquation(
            container_height=0.1, container_width=0.06
        )
        speedless_source = PourableContainer(
            name=PrefixedName("speedless_source"),
            root=Body(name=PrefixedName("speedless_source_body")),
        )
        assert speedless_source.current_outflow_velocity(world) is None
        no_spill = KeepProjectileInReceiver(receiver=cup, source=speedless_source)

        with pytest.raises(MissingExitSpeedError):
            no_spill.build(MotionStatechartContext(world=world))


class TestProjectileAimingErrorIsHorizontal:
    """
    :class:`~giskardpy.motion_statechart.tasks.pouring.KeepProjectileInReceiver`
    measures only where the arc lands within the receiver's opening.

    Aiming at the receiver's origin, which sits at the base of the container, would add
    a constant vertical error no arm motion can remove, so the task could never report
    the pour as aimed.
    """

    def test_task_reports_aimed_once_the_landing_is_within_the_threshold(
        self, tracy_transfer_world
    ) -> None:
        """
        The task's aiming error equals the horizontal distance from the landing point to the
        opening centre: a threshold above that distance is satisfied, one below it is not.
        """
        world, source_cup, receiving_cup, _left_tool_frame = tracy_transfer_world
        landing = receiving_cup.projectile_landing_point(
            source_cup, world, source_cup.current_outflow_velocity(world)
        )
        opening = receiving_cup.opening_point(world)
        horizontal_distance = math.hypot(
            landing.x.evaluate()[0] - opening.x.evaluate()[0],
            landing.y.evaluate()[0] - opening.y.evaluate()[0],
        )

        def is_aimed(threshold: float) -> bool:
            task = KeepProjectileInReceiver(
                receiver=receiving_cup, source=source_cup, threshold=threshold
            )
            artifacts = task.build(MotionStatechartContext(world=world))
            return bool(artifacts.observation.evaluate()[0])

        assert is_aimed(horizontal_distance * 1.5)
        assert not is_aimed(horizontal_distance * 0.5)


class TestKeepProjectileInReceiverDebugExpressions:
    """
    :class:`~giskardpy.motion_statechart.tasks.pouring.KeepProjectileInReceiver`
    registers the pour's exit point and its projectile landing point as debug
    expressions so they can be visualized as RViz markers.
    """

    def test_registers_exit_and_landing_points(self, tracy_transfer_world) -> None:
        """
        Building the task exposes the exit and landing points as colored point markers.
        """
        world, source_cup, receiving_cup, _left_tool_frame = tracy_transfer_world
        no_spill = KeepProjectileInReceiver(receiver=receiving_cup, source=source_cup)

        artifacts = no_spill.build(MotionStatechartContext(world=world))

        exit_point = debug_expression_by_name(artifacts.debug_expressions, "exit")
        landing_point = debug_expression_by_name(artifacts.debug_expressions, "landing")

        assert isinstance(exit_point.expression, Point3)
        assert isinstance(landing_point.expression, Point3)
        assert exit_point.color == KeepProjectileInReceiver.EXIT_POINT_COLOR
        assert landing_point.color == KeepProjectileInReceiver.LANDING_POINT_COLOR

        # The marker renderer resolves each expression against the live world state; the exit
        # marker must sit exactly on the source's liquid exit point.
        np.testing.assert_allclose(
            exit_point.expression.evaluate(),
            source_cup.liquid_exit_point(world).evaluate(),
        )
        landing_point.expression.evaluate()

    def test_landing_uses_current_outflow_velocity(self, tracy_transfer_world) -> None:
        """
        The landing point is computed from the source's live Torricelli exit speed, not
        the static nominal speed stored on the inflow coupling.
        """
        world, source_cup, receiving_cup, _left_tool_frame = tracy_transfer_world
        no_spill = KeepProjectileInReceiver(receiver=receiving_cup, source=source_cup)

        artifacts = no_spill.build(MotionStatechartContext(world=world))

        landing = debug_expression_by_name(artifacts.debug_expressions, "landing")
        live_speed = source_cup.current_outflow_velocity(world)
        assert live_speed is not None
        # Guard against a vacuous pass: the live Torricelli speed must actually differ from
        # the nominal fallback stored on the inflow coupling.
        nominal_exit_speed = receiving_cup.fill_connection.inflow_equation.exit_speed
        assert live_speed.evaluate()[0] != pytest.approx(nominal_exit_speed)
        expected = receiving_cup.projectile_landing_point(source_cup, world, live_speed)
        assert landing.expression.x.evaluate()[0] == pytest.approx(
            expected.x.evaluate()[0]
        )
        assert landing.expression.y.evaluate()[0] == pytest.approx(
            expected.y.evaluate()[0]
        )


class TestPerceptionCorrectedTransfer:
    """
    Stability of :class:`~giskardpy.motion_statechart.tasks.pouring.FillByTransferTask`
    under noisy fill-level perception at a rate lower than the control loop.

    Models the real-world scenario in which a perception pipeline (e.g., RoboKudo)
    supplies fill-level estimates at ``perception_hz`` while the controller runs at
    :data:`_POURING_TARGET_FREQUENCY`.  After each perception tick the receiver's ODE-
    integrated fill level is replaced by the true value plus additive Gaussian noise, so
    the QP linearizes at the (possibly inaccurate) corrected belief.

    Parametrized over noise standard deviation ``sigma`` and ``perception_hz`` so the
    stability boundary can be explored as both dimensions vary independently.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "sigma,perception_hz",
        [
            (0.01, _DEFAULT_PERCEPTION_HZ),
            (0.02, _DEFAULT_PERCEPTION_HZ),
            (0.01, 20),
            (0.01, 30),
        ],
        ids=[
            "sigma=0.01_10Hz",
            "sigma=0.02_10Hz",
            "sigma=0.01_20Hz",
            "sigma=0.01_30Hz",
        ],
    )
    def test_convergence_under_perception_noise(
        self,
        tracy_transfer_world,
        sigma: float,
        perception_hz: float,
    ) -> None:
        """
        Verifies that the transfer task converges to the fill-level goal when the
        receiver's fill level is observed with Gaussian noise at a sub-control-rate
        frequency.

        Perception updates arrive at ``perception_hz`` and overwrite the ODE-integrated fill
        belief; between updates the ODE integrates freely from the last corrected value.  The
        task terminates only once the perceived fill level reaches the goal within
        ``fill_level_tolerance``, so a noise-adjusted bound is used for the final assertion.

        :param sigma: Standard deviation of additive Gaussian noise on the fill measurement.
        :param perception_hz: Rate of perception updates in Hz.
        """
        goal_fill = 0.7
        tolerance = 0.01

        transfer = _build_transfer_motion(
            tracy_transfer_world, fill_level_tolerance=tolerance
        )
        world = transfer.world
        source_cup = transfer.source_cup
        receiving_cup = transfer.receiving_cup
        transfer_task = transfer.transfer_task

        transfer_executor = Executor(
            _pouring_context(world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        transfer_executor.compile(motion_statechart=transfer.motion_statechart)

        _tick_with_perception_correction(
            executor=transfer_executor,
            world=world,
            fill_connection=receiving_cup.fill_connection,
            sigma=sigma,
            perception_hz=perception_hz,
            rng=np.random.default_rng(seed=42),
        )

        assert transfer_task.observation_state == ObservationStateValues.TRUE, (
            f"Transfer task did not converge "
            f"(sigma={sigma}, perception_hz={perception_hz} Hz)"
        )
        source_loss = 1.0 - float(source_cup.fill_level)
        assert source_loss > tolerance, "source cup never poured"
        # The task terminates once the *perceived* fill reaches the goal band.
        # Noise can shift the perceived value by up to ~sigma relative to the true fill,
        # so the post-termination band is widened by a 2-sigma headroom on both sides.
        noise_headroom = 2.0 * sigma
        assert receiving_cup.fill_level == pytest.approx(
            goal_fill, abs=tolerance + noise_headroom
        ), (
            f"receiver fill {receiving_cup.fill_level:.3f} outside goal band around {goal_fill} "
            f"(sigma={sigma}, perception_hz={perception_hz} Hz)"
        )
