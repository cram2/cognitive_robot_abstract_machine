from __future__ import annotations

import math
import pytest
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    ObservationStateValues,
    DefaultWeights,
)
from giskardpy.motion_statechart.exceptions import NodeInitializationError
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.pouring import PouringTask
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Mesh,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from dataclasses import dataclass
from krrood.ormatic.utils import classproperty

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


def _pouring_context(world: World) -> MotionStatechartContext:
    """
    Builds a context whose QP runs at a high frequency over a long prediction horizon.

    The long horizon lets the linearized fill prediction span the pouring overshoot, so the
    constraint converges without a reactive damping term.
    """
    return MotionStatechartContext(
        world=world,
        qp_controller_config=QPControllerConfig(
            target_frequency=_POURING_TARGET_FREQUENCY,
            prediction_horizon=_POURING_PREDICTION_HORIZON,
        ),
    )


@dataclass(eq=False)
class PourableContainer(HasFillLevel):
    """
    Minimal pourable container for testing.

    Connected to its parent via a revolute joint representing the tilt angle.
    """

    @classproperty
    def _parent_connection_type(self):
        return RevoluteConnection


@pytest.fixture
def pr2_world_setup(pr2_world_copy):
    """Function-scoped PR2 world, suitable for tests that modify the world."""
    return pr2_world_copy


@pytest.fixture
def world_with_cup():
    """World containing a single pourable container with a tilt joint, filled to 100%."""
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    with world.modify_world():
        cup = PourableContainer.create_with_new_body_in_world(
            name=PrefixedName("cup"),
            world=world,
            active_axis=Vector3(0, 1, 0),
            connection_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi / 2, velocity=2.0),
            ),
            scale=Scale(0.4, 0.4, 1.0),
        )
    cup.initialize_fill_level(
        world=world,
        initial_fill=1.0,
        outflow_rate_constant=1,
    )
    world.set_positions_1DOF_connection({cup.root.parent_connection: 0.1})
    return world, cup


def _spawn_jeroen_cup_body(name: str) -> Body:
    """Create a Body with the Jeroen cup mesh geometry."""
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


class TestPouringTask:
    """Test suite for the PouringTask in Giskardpy."""

    def test_pouring_task_achieves_goal(self, world_with_cup, rclpy_node):
        """
        Test that PouringTask successfully tilts the cup and reduces fill level
        to the target value.
        """
        world, cup = world_with_cup
        VizMarkerPublisher(_world=world, node=rclpy_node).with_tf_publisher()
        goal_fill = 0.6
        tolerance = 0.05

        msc = MotionStatechart()
        pouring_task = PouringTask(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            root_link=world.root,
            tip_link=cup.root,
            goal_value=goal_fill,
            fill_level_tolerance=tolerance,
            reference_velocity=0.05,
        )
        msc.add_node(pouring_task)
        msc.add_node(EndMotion.when_true(pouring_task))

        executor = Executor(
            _pouring_context(world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        executor.compile(motion_statechart=msc)

        executor.tick_until_end(timeout=4000)

        assert pouring_task.observation_state == ObservationStateValues.TRUE
        assert cup.fill_level <= goal_fill + tolerance
        assert cup.fill_level >= goal_fill - tolerance
        assert cup.root.parent_connection.position > 0.1
        assert cup.fill_equation.symbolic_velocity(cup.fill_connection).evaluate()[
            0
        ] == pytest.approx(0.0, abs=1e-2)

    def test_build_rejects_non_world_root_link(self, world_with_cup):
        """
        The cup tilt is derived from the world root, so building with any other root link must
        fail loudly rather than silently mispredict the pour against a non-vertical reference.
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
        with pytest.raises(NodeInitializationError):
            pouring_task.build(_pouring_context(world))

    def test_proactive_tilt_back(self, world_with_cup, rclpy_node):
        """
        Verify that the linearized MPC starts reducing tilt before the fill level reaches
        the goal, demonstrating proactive rather than purely reactive control.

        The cup tilt must begin decreasing while the fill level is still strictly above
        ``goal_value + fill_level_tolerance``.
        """
        world, cup = world_with_cup
        goal_fill = 0.6
        tolerance = 0.05

        msc = MotionStatechart()
        pouring_task = PouringTask(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            root_link=world.root,
            tip_link=cup.root,
            goal_value=goal_fill,
            fill_level_tolerance=tolerance,
            reference_velocity=0.05,
        )
        msc.add_node(pouring_task)
        msc.add_node(EndMotion.when_true(pouring_task))

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
        executor.compile(motion_statechart=msc)
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

    def test_pr2_pouring_from_gripper(self, pr2_world_setup, rclpy_node):
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

        VizMarkerPublisher(_world=world, node=rclpy_node).with_tf_publisher()

        # Run PouringTask
        goal_fill = 0.6
        tolerance = 0.05
        msc = MotionStatechart()
        pouring_task = PouringTask(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            root_link=world.root,
            tip_link=cup_body,
            goal_value=goal_fill,
            fill_level_tolerance=tolerance,
        )
        msc.add_node(pouring_task)
        msc.add_node(EndMotion.when_true(pouring_task))

        executor = Executor(
            _pouring_context(world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        executor.compile(motion_statechart=msc)

        executor.tick_until_end(timeout=4000)

        assert pouring_task.observation_state == ObservationStateValues.TRUE
        assert cup.fill_level == pytest.approx(goal_fill, abs=tolerance)
        assert cup.fill_equation.symbolic_velocity(cup.fill_connection).evaluate()[
            0
        ] == pytest.approx(0.0, abs=1e-2)


class TestTracyPouring:
    """Test suite for PouringTask using the Tracy dual-arm robot."""

    def add_left_wrist_3_offset(self, world: World, offset: float) -> None:
        """Add an angular offset to the current left_wrist_3_joint position."""
        joint = world.get_connection_by_name("left_wrist_3_joint")
        world.set_positions_1DOF_connection({joint: joint.position + offset})

    def test_tracy_pouring(self, tracy_pouring_world, rclpy_node):
        """
        Test that PouringTask reduces the fill level of a Jeroen cup held in
        Tracy's left gripper from 1.0 to 0.5.

        A CartesianPose task first moves the left gripper to the upright pose,
        then the cup is grasped and pouring begins.
        """
        world, tracy = tracy_pouring_world
        VizMarkerPublisher(_world=world, node=rclpy_node).with_tf_publisher()
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

        msc_cartesian = MotionStatechart()
        cartesian_task = CartesianPose(
            root_link=world.root, tip_link=left_tool_frame, goal_pose=upright_pose
        )
        msc_cartesian.add_node(cartesian_task)
        msc_cartesian.add_node(EndMotion.when_true(cartesian_task))

        cartesian_executor = Executor(
            MotionStatechartContext(world=world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        cartesian_executor.compile(motion_statechart=msc_cartesian)
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

        msc_pouring = MotionStatechart()
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
            weight=DefaultWeights.WEIGHT_ABOVE_CA,
        )
        motion = Parallel([pouring_task, keep_position])
        msc_pouring.add_node(motion)
        msc_pouring.add_node(EndMotion.when_true(motion))

        pouring_executor = Executor(
            _pouring_context(world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        pouring_executor.compile(motion_statechart=msc_pouring)
        pouring_executor.tick_until_end(timeout=4000)

        assert pouring_task.observation_state == ObservationStateValues.TRUE
        assert grasped_cup.fill_level == pytest.approx(goal_fill, abs=tolerance)
        assert grasped_cup.fill_equation.symbolic_velocity(
            grasped_cup.fill_connection
        ).evaluate()[0] == pytest.approx(0.0, abs=1e-2)


class TestTracyLiquidTransfer:
    """
    Test suite for cup-to-cup liquid transfer driven by a fill-level goal on the receiver.

    The commanded goal is the fill level of a *receiving* cup standing on the table; the
    only controllable degrees of freedom belong to the arm holding the *source* cup. The
    optimizer must therefore tilt the source cup so that liquid leaving it lands in the
    receiver and raises the receiver's fill level to the goal.
    """

    def test_tracy_liquid_transfer_fills_receiver(
        self, tracy_pouring_world, rclpy_node
    ):
        """
        Commanding a fill-level goal on the receiving cup makes the optimizer tilt the
        grasped source cup until the receiver reaches the goal.

        The transfer is volume conserving: while the source rim is above the receiver the
        volume the source loses equals the volume the receiver gains, so no liquid is spilled.
        """
        from giskardpy.motion_statechart.tasks.pouring import (
            FillByTransferTask,
            KeepProjectileInReceiver,
            KeepSourceAboveReceiver,
        )

        world, tracy = tracy_pouring_world
        VizMarkerPublisher(_world=world, node=rclpy_node).with_tf_publisher()
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

        msc_cartesian = MotionStatechart()
        cartesian_task = CartesianPose(
            root_link=world.root, tip_link=left_tool_frame, goal_pose=upright_pose
        )
        msc_cartesian.add_node(cartesian_task)
        msc_cartesian.add_node(EndMotion.when_true(cartesian_task))

        cartesian_executor = Executor(
            MotionStatechartContext(world=world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        cartesian_executor.compile(motion_statechart=msc_cartesian)
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
        # Keep the source cup above the receiver so the optimizer never lowers it into the receiver.
        minimum_clearance = 0.05
        keep_above = KeepSourceAboveReceiver(
            source=source_cup,
            receiver=receiving_cup,
            minimum_clearance=minimum_clearance,
        )
        keep_plane = AlignPlanes(
            root_link=world.root,
            tip_link=left_tool_frame,
            goal_normal=Vector3.X(reference_frame=world.root),
            tip_normal=Vector3.Z(reference_frame=left_tool_frame),
        )
        motion = Parallel([transfer_task, no_spill, keep_above, keep_plane])
        msc_transfer = MotionStatechart()
        msc_transfer.add_node(motion)
        msc_transfer.add_node(EndMotion.when_true(motion))

        gate_history: list[float] = []
        tilt_history: list[float] = []
        clearance_history: list[float] = []
        original_on_tick = transfer_task.on_tick

        def recording_on_tick(context):
            gate_history.append(float(transfer_task.inflow_equation.gate.evaluate()[0]))
            tilt_history.append(
                float(
                    transfer_task.inflow_equation.source_tilt_expression.evaluate()[0]
                )
            )
            source_z = world.compute_forward_kinematics_np(world.root, source_cup_body)[
                2, 3
            ]
            receiver_z = world.compute_forward_kinematics_np(
                world.root, receiving_cup_body
            )[2, 3]
            clearance_history.append(float(source_z - receiver_z))
            return original_on_tick(context)

        transfer_task.on_tick = recording_on_tick

        transfer_executor = Executor(
            _pouring_context(world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        transfer_executor.compile(motion_statechart=msc_transfer)
        transfer_executor.tick_until_end(timeout=4000)

        assert transfer_task.observation_state == ObservationStateValues.TRUE
        assert receiving_cup.fill_level == pytest.approx(goal_fill, abs=tolerance)

        receiver_gain = receiving_cup.fill_level
        print(receiver_gain)
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
        assert all(gate > 0.5 for gate in gate_history[first_open_tick:]), (
            "the liquid's projectile left the receiver mid-pour (no-spill failed): "
            f"minimum gate after aiming was {min(gate_history[first_open_tick:]):.3f}"
        )
        assert min(clearance_history) > 0.0, (
            "the source cup dropped to or below the receiver during the pour: "
            f"minimum clearance was {min(clearance_history):.3f}"
        )
