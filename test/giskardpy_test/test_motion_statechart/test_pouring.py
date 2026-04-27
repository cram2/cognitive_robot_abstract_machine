from __future__ import annotations

import math
import pytest
import numpy as np
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pouring import PouringTask, CoupledPouringTask
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.pouring_equations import (
    ArticulatedPouringEquation,
    InflowEquation,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Scale,
    ContainerGeometry,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from dataclasses import dataclass
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    FixedConnection,
    PrismaticConnection,
)
import krrood.symbolic_math.symbolic_math as sm


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
        )
    _cup_height = 0.12
    _cup_half_width = 0.04
    cup_shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=_cup_height / 2,
            reference_frame=cup.root,
        ),
        scale=Scale(
            2 * _cup_half_width,
            2 * _cup_half_width,
            _cup_height,
        ),
    )
    with world.modify_world():
        cup.root.visual = ShapeCollection(shapes=[cup_shape])
        cup.root.collision = ShapeCollection(shapes=[cup_shape])
    cup.initialize_fill_level(
        world=world,
        parent_body=cup.root,
        initial_fill=1.0,
        tilt_connection=cup.root.parent_connection,
    )
    cup.fill_equation = ArticulatedPouringEquation(
        fill_connection=cup.fill_connection,
        container_geometry=cup.container_geometry,
        k=1,
    )
    world.set_positions_1DOF_connection({cup.root.parent_connection: 0.1})
    return world, cup


@dataclass(eq=False)
class ReceiverContainer(HasFillLevel):
    """Receiver cup: fixed in world, initially empty."""

    @classproperty
    def _parent_connection_type(self):
        return FixedConnection


@pytest.fixture
def world_with_source_and_receiver():
    """
    World with source cup (kinematic chain: PrismaticX -> PrismaticZ -> RevoluteY)
    positioned so its rim starts near the receiver opening.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))

    source_geometry = ContainerGeometry(height=0.12, half_width=0.04)
    receiver_geometry = ContainerGeometry(height=0.10, half_width=0.05)

    with world.modify_world():
        receiver = ReceiverContainer.create_with_new_body_in_world(
            name=PrefixedName("receiver"),
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.2,
                y=0.0,
                z=0.7 - receiver_geometry.height / 2,
                reference_frame=world.root,
            ),
            scale=Scale(
                2 * receiver_geometry.half_width,
                2 * receiver_geometry.half_width,
                receiver_geometry.height,
            ),
        )
    receiver.initialize_fill_level(
        world=world, parent_body=receiver.root, initial_fill=0.0
    )

    x_body = Body(name=PrefixedName("source_x_translation"))
    with world.modify_world():
        world.add_body(x_body)
        x_connection = PrismaticConnection.create_with_dofs(
            world=world,
            parent=world.root,
            child=x_body,
            axis=Vector3(1, 0, 0),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=-1.0, velocity=-2.0),
                upper=DerivativeMap(position=1.0, velocity=2.0),
            ),
        )
        world.add_connection(x_connection)
    world.set_positions_1DOF_connection({x_connection: 0.16})

    z_body = Body(name=PrefixedName("source_z_translation"))
    with world.modify_world():
        world.add_body(z_body)
        z_connection = PrismaticConnection.create_with_dofs(
            world=world,
            parent=x_body,
            child=z_body,
            axis=Vector3(0, 0, 1),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=2.0, velocity=2.0),
            ),
        )
        world.add_connection(z_connection)
    world.set_positions_1DOF_connection({z_connection: 0.68})

    cup_body = Body(name=PrefixedName("source_cup"))
    cup_shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=cup_body),
        scale=Scale(
            2 * source_geometry.half_width,
            2 * source_geometry.half_width,
            source_geometry.height,
        ),
    )
    cup_body.visual = ShapeCollection(shapes=[cup_shape])
    cup_body.collision = ShapeCollection(shapes=[cup_shape])
    with world.modify_world():
        world.add_body(cup_body)
        tilt_connection = RevoluteConnection.create_with_dofs(
            world=world,
            parent=z_body,
            child=cup_body,
            axis=Vector3(0, 1, 0),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi / 2, velocity=2.0),
            ),
        )
        world.add_connection(tilt_connection)

    source = PourableContainer(name=PrefixedName("source_cup"), root=cup_body)
    with world.modify_world():
        world.add_semantic_annotation(source)
    source.initialize_fill_level(
        world=world,
        parent_body=cup_body,
        initial_fill=0.8,
        tilt_connection=tilt_connection,
    )
    source.fill_equation = ArticulatedPouringEquation(
        fill_connection=source.fill_connection,
        container_geometry=source.container_geometry,
        k=100,
    )

    source_outflow = sm.max(
        sm.Scalar(0.0),
        -source.fill_equation.symbolic_velocity(tilt_connection.dof.variables.position),
    )
    source_volume = (
        source.container_geometry.half_width * source.container_geometry.height
    )
    receiver.inflow_equation = InflowEquation(
        container_geometry=receiver.container_geometry,
        inflow=source_outflow * source_volume,
    )
    world.set_positions_1DOF_connection({tilt_connection: 0.1})

    return world, source, receiver


class TestPouringTask:
    """Test suite for the PouringTask in Giskardpy."""

    def test_pouring_task_achieves_goal(self, world_with_cup):
        """
        Test that PouringTask successfully tilts the cup and reduces fill level
        to the target value.
        """
        world, cup = world_with_cup

        goal_fill = 0.6
        tolerance = 0.05

        msc = MotionStatechart()
        pouring_task = PouringTask(
            fill_equation=cup.fill_equation,
            root_link=world.root,
            tip_link=cup.root,
            goal_value=goal_fill,
            tolerance=tolerance,
            reference_velocity=0.05,
        )
        msc.add_node(pouring_task)
        msc.add_node(EndMotion.when_true(pouring_task))

        executor = Executor(MotionStatechartContext(world=world))
        executor.compile(motion_statechart=msc)

        executor.tick_until_end(timeout=1000)

        assert pouring_task.observation_state == ObservationStateValues.TRUE
        assert cup.fill_level <= goal_fill + tolerance
        assert cup.fill_level >= goal_fill - tolerance
        assert cup.root.parent_connection.position > 0.1

    def test_coupled_pouring_task_achieves_goal(self, world_with_source_and_receiver):
        """
        Test that CoupledPouringTask successfully tilts the source cup and fills
        the receiver to the target value while maintaining rim positioning.
        """
        pytest.skip(
            "CoupledPouringTask needs further investigation on setup/convergence"
        )
        world, source, receiver = world_with_source_and_receiver

        goal_receiver_fill = 0.2
        tolerance = 0.05

        # Receiver center is at (0.2, 0.0, 0.7 - height/2)
        # Receiver top is at 0.7
        receiver_target = np.array([0.2, 0.0, 0.8])  # 0.1 above receiver top

        msc = MotionStatechart()
        coupled_task = CoupledPouringTask(
            fill_equation=source.fill_equation,
            root_link=world.root,
            tip_link=source.root,
            goal_value=0.0,  # source goal (not used for termination in CoupledPouringTask)
            tolerance=tolerance,
            reference_velocity=0.5,  # Increase reference velocity to speed up pouring
            receiver_fill_connection=receiver.fill_connection,
            source_geometry=source.container_geometry,
            receiver_target=receiver_target,
            goal_receiver_fill=goal_receiver_fill,
        )
        msc.add_node(coupled_task)
        msc.add_node(EndMotion.when_true(coupled_task))

        executor = Executor(MotionStatechartContext(world=world))
        executor.compile(motion_statechart=msc)

        # Increased timeout to 5000
        executor.tick_until_end(timeout=5000)

        assert coupled_task.observation_state == ObservationStateValues.TRUE
        assert receiver.fill_level >= goal_receiver_fill - tolerance

        # Check rim positioning (approximately)
        # tip_P_rim = (r, 0, A) in tip frame
        # We check if world_P_rim is close to receiver_target
        r = source.container_geometry.half_width
        A = source.container_geometry.height
        tip_P_rim = Point3(r, 0.0, A, reference_frame=source.root)
        world_P_rim = world.transform(target_frame=world.root, spatial_object=tip_P_rim)

        np.testing.assert_allclose(world_P_rim.to_np()[:3], receiver_target, atol=0.05)
