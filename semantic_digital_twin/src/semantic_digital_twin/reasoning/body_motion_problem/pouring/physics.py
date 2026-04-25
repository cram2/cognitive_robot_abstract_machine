"""
Physics model for the pouring domain (D_pour).

Implements Φ_pour using a MotionStatechart with a PouringTask that integrates
the fill-level ODE and controls the tilt joint via QP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pouring import CoupledPouringTask, PouringTask
from giskardpy.qp import qp_controller
from giskardpy.qp.qp_controller_config import QPControllerConfig
import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.physics.differential_equation import DifferentialEquation
from semantic_digital_twin.semantic_annotations.mixins import ContainerGeometry

from semantic_digital_twin.reasoning.body_motion_problem.pouring.articulated import (
    PouringEquation,
)
from semantic_digital_twin.reasoning.body_motion_problem.types import (
    Effect,
    PhysicsModel,
)

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap

from semantic_digital_twin.world_description.connections import (
    HasUpdateState,
    ActiveConnection1DOF,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)


class HasFillLevel:
    """
    Mixin that adds a virtual fill-level DOF to any semantic annotation.

    The fill level is represented as a virtual :class:`LiquidConnection` whose
    position encodes fill in the range ``[0, 1]``. Call :meth:`initialize_fill_level`
    once after the annotation is placed in a world.

    Optionally pass ``tilt_connection`` to :meth:`initialize_fill_level` to also
    create a :attr:`liquid_surface_connection` — a revolute joint that counter-rotates
    by the same DOF so the liquid surface body's XY plane stays parallel to the world
    XY plane regardless of cup tilt.

    After initialisation assign :attr:`fill_equation` (how the cup drains) and
    :attr:`inflow_equation` (how the cup fills from an external source). Both are
    geometry-derived and propagate automatically to the underlying
    :class:`LiquidConnection`.
    Assign :attr:`container_geometry` to provide physical dimensions for
    geometry-aware fill equations.
    """

    fill_connection: Optional[PrismaticConnection] = field(default=None, kw_only=True)
    """The virtual connection whose position encodes fill level in [0, 1]."""

    liquid_surface_connection: Optional[RevoluteConnection] = field(
        default=None, kw_only=True
    )
    """
    Counter-rotation connection whose child body is always world-frame aligned.

    Only present when ``tilt_connection`` was passed to :meth:`initialize_fill_level`.
    """

    container_geometry: Optional[ContainerGeometry] = field(default=None, kw_only=True)
    """Physical dimensions used by geometry-aware fill equations."""

    @property
    def fill_equation(self) -> Optional[PouringEquation]:
        """ODE governing how this container drains when tilted."""
        return self.__dict__.get("_fill_equation")

    @fill_equation.setter
    def fill_equation(self, equation: Optional[PouringEquation]) -> None:
        self.__dict__["_fill_equation"] = equation
        if self.fill_connection is not None:
            self.fill_connection.outflow_equation = equation

    @property
    def inflow_equation(self) -> Optional["InflowEquation"]:
        """ODE governing how this container fills from an external source."""
        return self.__dict__.get("_inflow_equation")

    @inflow_equation.setter
    def inflow_equation(self, equation: Optional["InflowEquation"]) -> None:
        self.__dict__["_inflow_equation"] = equation
        if self.fill_connection is not None:
            self.fill_connection.inflow_equation = equation

    def initialize_fill_level(
        self,
        world: World,
        parent_body: Body,
        initial_fill: float = 1.0,
        tilt_connection: Optional[RevoluteConnection] = None,
    ) -> None:
        """
        Create the virtual fill-level DOF and attach it to the world.

        :param world: The world to add the fill-level DOF to.
        :param parent_body: The body the fill-level DOF is attached to.
        :param initial_fill: Starting fill level in [0, 1].
        :param tilt_connection: When provided, adds a liquid surface body whose
            orientation is always aligned with the world frame by sharing this
            connection's DOF with an inverted multiplier.
        """
        phantom = Body(name=PrefixedName(f"{parent_body.name.name}_fill_level_phantom"))
        with world.modify_world():
            world.add_body(phantom)
            connection = LiquidConnection.create_with_dofs(
                world=world,
                parent=parent_body,
                child=phantom,
                axis=Vector3(0, 0, 1),
                dof_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=0.0, velocity=-1.0),
                    upper=DerivativeMap(position=1.0, velocity=1.0),
                ),
            )
            world.add_connection(connection)
        self.fill_connection = connection
        world.set_positions_1DOF_connection({connection: initial_fill})

        if tilt_connection is not None:
            surface_phantom = Body(
                name=PrefixedName(f"{parent_body.name.name}_liquid_surface_phantom")
            )
            with world.modify_world():
                world.add_body(surface_phantom)
                surface_connection = RevoluteConnection(
                    parent=phantom,
                    child=surface_phantom,
                    axis=tilt_connection.axis,
                    dof_id=tilt_connection.dof_id,
                    multiplier=-tilt_connection.multiplier,
                )
                world.add_connection(surface_connection)
            self.liquid_surface_connection = surface_connection

    @property
    def fill_level(self) -> float:
        """Current fill level in ``[0, 1]``."""
        return float(self.fill_connection.position)

    @property
    def liquid_surface_body(self) -> Body:
        """The body whose XY plane is always parallel to the world XY plane."""
        return self.liquid_surface_connection.child


@dataclass(eq=False)
class LiquidConnection(ActiveConnection1DOF, HasUpdateState):
    """
    Translating DOF representing the fill level of a container.

    Integrates :attr:`outflow_equation` and :attr:`inflow_equation` each tick.
    Either may be ``None``; the net velocity is their sum.
    """

    outflow_equation: Optional[PouringEquation] = field(
        default=None, kw_only=True, init=False
    )
    """ODE governing how liquid leaves this container (e.g. tilting to pour)."""

    inflow_equation: Optional["InflowEquation"] = field(
        default=None, kw_only=True, init=False
    )
    """ODE governing how liquid enters this container from an external source."""

    def add_to_world(self, world: World):
        super().add_to_world(world)
        translation_axis = self.axis * self.dof.variables.position
        self._kinematics = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=translation_axis[0],
            y=translation_axis[1],
            z=translation_axis[2],
            child_frame=self.child,
        )

    def update_state(self, dt: float):
        state = self._world.state
        old_vel = state[self.dof.id].velocity
        old_acc = state[self.dof.id].acceleration
        velocity = sum(
            eq.symbolic_velocity().evaluate()[0]
            for eq in (self.outflow_equation, self.inflow_equation)
            if eq is not None
        )
        state[self.dof.id].velocity = velocity
        state[self.dof.id].acceleration = (velocity - old_vel) / dt
        state[self.dof.id].jerk = (state[self.dof.id].acceleration - old_acc) / dt
        state[self.dof.id].position = state[self.dof.id].position + velocity * dt


@dataclass
class InflowEquation(DifferentialEquation):
    """
    Fill-level ODE for a container receiving liquid from a source.

    Converts the source's outflow volume rate to a normalised fill velocity
    for this container using its own cross-sectional geometry. The
    :attr:`source` field is ``None`` by default and wired per-action by the
    orchestrating model.

    :param container_geometry: Physical dimensions of the receiving container.
    :param source: The container currently pouring into this one.
    """

    container_geometry: ContainerGeometry
    source: Optional[HasFillLevel] = field(default=None)

    def symbolic_velocity(self) -> Scalar:
        """
        :return: Normalised fill velocity from inflow, or zero when no source is active.
        """
        if self.source is None:
            return sm.Scalar(0.0)
        source_outflow = sm.max(
            sm.Scalar(0.0), -self.source.fill_equation.symbolic_velocity()
        )
        source_volume = (
            self.source.container_geometry.half_width
            * self.source.container_geometry.height
        )
        receiver_volume = (
            self.container_geometry.half_width * self.container_geometry.height
        )
        return source_outflow * source_volume / receiver_volume


@dataclass
class PouringMSCModel(PhysicsModel):
    """
    Concrete physics model Φ_pour: execute a PouringTask MotionStatechart against a World.

    The MotionStatechart contains a single :class:`PouringTask` that integrates the
    fill-level ODE and drives the tilt joint via QP. This model compiles the MSC,
    ticks it until EndMotion, records both the tilt and fill trajectories, and
    resets the World state before returning.

    ODE parameters and both connections are carried by :attr:`fill_equation`.
    """

    fill_equation: PouringEquation
    """Pouring ODE — owns the tilt connection, fill connection, and k."""

    target_outflow: float = field(default=0.05)
    """Desired rate of decrease for the normalized fill level."""

    initial_tilt: float = field(default=0.1)
    """Initial tilt angle in radians to help the gradient-driven task."""

    timeout: int = field(default=900)
    """Maximum number of ticks before aborting the rollout."""

    last_fill_trajectory: List[float] = field(default_factory=list, init=False)
    """Fill-level trajectory from the most recent :meth:`run` call."""

    def run(self, effect: Effect, world: World) -> Tuple[Optional[List[float]], bool]:
        """
        Simulate pouring via the PouringTask MotionStatechart and return the tilt trajectory.

        :param effect: Desired fill-level effect; provides goal_value and tolerance.
        :param world: World whose state is simulated and reset on return.
        :return: ``(tilt_trajectory, achieved)`` — tilt samples at each tick and success flag.
        """
        msc = self._build_msc(effect)
        context = MotionStatechartContext(
            world=world,
            qp_controller_config=QPControllerConfig(
                target_frequency=50,
                prediction_horizon=30,
            ),
        )
        executor = Executor(
            context=context,
            pacer=SimulationPacer(real_time_factor=1.0),
        )
        executor.compile(motion_statechart=msc)

        tilt_connection = self.fill_equation.tilt_connection
        fill_connection = self.fill_equation.fill_connection

        tilt_trajectory: List[float] = []
        fill_trajectory: List[float] = []
        with world.reset_state_context():
            # Start with a small tilt to jump-start the gradient-driven task
            world.set_positions_1DOF_connection({tilt_connection: self.initial_tilt})

            for _ in range(self.timeout):
                executor.tick()
                tilt_trajectory.append(float(tilt_connection.position))
                fill_trajectory.append(float(fill_connection.position))
                if msc.is_end_motion():
                    break

            achieved = effect.is_achieved()

        self.last_fill_trajectory = fill_trajectory
        return tilt_trajectory, achieved

    def build_secondary_trajectories(
        self, effect: Effect
    ) -> List[Tuple[Connection, List[float]]]:
        """
        Return the fill-level trajectory recorded by the most recent run().

        :param effect: Effect whose target_object provides the fill connection.
        :return: Single-entry list of (fill_connection, fill_level_positions).
        """
        return [(effect.target_object.fill_connection, self.last_fill_trajectory)]

    def _build_msc(self, effect: Effect) -> MotionStatechart:
        """
        Build the MotionStatechart for a single pouring manoeuvre.

        :param effect: Effect providing goal_value and tolerance for PouringTask.
        :return: Compiled-ready MotionStatechart with PouringTask and EndMotion.
        """
        msc = MotionStatechart()
        task = PouringTask(
            fill_equation=self.fill_equation,
            goal_value=effect.goal_value,
            tolerance=effect.tolerance,
            target_outflow=self.target_outflow,
        )
        msc.add_node(task)
        msc.add_node(local_min := LocalMinimumReached())
        msc.add_node(EndMotion.when_true(local_min))
        return msc


@dataclass
class CoupledPouringMSCModel(PhysicsModel):
    """
    Physics model for the coupled pouring chain: source → receiver.

    Runs a :class:`~giskardpy.motion_statechart.tasks.pouring.CoupledPouringTask`
    MotionStatechart that drives tilt, source fill, receiver fill, and cup body
    position simultaneously via QP constraints. The full FK from ``root_link`` to the
    source cup body keeps the spilling rim above the receiver at every tick.

    :param source: Semantic annotation carrying ``fill_equation``, ``fill_connection``,
        ``container_geometry``, and ``root`` (cup body = tip link).
    :param receiver: Semantic annotation carrying ``fill_connection``,
        ``container_geometry``, and ``root`` (used to read world position).
    :param root_link: Root of the kinematic chain that positions the source cup.
    :param target_outflow: Desired rate of decrease for the normalized fill level.
    :param initial_tilt: Initial tilt angle in radians to help the gradient-driven task.
    :param timeout: Maximum simulation ticks before aborting.
    :param height_above_receiver: Desired rim clearance above the receiver opening (m).
    """

    source: HasFillLevel
    receiver: HasFillLevel
    root_link: Body
    target_outflow: float = field(default=0.05)
    initial_tilt: float = field(default=0.1)
    timeout: int = field(default=2000)
    height_above_receiver: float = field(default=0.1)

    last_source_fill_trajectory: List[float] = field(default_factory=list, init=False)
    last_receiver_fill_trajectory: List[float] = field(default_factory=list, init=False)
    _last_chain_trajectories: Dict[Connection, List[float]] = field(
        default_factory=dict, init=False
    )

    def run(self, effect: Effect, world: World) -> Tuple[Optional[List[float]], bool]:
        """
        Simulate the full coupled pouring chain and return the tilt trajectory.

        :param effect: ReceiverFillEffect targeting the receiver container.
        :param world: World whose state is simulated and reset on return.
        :return: ``(tilt_trajectory, achieved)`` — tilt samples and whether the
            receiver fill reached the effect's goal value.
        """
        self.receiver.inflow_equation.source = self.source
        msc = self._build_msc(effect)
        context = MotionStatechartContext(world=world)
        executor = Executor(
            context=context, pacer=SimulationPacer(real_time_factor=1.0)
        )
        executor.compile(motion_statechart=msc)

        tilt_connection = self.source.fill_equation.tilt_connection
        source_fill_connection = self.source.fill_connection
        receiver_fill_connection = self.receiver.fill_connection
        chain_connections = self._get_kinematic_chain_connections()

        tilt_traj: List[float] = []
        source_fill_traj: List[float] = []
        receiver_fill_traj: List[float] = []
        chain_trajs: Dict[Connection, List[float]] = {c: [] for c in chain_connections}

        with world.reset_state_context():
            # Start with a small tilt to jump-start the gradient-driven task
            world.set_positions_1DOF_connection({tilt_connection: self.initial_tilt})

            for _ in range(self.timeout):
                executor.tick()
                tilt_traj.append(float(tilt_connection.position))
                source_fill_traj.append(float(source_fill_connection.position))
                receiver_fill_traj.append(float(receiver_fill_connection.position))
                for conn in chain_connections:
                    chain_trajs[conn].append(float(conn.position))
                if msc.is_end_motion():
                    break
            achieved = effect.is_achieved()

        self.receiver.inflow_equation.source = None
        self.last_source_fill_trajectory = source_fill_traj
        self.last_receiver_fill_trajectory = receiver_fill_traj
        self._last_chain_trajectories = chain_trajs
        return tilt_traj, achieved

    def build_secondary_trajectories(
        self, effect: Effect
    ) -> List[Tuple[Connection, List[float]]]:
        """
        Return secondary trajectories recorded by the most recent run().

        :param effect: Effect (unused; receiver fill connection is taken from ``self.receiver``).
        :return: Source fill, receiver fill, and all kinematic chain DOF trajectories.
        """
        result: List[Tuple[Connection, List[float]]] = [
            (self.source.fill_connection, self.last_source_fill_trajectory),
            (self.receiver.fill_connection, self.last_receiver_fill_trajectory),
        ]
        result.extend(self._last_chain_trajectories.items())
        return result

    def _get_kinematic_chain_connections(self) -> List[Connection]:
        """Collect all connections from just above the tilt joint up to root_link."""
        connections: List[Connection] = []
        current = self.source.fill_equation.tilt_connection.parent
        while current is not None and current != self.root_link:
            conn = current.parent_connection
            connections.append(conn)
            current = conn.parent
        return connections

    def _build_msc(self, effect: Effect) -> MotionStatechart:
        """
        Build the MotionStatechart for the full coupled pouring manoeuvre.

        :param effect: ReceiverFillEffect providing goal_value and tolerance.
        :return: Compiled-ready MotionStatechart with CoupledPouringTask and EndMotion.
        """
        receiver_center = self.receiver.root.global_transform.to_position().to_np()[:3]
        receiver_target = receiver_center.copy()
        receiver_target[2] += (
            self.receiver.container_geometry.height / 2 + self.height_above_receiver
        )

        msc = MotionStatechart()
        task = CoupledPouringTask(
            fill_equation=self.source.fill_equation,
            goal_value=0.0,
            tolerance=effect.tolerance,
            target_outflow=self.target_outflow,
            receiver_fill_connection=self.receiver.fill_connection,
            root_link=self.root_link,
            tip_link=self.source.root,
            source_geometry=self.source.container_geometry,
            receiver_target=receiver_target,
            goal_receiver_fill=effect.goal_value,
        )
        msc.add_node(task)
        msc.add_node(EndMotion.when_true(task))
        return msc
