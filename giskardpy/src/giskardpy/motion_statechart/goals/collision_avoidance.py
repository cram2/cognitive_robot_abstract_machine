from dataclasses import field, dataclass
from itertools import combinations
from typing import Optional

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import (
    NodeInitializationError,
    CollisionViolatedError,
)
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
    CancelMotion,
)
from giskardpy.motion_statechart.graph_node import Task
from krrood.symbolic_math.symbolic_math import Scalar, FloatVariable
from semantic_digital_twin.collision_checking.collision_groups import CollisionGroup
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionRule,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.collision_rules import AvoidSelfCollisions
from semantic_digital_twin.collision_checking.collision_variable_managers import (
    SelfCollisionVariableManager,
    ExternalCollisionVariableManager,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import (
    Vector3,
    Point3,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    Body,
)


@dataclass(eq=False, repr=False)
class _CollisionAvoidanceTask(Task):
    """
    Superclass with helper methods for collision avoidance tasks.
    """


@dataclass(eq=False, repr=False)
class _ExternalCollisionAvoidanceNode(_CollisionAvoidanceTask):
    """
    Avoids external collisions between a collision group and its collision_index-closest object in the environment.
    Moves `root_T_tip @ tip_P_contact` in `root_T_contact_normal` direction until the distance is larger than buffer_zone.
    Limits the slack variables to prevent the tip from coming closer than violated_distance.
    .. warning: Can result in insolvable QPs if multiple of these constraints are violated.
    """

    collision_group: CollisionGroup = field(kw_only=True)
    """
    The collision group avoiding external collisions.
    """
    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """
    collision_index: int = field(default=0, kw_only=True)
    """
    The index of the closest object in the collision group.
    e.g. of collision_index=1 it will avoid the 2. closest contact.
    """
    external_collision_manager: ExternalCollisionVariableManager = field(kw_only=True)
    """
    Reference to the external collision variable manager shared by other external collision avoidance nodes.
    """

    @property
    def root_V_contact_normal(self) -> Vector3:
        return self.external_collision_manager.get_root_V_contact_normal_symbol(
            self.collision_group, self.collision_index
        )

    @property
    def group_a_P_point_on_a(self) -> Point3:
        return self.external_collision_manager.get_group_a_P_point_on_a_symbol(
            self.collision_group, self.collision_index
        )

    @property
    def contact_distance(self):
        return self.external_collision_manager.get_contact_distance_symbol(
            self.collision_group, self.collision_index
        )

    @property
    def buffer_zone_distance(self):
        return self.external_collision_manager.get_buffer_distance_symbol(
            self.collision_group, self.collision_index
        )

    @property
    def violated_distance(self):
        return self.external_collision_manager.get_violated_distance_symbol(
            self.collision_group, self.collision_index
        )

    @property
    def has_collision_data(self) -> Scalar:
        return self.root_V_contact_normal.norm() == 0


@dataclass(eq=False, repr=False)
class _ExternalCollisionHasData(_ExternalCollisionAvoidanceNode):
    """
    Monitors whether data was computed for the external collision avoidance task.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        artifacts.observation = self.has_collision_data

        return artifacts


@dataclass(eq=False, repr=False)
class _ExternalCollisionAvoidanceTask(_ExternalCollisionAvoidanceNode):
    """
    Avoids external collisions between a collision group and its collision_index-closest object in the environment.
    Moves `root_T_tip @ tip_P_contact` in `root_T_contact_normal` direction until the distance is larger than buffer_zone.
    Limits the slack variables to prevent the tip from coming closer than violated_distance.
    .. warning: Can result in insolvable QPs if multiple of these constraints are violated.
    """

    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """

    @property
    def tip(self) -> KinematicStructureEntity:
        return self.collision_group.root

    def create_weight(self, context: MotionStatechartContext) -> sm.Scalar:
        """
        Creates a weight expression for this task which is scaled by the number of external collisions.
        """
        max_avoided_bodies = self.collision_group.get_max_avoided_bodies(
            context.collision_manager
        )
        number_of_external_collisions = 0
        for index in range(max_avoided_bodies):
            has_collision_data = (
                self.external_collision_manager.get_root_V_contact_normal_symbol(
                    self.collision_group, index
                ).norm()
            )
            is_active = has_collision_data > 0
            number_of_external_collisions += is_active
        weight = sm.Scalar(
            data=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE
        ).safe_division(sm.min(number_of_external_collisions, max_avoided_bodies))
        return weight

    def is_no_collision_violated(self, context: MotionStatechartContext) -> Scalar:
        max_avoided_bodies = self.collision_group.get_max_avoided_bodies(
            context.collision_manager
        )
        return sm.logic_all(
            [
                self.contact_distance >= self.violated_distance
                for _ in range(max_avoided_bodies)
            ]
        )

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        root_T_group_a = context.world.compose_forward_kinematics_expression(
            context.world.root, self.tip
        )

        root_V_point_on_a = (root_T_group_a @ self.group_a_P_point_on_a).to_vector3()

        # the position distance is not accurate, but the derivative is still correct
        a_projected_on_normal = self.root_V_contact_normal @ root_V_point_on_a

        lower_limit = self.buffer_zone_distance - self.contact_distance

        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.max_velocity,
            lower_error=lower_limit,
            upper_error=float("inf"),
            quadratic_weight=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE,
            task_expression=a_projected_on_normal,
        )

        artifacts.observation = self.is_no_collision_violated(context)

        return artifacts


@dataclass(eq=False, repr=False)
class _CancelBecauseExternalCollisionViolated(CancelMotion):
    """
    Cancels the motion by raising an exception detailing which external collision tasks were violated.
    """

    tasks: list[_ExternalCollisionAvoidanceTask] = field(kw_only=True)
    """
    The list of external collision avoidance tasks to check for collisions.
    """
    exception: Exception = field(init=False, default=Exception)
    """
    Set to init=False, because this class creates its own exception.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        if len(self.tasks) == 1:
            self.start_condition = sm.trinary_logic_not(
                self.tasks[0].observation_variable
            )
        else:
            self.start_condition = sm.trinary_logic_or(
                *[
                    sm.trinary_logic_not(node.observation_variable)
                    for node in self.tasks
                ]
            )
        return NodeArtifacts()

    def on_tick(self, context: MotionStatechartContext) -> Optional[float]:
        violated_tasks = [
            task
            for task in self.tasks
            if task.observation_state == ObservationStateValues.FALSE
        ]
        collisions = []
        thresholds = []
        for task in violated_tasks:
            collision = context.external_collision_manager.last_closest_contacts[
                task.collision_group
            ][0]
            collisions.append(collision)
            thresholds.append(task.violated_distance.evaluate()[0])
        raise CollisionViolatedError(
            violated_collisions=collisions, thresholds=thresholds
        )


@dataclass(eq=False, repr=False)
class UpdateTemporaryCollisionRules(MotionStatechartNode):
    """
    Updates the temporary collision rules for the robot.
    """

    temporary_rules: list[CollisionRule] = field(kw_only=True)
    collision_matrix: CollisionMatrix = field(init=False)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        # safe old rules
        old_temporary_rules = context.collision_manager.temporary_rules

        # compute collision matrix with new rules
        context.collision_manager.clear_temporary_rules()
        context.collision_manager.extend_temporary_rule(self.temporary_rules)
        context.collision_manager.update_collision_matrix()
        self.collision_matrix = context.collision_manager.collision_matrix

        context.collision_manager.clear_temporary_rules()
        context.collision_manager.extend_temporary_rule(old_temporary_rules)
        context.collision_manager.update_collision_matrix()

        artifacts.observation = sm.Scalar.const_true()
        return artifacts

    def on_start(self, context: MotionStatechartContext):
        context.collision_manager.clear_temporary_rules()
        context.collision_manager.extend_temporary_rule(self.temporary_rules)
        context.collision_manager.set_collision_matrix(self.collision_matrix)


@dataclass(eq=False, repr=False)
class SetInitialTemporaryCollisionRules(MotionStatechartNode):
    """
    Updates the temporary collision rules for the robot.
    """

    temporary_rules: list[CollisionRule] = field(kw_only=True)
    collision_matrix: CollisionMatrix = field(init=False)
    set_on_build: bool = field(default=True, kw_only=True)
    """Whether to set the collision matrix on build."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        # safe old rules
        old_temporary_rules = context.collision_manager.temporary_rules

        # compute collision matrix with new rules
        context.collision_manager.clear_temporary_rules()
        context.collision_manager.extend_temporary_rule(self.temporary_rules)
        context.collision_manager.update_collision_matrix()
        self.collision_matrix = context.collision_manager.collision_matrix

        # restore old rules
        if not self.set_on_build:
            context.collision_manager.clear_temporary_rules()
            context.collision_manager.extend_temporary_rule(old_temporary_rules)
            context.collision_manager.update_collision_matrix()

        artifacts.observation = sm.Scalar.const_true()
        return artifacts

    def on_start(self, context: MotionStatechartContext):
        context.collision_manager.set_collision_matrix(self.collision_matrix)


@dataclass(eq=False, repr=False)
class ExternalCollisionAvoidance(Goal):
    """
    A goal combining an ExternalCollisionDistanceMonitor and an ExternalCollisionAvoidanceTask.
    One pair will be added for all collision groups of the robot.
    The task will only be active if the monitor detects that a collision is close.
    """

    robot: AbstractRobot = field(kw_only=True, default=None)
    """
    The robot for which the collision avoidance goal is defined.
    """
    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """
    external_collision_manager: ExternalCollisionVariableManager = field(init=False)
    """
    Reference to the external collision variable manager shared by other external collision avoidance nodes.
    """
    cancel_if_collision_violated: bool = field(default=True, kw_only=True)
    """
    If True, the motion will be canceled if a collision is violated.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        if self.robot is None:
            robots = context.world.get_semantic_annotations_by_type(AbstractRobot)
            if len(robots) != 1:
                raise NodeInitializationError(
                    self, f"Expected exactly one robot, got {len(robots)}"
                )
            self.robot = robots[0]
        self.external_collision_manager = context.external_collision_manager

        for body in self.robot.bodies_with_collision:
            if context.collision_manager.get_max_avoided_bodies(body):
                self.external_collision_manager.register_group_of_body(body)

        robot_bodies = self.robot.bodies

        tasks = []

        for group in self.external_collision_manager.registered_groups:
            if group.root not in robot_bodies:
                continue
            max_avoided_bodies = group.get_max_avoided_bodies(context.collision_manager)
            for index in range(max_avoided_bodies):
                distance_monitor = _ExternalCollisionHasData(
                    name=f"{self.name}/monitor({group.root.name.name, index})",
                    collision_group=group,
                    collision_index=index,
                    external_collision_manager=self.external_collision_manager,
                )
                self.add_node(distance_monitor)

                task = _ExternalCollisionAvoidanceTask(
                    name=f"{self.name}/task({group.root.name.name, index})",
                    collision_group=group,
                    max_velocity=self.max_velocity,
                    collision_index=index,
                    external_collision_manager=self.external_collision_manager,
                )
                self.add_node(task)
                task.pause_condition = distance_monitor.observation_variable
                tasks.append(task)

        if self.cancel_if_collision_violated:
            self.add_node(
                _CancelBecauseExternalCollisionViolated(
                    tasks=tasks,
                    name="External Collision Violated",
                )
            )


@dataclass(eq=False, repr=False)
class ExternalCollisionDistanceMonitor(MotionStatechartNode):
    """
    Monitors the distance to the closest external object for a specific collision group of a body.
    Turns True if the distance falls below a given threshold.

    .. note:: the input bodies are only used to look up the collision groups.
    """

    body: Body = field(kw_only=True)
    """The robot body to monitor."""
    threshold: float = field(kw_only=True)
    """Distance threshold in meters."""
    collision_index: int = field(default=0, kw_only=True)
    """Index of the closest collision (0 = closest, 1 = second closest, etc.)."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        # 1. Access the shared external collision manager
        # This automatically registers the manager with the CollisionManager
        manager = context.external_collision_manager

        # 2. Register the body to ensure the manager tracks its collisions
        manager.register_group_of_body(self.body)
        group = manager.get_collision_group(self.body)

        # 3. Retrieve the symbolic variable for the contact distance
        distance_symbol = manager.get_contact_distance_symbol(
            group=group, idx=self.collision_index
        )

        # 4. Return an observation artifact
        # The node's observation_variable will be True when distance < threshold
        return NodeArtifacts(observation=distance_symbol < self.threshold)


@dataclass(eq=False, repr=False)
class _SelfCollisionAvoidanceNode(_CollisionAvoidanceTask):
    """
    Avoids self collisions between two collision groups.
    Moves `group_a_P_point_on_a @ group_b_P_point_on_b` in `group_a_T_group_b_contact_normal` direction until the distance is larger than buffer_zone.
    Limits the slack variables to prevent the tip from coming closer than violated_distance.
    """

    collision_group_a: CollisionGroup = field(kw_only=True)
    """
    The first collision group to avoid self collisions with.
    """
    collision_group_b: CollisionGroup = field(kw_only=True)
    """
    The second collision group to avoid self collisions with.
    """
    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """
    self_collision_manager: SelfCollisionVariableManager = field(kw_only=True)
    """
    Reference to the self collision variable manager shared by other self collision avoidance nodes.
    """

    @property
    def group_a_P_point_on_a(self) -> Point3:
        return self.self_collision_manager.get_group_a_P_point_on_a_symbol(
            self.collision_group_a, self.collision_group_b
        )

    @property
    def group_b_P_point_on_b(self) -> Point3:
        return self.self_collision_manager.get_group_b_P_point_on_b_symbol(
            self.collision_group_a, self.collision_group_b
        )

    @property
    def group_b_V_contact_normal(self) -> Vector3:
        return self.self_collision_manager.get_group_b_V_contact_normal_symbol(
            self.collision_group_a, self.collision_group_b
        )

    @property
    def contact_distance(self) -> FloatVariable:
        return self.self_collision_manager.get_contact_distance_symbol(
            self.collision_group_a,
            self.collision_group_b,
        )

    @property
    def buffer_zone_distance(self) -> FloatVariable:
        return self.self_collision_manager.get_buffer_distance_symbol(
            self.collision_group_a, self.collision_group_b
        )

    @property
    def violated_distance(self) -> FloatVariable:
        return self.self_collision_manager.get_violated_distance_symbol(
            self.collision_group_a, self.collision_group_b
        )

    @property
    def has_collision_data(self) -> Scalar:
        return self.group_b_V_contact_normal.norm() == 0


@dataclass(eq=False, repr=False)
class _SelfCollisionHasData(_SelfCollisionAvoidanceNode):
    """
    Monitors whether data was computed for the self collision avoidance task.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        artifacts.observation = self.has_collision_data

        return artifacts


@dataclass(eq=False, repr=False)
class _SelfCollisionAvoidanceTask(_SelfCollisionAvoidanceNode):
    """
    Avoids self collisions between two collision groups.
    Moves `group_a_P_point_on_a @ group_b_P_point_on_b` in `group_a_T_group_b_contact_normal` direction until the distance is larger than buffer_zone.
    Limits the slack variables to prevent the tip from coming closer than violated_distance.
    """

    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """

    def is_collision_not_violated(self) -> Scalar:
        return self.contact_distance >= self.violated_distance

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        group_b_T_group_a = context.world.compose_forward_kinematics_expression(
            self.collision_group_b.root, self.collision_group_a.root
        )

        group_b_P_point_on_a = group_b_T_group_a @ self.group_a_P_point_on_a

        group_b_V_point_on_b_to_point_on_a = (
            group_b_P_point_on_a - self.group_b_P_point_on_b
        )

        a_projected_on_normal = (
            self.group_b_V_contact_normal @ group_b_V_point_on_b_to_point_on_a
        )

        lower_limit = self.buffer_zone_distance - self.contact_distance

        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.max_velocity,
            lower_error=lower_limit,
            upper_error=float("inf"),
            quadratic_weight=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE,
            task_expression=a_projected_on_normal,
        )

        artifacts.observation = self.is_collision_not_violated()
        return artifacts


@dataclass(eq=False, repr=False)
class _CancelBecauseSelfCollisionViolated(CancelMotion):
    """
    Cancels the motion by raising an exception detailing which self collision tasks were violated.
    """

    tasks: list[_SelfCollisionAvoidanceTask] = field(kw_only=True)
    """
    The list of self collision avoidance tasks to check for collisions.
    """
    exception: Exception = field(init=False, default=Exception)
    """
    Set to init=False, because this class creates its own exception.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        if len(self.tasks) == 1:
            self.start_condition = sm.trinary_logic_not(
                self.tasks[0].observation_variable
            )
        else:
            self.start_condition = sm.trinary_logic_or(
                *[
                    sm.trinary_logic_not(node.observation_variable)
                    for node in self.tasks
                ]
            )
        return NodeArtifacts()

    def on_tick(self, context: MotionStatechartContext) -> Optional[float]:
        violated_tasks = [
            task
            for task in self.tasks
            if task.observation_state == ObservationStateValues.FALSE
        ]
        collisions = []
        thresholds = []
        for task in violated_tasks:
            collision = context.self_collision_manager.last_closest_contacts[
                task.collision_group_a, task.collision_group_b
            ][0]
            collisions.append(collision)
            thresholds.append(task.violated_distance.evaluate()[0])
        raise CollisionViolatedError(
            violated_collisions=collisions, thresholds=thresholds
        )


@dataclass(eq=False, repr=False)
class SelfCollisionAvoidance(Goal):
    """
    A goal combining a SelfCollisionDistanceMonitor and a SelfCollisionAvoidanceTask.
    One pair will be added for all collision groups of the robot.
    The task will only be active if the monitor detects that a collision is close.
    """

    robot: AbstractRobot = field(kw_only=True, default=None)
    """
    The robot for which the collision avoidance goal is defined.
    """
    max_velocity: float = field(default=0.2, kw_only=True)
    """
    The maximum velocity for the collision avoidance task.
    """
    self_collision_manager: SelfCollisionVariableManager = field(init=False)
    """
    Reference to the self collision variable manager shared by other self collision avoidance nodes.
    """
    cancel_if_collision_violated: bool = field(default=True, kw_only=True)
    """
    If True, the motion will be canceled if a collision is violated.
    """

    def create_self_collision_matrix(
        self, context: MotionStatechartContext
    ) -> CollisionMatrix:
        """
        Creates a collision matrix that contains all body combinations except those that are always filtered.
        We need this because we don't know how the collision matrix might change during the motion
        """
        collision_matrix = CollisionMatrix()
        avoid_self_collisions = AvoidSelfCollisions(robot=self.robot)
        avoid_self_collisions.update(context.world)
        avoid_self_collisions.apply_to_collision_matrix(collision_matrix)
        for ignore_collision_rule in context.collision_manager.ignore_collision_rules:
            ignore_collision_rule.apply_to_collision_matrix(collision_matrix)
        return collision_matrix

    def expand(self, context: MotionStatechartContext) -> None:
        if self.robot is None:
            robots = context.world.get_semantic_annotations_by_type(AbstractRobot)
            if len(robots) != 1:
                raise NodeInitializationError(
                    self, f"Expected exactly one robot, got {len(robots)}"
                )
            self.robot = robots[0]

        self.self_collision_manager = context.self_collision_manager
        collision_matrix = self.create_self_collision_matrix(context)

        kinematic_structure_entities = self.robot.kinematic_structure_entities

        tasks = []

        for group_a, group_b in combinations(
            self.self_collision_manager.collision_groups, 2
        ):
            if (
                group_a.root not in kinematic_structure_entities
                or group_b.root not in kinematic_structure_entities
            ):
                # this is no self collision
                continue
            if not collision_matrix.is_collision_groups_combination_checked(
                group_a, group_b
            ):
                # skip because this self collision is never checked
                continue
            self.self_collision_manager.register_groups_of_body_combination(
                group_a.root, group_b.root
            )
            (group_a, group_b) = self.self_collision_manager.body_pair_to_group_pair(
                group_a.root, group_b.root
            )

            distance_monitor = _SelfCollisionHasData(
                name=f"{self.name}/{group_a.root.name.name, group_b.root.name.name}/monitor",
                collision_group_a=group_a,
                collision_group_b=group_b,
                self_collision_manager=self.self_collision_manager,
            )
            self.add_node(distance_monitor)

            task = _SelfCollisionAvoidanceTask(
                name=f"{self.name}/{group_a.root.name.name, group_b.root.name.name}/task",
                collision_group_a=group_a,
                collision_group_b=group_b,
                max_velocity=self.max_velocity,
                self_collision_manager=self.self_collision_manager,
            )
            self.add_node(task)
            task.pause_condition = distance_monitor.observation_variable
            tasks.append(task)

        if self.cancel_if_collision_violated:
            self.add_node(
                _CancelBecauseSelfCollisionViolated(
                    name="self collision violated", tasks=tasks
                )
            )


@dataclass(eq=False, repr=False)
class SelfCollisionDistanceMonitor(MotionStatechartNode):
    """
    Monitors the distance to the closest external object for the group of a body.
    Turns True if the distance falls below a given threshold.
    .. note:: the input bodies are only used to look up the collision groups.
    """

    body_a: Body = field(kw_only=True)
    """First robot body to monitor."""
    body_b: Body = field(kw_only=True)
    """Second robot body to monitor."""
    threshold: float = field(kw_only=True)
    """Distance threshold in meters."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        manager = context.self_collision_manager

        manager.register_groups_of_body_combination(self.body_a, self.body_b)
        (group_a, group_b) = manager.body_pair_to_group_pair(self.body_a, self.body_b)

        distance_symbol = manager.get_contact_distance_symbol(group_a, group_b)

        return NodeArtifacts(observation=distance_symbol < self.threshold)
