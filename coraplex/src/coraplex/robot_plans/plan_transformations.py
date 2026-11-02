from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Generic

from coraplex.datastructures.enums import (
    DetectionTechnique,
    InsertionPosition,
    PerceptionSource,
)
from coraplex.exceptions import (
    GraspHasNoClosingGoal,
    ReachHasNoFinalApproach,
)
from coraplex.locations.locations import ReachabilityLocation
from coraplex.querying.grasp_closeness import GraspsByCloseness
from coraplex.plans.plan_transformation import (
    ConfigurationTransformation,
    SequentialInsertion,
    MatchedType,
    PlanTransformation,
)
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndOpenAction,
    MoveAndPickUpAction,
    PickAndPlaceAction,
)
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import (
    DrivesBase,
    LookAtAction,
    NavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import (
    GraspingAction,
    PickUpAction,
    ReachAction,
)
from coraplex.robot_plans.actions.core.robot_body import (
    MoveFingersToWidthAction,
    ParkArmsAction,
)
from coraplex.robot_plans.mixins import ReattachesBranches
from cramph.composites import Sequence, StatementNode
from cramph.node import StatechartNode
from giskardpy.motion_statechart.goals.base_pose_goals import HolonomicBaseGoal
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from giskardpy.motion_statechart.goals.gripper import MoveFingers
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianTask
from krrood.entity_query_language.core.mapped_variable import FlatVariable
from krrood.entity_query_language.core.variable import InstantiatedVariable, Variable
from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.spatial_types import Pose
from coraplex.plans.context_extensions import RobotAccess
from cramph.context import StatechartContext

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% perceiving before a grasp


@dataclass
class DetectBeforeGrasp(SequentialInsertion[ReachAction]):
    """
    Looks at the object and detects it before a reach makes its final approach, so that
    the approach acts on a freshly perceived pose instead of the one the world holds.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: ReachAction) -> bool:
        return True

    def final_approach(self, plan_node: ReachAction) -> StatechartNode:
        """
        :param plan_node: The reach
        :raises ReachHasNoFinalApproach: If no step of the reach moves its tool center
            point, which it has none of before it is expanded.
        :return: The last step of a sequence below the reach moving the tool center
            point of its arm, which brings the gripper onto the object, whatever a
            transformation put after it or around the sequence.
        """
        steps = [
            node
            for node in plan_node.descendants
            if isinstance(node.parent_node, Sequence)
        ]
        for step in reversed(steps):
            if self._moves_the_tool_center_point(step, plan_node.arm):
                return step
        raise ReachHasNoFinalApproach(plan_node)

    @staticmethod
    def _moves_the_tool_center_point(step: StatechartNode, arm: Arm) -> bool:
        """
        :return: Whether a Cartesian task in `step` moves the tool center point of
            `arm`.
        """
        return any(
            isinstance(node, CartesianTask)
            and node.tip_link is arm.end_effector.tool_frame
            for node in [step, *step.descendants]
        )

    def anchor(self, plan_node: ReachAction) -> StatechartNode:
        return self.final_approach(plan_node)

    def nodes_to_insert(self, plan_node: ReachAction) -> list[StatechartNode]:
        approached_pose = plan_node.grasp_pose_sequence(
            plan_node.grasp.grasp_pose, plan_node.arm.end_effector, plan_node.grasp
        ).grasp
        return [
            LookAtAction(target=approached_pose),
            DetectAction(
                technique=DetectionTechnique.TYPES,
                object_sem_annotation=type(plan_node.grasp.graspable),
                accept_first_if_multiple=True,
            ),
        ]


# %% opening what the object lies in


@dataclass
class DrawerOpening(
    SequentialInsertion[MatchedType],
    Generic[MatchedType],
    SubClassSafeGeneric,
    ABC,
):
    """
    The shared part of the rewrites that open the drawers an object lies in.
    """

    minimum_containment_ratio: float = 0.9
    """
    How much of the object has to lie within a drawer for it to count as being in it.
    """

    minimum_opening_ratio: float = 0.9
    """
    How far along its travel a drawer has to stand pulled out to count as open already.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def _closed_drawers_containing(
        self, annotation: HasRootBody, world: World
    ) -> list[Drawer]:
        """
        :param annotation: The object to locate
        :param world: The world the object and the drawers belong to
        :return: The drawers the object lies in that do not already stand open.
        """
        object_body = annotation.root
        return [
            drawer
            for drawer in world.get_semantic_annotations_by_type(Drawer)
            if InsideOf(object_body, drawer.root).compute_containment_ratio()
            > self.minimum_containment_ratio
            and drawer.opening_ratio < self.minimum_opening_ratio
        ]

    def opening_nodes(
        self, drawer: Drawer, arm: Arm, context: StatechartContext
    ) -> list[StatechartNode]:
        """
        :param drawer: The drawer to open
        :param arm: The arm that opens it
        :param context: The context of the statechart the opening is inserted in,
            holding the robot that opens the drawer
        :return: The opening, from a standing pose tried together with it.
        """
        open_the_drawer = MoveAndOpenAction.from_handle_by_closest_grasps(
            drawer.handle, arm, context
        )
        return [StatementNode(statement=open_the_drawer)]

    def anchor(self, plan_node: StatechartNode) -> StatechartNode:
        return plan_node


@dataclass
class OpenDrawerBeforePickUp(DrawerOpening[PickUpAction]):
    """
    Opens the drawers an object lies in before the robot picks it up, so that it reaches
    into an open drawer instead of a closed one.

    Nothing else positions the robot for a pick-up of its own, and opening a drawer
    leaves the robot standing at its handle, so the rewrite ends by parking and driving
    to a pose the object itself can be reached from.
    """

    def is_applicable(self, plan_node: PickUpAction) -> bool:
        return bool(
            self._closed_drawers_containing(plan_node.grasp.graspable, plan_node.world)
        )

    def nodes_to_insert(self, plan_node: PickUpAction) -> list[StatechartNode]:
        graspable = plan_node.grasp.graspable
        nodes = []
        for drawer in self._closed_drawers_containing(graspable, plan_node.world):
            nodes.extend(self.opening_nodes(drawer, plan_node.arm, plan_node.context))
        drive_to_the_object = a(NavigateAction)(
            target_location=variable(
                Pose,
                # A location samples its poses only once the drive is grounded, by
                # which time the drawers this rewrite opens stand open.
                domain=ReachabilityLocation(
                    Pose(reference_frame=graspable.root),
                    plan_node.arm,
                    context=plan_node.context,
                ),
            ),
        )
        nodes.extend(
            [
                ParkArmsAction(arms=plan_node.robot.all_arms),
                StatementNode(statement=drive_to_the_object),
            ]
        )
        return nodes


@dataclass
class PickUpTarget:
    """
    The object a pick-up takes hold of, and the arm it takes hold with.
    """

    graspable: HasGraspCandidates
    """
    The object that is picked up.
    """

    arm: Arm
    """
    The arm that picks it up.
    """


@dataclass
class OpenDrawerBeforeMoveAndPickUp(DrawerOpening[MoveAndPickUpAction]):
    """
    Opens the drawers an object lies in before the robot moves to it and picks it up.

    The opening precedes the whole move-and-pick-up, whose own drive then positions the
    robot at the object. When every candidate of a move-and-pick-up still to be grounded
    picks up the same object with the same arm, the drawer is opened once in front of
    it, so every candidate is grounded and tried with the drawer standing open.
    Otherwise each candidate gets its own opening, inside the sequence it is tried in.
    """

    def matches_node(self, plan_node: StatechartNode) -> bool:
        if isinstance(plan_node, StatementNode):
            return issubclass(plan_node.statement._type_, MoveAndPickUpAction)
        return super().matches_node(plan_node)

    def is_applicable(self, plan_node: MoveAndPickUpAction | StatementNode) -> bool:
        target = self._pick_up_target(plan_node)
        return target is not None and bool(
            self._closed_drawers_containing(target.graspable, plan_node.context.world)
        )

    def nodes_to_insert(
        self, plan_node: MoveAndPickUpAction | StatementNode
    ) -> list[StatechartNode]:
        target = self._pick_up_target(plan_node)
        nodes = []
        for drawer in self._closed_drawers_containing(
            target.graspable, plan_node.context.world
        ):
            nodes.extend(self.opening_nodes(drawer, target.arm, plan_node.context))
        if isinstance(plan_node, StatementNode):
            # The candidates are grounded after the opening, which leaves the arms at
            # the handle, where they would stand in collision at every standing pose.
            robot = plan_node.context.require_extension(RobotAccess).robot
            nodes.append(ParkArmsAction(arms=robot.all_arms))
        return nodes

    def _pick_up_target(
        self, plan_node: MoveAndPickUpAction | StatementNode
    ) -> PickUpTarget | None:
        """
        :param plan_node: A node this matches.
        :return: What the move-and-pick-up picks up and with which arm, or ``None`` if
            it is still to be grounded and its candidates differ in either.
        """
        if isinstance(plan_node, StatementNode):
            return self._pick_up_target_shared_by_candidates_of(plan_node.statement)
        pick_up = plan_node.pick_up
        return PickUpTarget(graspable=pick_up.grasp.graspable, arm=pick_up.arm)

    @staticmethod
    def _pick_up_target_shared_by_candidates_of(
        move_and_pick_up: Match[MoveAndPickUpAction],
    ) -> PickUpTarget | None:
        """
        :param move_and_pick_up: A move-and-pick-up still to be grounded.
        :return: The object and arm every one of its candidates picks up with, or
            ``None`` if they are not the same for all of them, or not known before
            grounding.
        """
        grasp = move_and_pick_up.pick_up.grasp.apply_mapping_on_external_root(
            move_and_pick_up
        )
        arm = move_and_pick_up.pick_up.arm.apply_mapping_on_external_root(
            move_and_pick_up
        )
        if not isinstance(arm, Arm):
            return None
        if (
            isinstance(grasp, FlatVariable)
            and isinstance(grasp._child_, InstantiatedVariable)
            and grasp._child_._type_ is GraspsByCloseness
        ):
            return PickUpTarget(graspable=grasp._child_._kwargs_["graspable"], arm=arm)
        grasps = grasp._domain_ if isinstance(grasp, Variable) else [grasp]
        if not all(isinstance(candidate, GraspCandidate) for candidate in grasps):
            return None
        graspables = {candidate.graspable for candidate in grasps}
        if len(graspables) != 1:
            return None
        [graspable] = graspables
        return PickUpTarget(graspable=graspable, arm=arm)


# %% parking around a pick-and-place


@dataclass
class ParkArmsAroundPickAndPlaceSteps(PlanTransformation[PickAndPlaceAction]):
    """
    Parks the robot's arms before the pick-up of a pick-and-place, between it and the
    place, and after the place, so that neither step starts with the arms wherever the
    one before it left them.
    """

    def is_applicable(self, plan_node: PickAndPlaceAction) -> bool:
        return True

    def apply(self, plan_node: PickAndPlaceAction) -> None:
        for step in list(plan_node.nodes):
            plan_node.insert_before(step, ParkArmsAction(arms=plan_node.robot.all_arms))
        plan_node.insert_after(
            plan_node.nodes[-1], ParkArmsAction(arms=plan_node.robot.all_arms)
        )


# %% parking before anything else


@dataclass
class ParkArmsBeforeFirstAction(SequentialInsertion[Action]):
    """
    Parks the robot's arms in front of the first action of a plan.

    An action that is grounded against the world, such as a drive to a pose the object
    can be reached from, judges the robot in the configuration it is in. Arms left
    wherever an earlier plan dropped them stand in collision at every candidate pose,
    which rules out the whole location before it is ever checked for reachability.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: Action) -> bool:
        return self._first_action_of_the_plan_of(plan_node) is plan_node and (
            not isinstance(plan_node, ParkArmsAction)
        )

    def anchor(self, plan_node: Action) -> StatechartNode:
        """
        :return: The node that holds `plan_node` as a step of the nearest
            :class:`~cramph.composites.Sequence` above it, or the plan itself when no
            sequence runs `plan_node`, so the parking ends before anything running
            alongside `plan_node` starts.
        """
        holders = [plan_node, *plan_node.path]
        return next(
            (holder for holder in holders if isinstance(holder.parent_node, Sequence)),
            holders[-1],
        )

    def nodes_to_insert(self, plan_node: Action) -> list[StatechartNode]:
        return [ParkArmsAction(arms=plan_node.robot.all_arms)]

    @staticmethod
    def _first_action_of_the_plan_of(node: StatechartNode) -> Action | None:
        """
        :param node: A node of a plan.
        :return: The action of that plan that runs first.
        """
        root = node.path[-1] if node.path else node
        return next(
            (
                candidate
                for candidate in [root, *root.descendants]
                if isinstance(candidate, Action)
            ),
            None,
        )


# %% holding an object at its own width


@dataclass
class CloseFingersToObjectWidth(PlanTransformation[GraspingAction]):
    """
    Closes a grasp onto the width of the object it takes hold of, instead of shutting
    the gripper onto it, where the object pushes back.

    Shutting the gripper all the way onto an object that is not centred between the
    fingers wedges it sideways instead of holding it, so a robot whose grasp holds by
    contact alone meets the object at its own width, measured on the object. A gripper
    whose fingers are not modelled like the real ones keeps shutting, since a width
    reached on its model need not hold on the real gripper, and so does a gripper the
    grasp does not move by goals for its fingers.
    """

    squeeze: float = 0.0
    """
    How far past the object's width each finger is sent, in metres, so the fingers press
    into it rather than merely touching it.
    """

    def is_applicable(self, plan_node: GraspingAction) -> bool:
        gripper = plan_node.arm.end_effector
        return (
            isinstance(gripper, HasTwoFingers)
            and gripper.fingers_are_modelled_like_the_real_ones
            and issubclass(plan_node.gripper_goal_type, MoveFingers)
            and plan_node.grasp.graspable.root.has_collision()
        )

    def apply(self, plan_node: GraspingAction) -> None:
        closing_step = self.closing_step(plan_node)
        plan_node.statechart.replace(
            closing_step,
            plan_node.closing_step(
                MoveFingersToWidthAction(
                    gripper=plan_node.arm.end_effector,
                    width=plan_node.grasp.width_along_closing_axis,
                    squeeze=self.squeeze,
                )
            ),
        )

    @staticmethod
    def closing_step(plan_node: GraspingAction) -> StatechartNode:
        """
        :param plan_node: The grasp.
        :raises GraspHasNoClosingGoal: If no goal moving the fingers lies below the
            grasp, which it has none of before it is expanded.
        :return: The last step of the grasp that moves its fingers, which shuts them.
        """
        for step in reversed(plan_node.nodes):
            if any(isinstance(node, MoveFingers) for node in [step, *step.descendants]):
                return step
        raise GraspHasNoClosingGoal(plan_node)


# %% putting the base at a pose where nothing has to drive it there


@dataclass
class WriteTheOdometryInsteadOfDriving(ConfigurationTransformation[DrivesBase]):
    """
    Puts the base at the poses it drives to by writing the odometry, however it would be
    driven there.
    """

    def is_applicable(self, plan_node: DrivesBase) -> bool:
        return True

    def apply(self, plan_node: DrivesBase) -> None:
        plan_node.base_goal_type = SetOdometry


@dataclass
class WriteTheOdometryInsteadOfCommandingAPose(WriteTheOdometryInsteadOfDriving):
    """
    Puts the base at the poses it drives to by writing the odometry where the controller
    would drive it there in any direction, which a world moved kinematically has no
    drive to follow.
    """

    def is_applicable(self, plan_node: DrivesBase) -> bool:
        return issubclass(plan_node.base_goal_type, HolonomicBaseGoal)


# %% detecting with the real robot's perception


@dataclass
class DetectWithRoboKudo(ConfigurationTransformation[DetectAction]):
    """
    Answers a detection with RoboKudo instead of the world model, where the world model
    does not know what is in front of the robot.
    """

    def is_applicable(self, plan_node: DetectAction) -> bool:
        return True

    def apply(self, plan_node: DetectAction) -> None:
        plan_node.perception_source = PerceptionSource.ROBOKUDO


# %% letting contact carry a body


@dataclass
class LetContactCarryTheBodies(ConfigurationTransformation[ReattachesBranches]):
    """
    Leaves the world model alone where an action would move a body under a new parent,
    where a running physics moves the world: contact carries the body there, and putting
    a picked object under the gripper would hold it whether or not the grasp did.
    """

    def is_applicable(self, plan_node: ReattachesBranches) -> bool:
        return True

    def apply(self, plan_node: ReattachesBranches) -> None:
        plan_node.reattaches_in_world_model = False
