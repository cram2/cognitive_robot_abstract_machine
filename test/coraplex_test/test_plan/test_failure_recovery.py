"""
Tests for recovering from a failure of a plan performed by a simulated robot: running a
real action again, and inserting real actions with a plan transformation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from coraplex.datastructures.enums import InsertionPosition
from coraplex.exceptions import MotionDidNotFinish
from coraplex.plans.plan_transformation import (
    PlanRewriting,
    PlanTransformation,
    RecoveryByInsertion,
    SequentialInsertion,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import GraspingAction, PickUpAction
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from cramph.composites import Sequence
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.exceptions import StatechartAlreadyCompiledError
from cramph.failure_handling import HandledOnFailure, RetryFailedNode
from cramph.node import FailsOnObservingFalse, StatechartNode, SucceedsOnObservingTrue
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.world import World
from ...plan_running import run_plan

# %% mimics


@dataclass(eq=False, repr=False, kw_only=True)
class StepFailingItsFirstRuns(
    SucceedsOnObservingTrue, FailsOnObservingFalse, StatechartNode
):
    """
    A step that fails on its first :attr:`failing_runs` runs and succeeds after that.
    """

    failing_runs: int
    """
    How many runs fail before one succeeds.
    """

    runs: int = field(default=0, init=False)
    """
    How often this step started.
    """

    def on_start(self) -> None:
        self.runs += 1

    def on_tick(self) -> ObservationStateValues:
        if self.runs <= self.failing_runs:
            return ObservationStateValues.FALSE
        return ObservationStateValues.TRUE


@dataclass(kw_only=True)
class GraspFailingItsFirstRuns(SequentialInsertion[GraspingAction]):
    """
    Lets a grasp fail its first runs, right after closing and before the object is moved
    under the gripper.
    """

    failing_runs: int
    """
    How many runs of the grasp fail before one succeeds.
    """

    inserted_steps: list[StepFailingItsFirstRuns] = field(default_factory=list)
    """
    The steps this transformation inserted.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.AFTER

    def is_applicable(self, plan_node: GraspingAction) -> bool:
        return True

    def anchor(self, plan_node: GraspingAction) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, plan_node: GraspingAction) -> list[StatechartNode]:
        step = StepFailingItsFirstRuns(failing_runs=self.failing_runs)
        self.inserted_steps.append(step)
        return [step]


@dataclass(eq=False, repr=False, kw_only=True)
class GripperIsOpen(SucceedsOnObservingTrue, FailsOnObservingFalse, StatechartNode):
    """
    A step that succeeds if a gripper stands at its open configuration and fails
    otherwise.
    """

    end_effector: EndEffector
    """
    The end effector whose gripper is read.
    """

    tolerance: float = 0.01
    """
    How far a joint may stand from its open position and still count as open.
    """

    runs: int = field(default=0, init=False)
    """
    How often this step started.
    """

    def on_start(self) -> None:
        self.runs += 1

    def on_tick(self) -> ObservationStateValues:
        open_state = self.end_effector.default_configuration(
            GripperState.OPEN
        ).joint_state
        is_open = all(
            abs(connection.position - target_value) < self.tolerance
            for connection, target_value in zip(
                open_state.connections, open_state.target_values
            )
        )
        return ObservationStateValues.TRUE if is_open else ObservationStateValues.FALSE


@dataclass(kw_only=True)
class OpenGripperBeforeReadingIt(SequentialInsertion[GripperIsOpen]):
    """
    Opens the gripper a :class:`GripperIsOpen` reads before that step runs.
    """

    applicable: bool = True
    """
    What :meth:`is_applicable` answers.
    """

    gripper_state: GripperState = GripperState.OPEN
    """
    The state the inserted action moves the gripper to.
    """

    inserted_actions: list[SetGripperAction] = field(default_factory=list)
    """
    The actions this transformation inserted.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: GripperIsOpen) -> bool:
        return self.applicable

    def anchor(self, plan_node: GripperIsOpen) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, plan_node: GripperIsOpen) -> list[StatechartNode]:
        action = SetGripperAction(
            configuration=plan_node.end_effector.default_configuration(
                self.gripper_state
            )
        )
        self.inserted_actions.append(action)
        return [action]


@dataclass(kw_only=True)
class TransformationOnADifferentType(SequentialInsertion[ParkArmsAction]):
    """
    A transformation that never matches a :class:`GripperIsOpen`.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: ParkArmsAction) -> bool:
        return True

    def anchor(self, plan_node: ParkArmsAction) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, plan_node: ParkArmsAction) -> list[StatechartNode]:
        return []


@dataclass(kw_only=True)
class TransformationRecordingGripperActions(PlanTransformation[SetGripperAction]):
    """
    Records every gripper action of a plan it is offered, and changes nothing.
    """

    offered_actions: list[SetGripperAction] = field(default_factory=list)
    """
    The gripper actions this transformation was applied to, in order.
    """

    def is_applicable(self, plan_node: SetGripperAction) -> bool:
        return True

    def apply(self, plan_node: SetGripperAction) -> None:
        self.offered_actions.append(plan_node)


@dataclass(kw_only=True)
class CloseGripperBeforeAFailingStep(SequentialInsertion[StepFailingItsFirstRuns]):
    """
    Inserts a gripper action before a :class:`StepFailingItsFirstRuns`.
    """

    end_effector: EndEffector
    """
    The end effector the inserted action moves.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: StepFailingItsFirstRuns) -> bool:
        return True

    def anchor(self, plan_node: StepFailingItsFirstRuns) -> StatechartNode:
        return plan_node

    def nodes_to_insert(
        self, plan_node: StepFailingItsFirstRuns
    ) -> list[StatechartNode]:
        return [
            SetGripperAction(
                configuration=self.end_effector.default_configuration(
                    GripperState.CLOSE
                )
            )
        ]


# %% helpers


def _closed_gripper(end_effector: EndEffector) -> SetGripperAction:
    """
    :return: An action closing the gripper of `end_effector`.
    """
    return SetGripperAction(
        configuration=end_effector.default_configuration(GripperState.CLOSE)
    )


def _pick_up_of_the_milk(
    world: World, robot: PR2, recovery: RetryFailedNode
) -> tuple[Sequence, PickUpAction, Milk]:
    """
    :return: A plan driving to the counter and picking up the milk with the right arm,
        recovering from a failure with `recovery`, its pick-up, and the milk.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up = PickUpAction(grasp=milk.grasp_candidates()[0], arm=robot.right_arm)
    plan = Sequence(
        nodes=[
            ParkArmsAction(arms=robot.all_arms),
            NavigateAction(
                target_location=Pose.from_xyz_rpy(
                    1.5, 2.4, 0.0, reference_frame=world.root
                )
            ),
            HandledOnFailure(handled_node=pick_up, failure_handling_strategy=recovery),
        ]
    )
    return plan, pick_up, milk


# %% running a real action again


def test_a_pick_up_whose_grasp_failed_is_run_again_until_it_holds_the_milk(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    failing_runs = 2
    failing_grasp = GraspFailingItsFirstRuns(failing_runs=failing_runs)
    plan, _, milk = _pick_up_of_the_milk(
        world, robot, RetryFailedNode(maximum_attempts=failing_runs)
    )

    run_plan(
        plan,
        [*extensions, PlanRewriting(transformations=[failing_grasp])],
    )

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
    [inserted_step] = failing_grasp.inserted_steps
    assert inserted_step.runs == failing_runs + 1
    assert milk.root.parent_connection.parent is robot.right_arm.end_effector.tool_frame


def test_a_pick_up_is_given_up_on_and_leaves_the_milk_once_the_attempts_ran_out(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    maximum_attempts = 1
    failing_grasp = GraspFailingItsFirstRuns(failing_runs=maximum_attempts + 1)
    plan, pick_up, milk = _pick_up_of_the_milk(
        world, robot, RetryFailedNode(maximum_attempts=maximum_attempts)
    )
    parent_of_the_milk = milk.root.parent_connection.parent

    with pytest.raises(MotionDidNotFinish):
        run_plan(
            plan,
            [*extensions, PlanRewriting(transformations=[failing_grasp])],
        )

    assert pick_up.life_cycle_state == LifeCycleValues.FAILED
    assert plan.life_cycle_state == LifeCycleValues.FAILED
    assert milk.root.parent_connection.parent is parent_of_the_milk


# %% inserting real actions


def test_an_inserted_action_lets_the_failed_step_succeed(pr2_apartment_context) -> None:
    world, robot, extensions = pr2_apartment_context
    end_effector = robot.right_arm.end_effector
    check = GripperIsOpen(end_effector=end_effector)
    transformation = OpenGripperBeforeReadingIt()
    close = _closed_gripper(end_effector)
    handled = HandledOnFailure(
        handled_node=check,
        failure_handling_strategy=RecoveryByInsertion(
            maximum_attempts=1, transformation=transformation
        ),
    )
    plan = Sequence(nodes=[close, handled])

    run_plan(plan, extensions)

    [opening] = transformation.inserted_actions
    assert plan.nodes == [close, opening, handled]
    assert opening.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert check.runs == 2
    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_inserting_gives_up_once_the_attempts_ran_out(pr2_apartment_context) -> None:
    world, robot, extensions = pr2_apartment_context
    end_effector = robot.right_arm.end_effector
    maximum_attempts = 1
    transformation = OpenGripperBeforeReadingIt(gripper_state=GripperState.CLOSE)
    handled = HandledOnFailure(
        handled_node=GripperIsOpen(end_effector=end_effector),
        failure_handling_strategy=RecoveryByInsertion(
            maximum_attempts=maximum_attempts, transformation=transformation
        ),
    )
    plan = Sequence(nodes=[_closed_gripper(end_effector), handled])

    with pytest.raises(MotionDidNotFinish):
        run_plan(plan, extensions)

    assert len(transformation.inserted_actions) == maximum_attempts
    assert handled.life_cycle_state == LifeCycleValues.FAILED


def test_a_transformation_that_does_not_apply_gives_up(pr2_apartment_context) -> None:
    world, robot, extensions = pr2_apartment_context
    end_effector = robot.right_arm.end_effector
    transformation = OpenGripperBeforeReadingIt(applicable=False)
    handled = HandledOnFailure(
        handled_node=GripperIsOpen(end_effector=end_effector),
        failure_handling_strategy=RecoveryByInsertion(
            maximum_attempts=1, transformation=transformation
        ),
    )
    plan = Sequence(nodes=[_closed_gripper(end_effector), handled])

    with pytest.raises(MotionDidNotFinish):
        run_plan(plan, extensions)

    assert transformation.inserted_actions == []
    assert handled.life_cycle_state == LifeCycleValues.FAILED


def test_a_transformation_that_does_not_match_gives_up(pr2_apartment_context) -> None:
    world, robot, extensions = pr2_apartment_context
    end_effector = robot.right_arm.end_effector
    close = _closed_gripper(end_effector)
    handled = HandledOnFailure(
        handled_node=GripperIsOpen(end_effector=end_effector),
        failure_handling_strategy=RecoveryByInsertion(
            maximum_attempts=1, transformation=TransformationOnADifferentType()
        ),
    )
    plan = Sequence(nodes=[close, handled])

    with pytest.raises(MotionDidNotFinish):
        run_plan(plan, extensions)

    assert plan.nodes == [close, handled]
    assert handled.life_cycle_state == LifeCycleValues.FAILED


def test_an_inserted_action_is_offered_to_the_plan_rewriting(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    end_effector = robot.right_arm.end_effector
    transformation = OpenGripperBeforeReadingIt()
    recording = TransformationRecordingGripperActions()
    close = _closed_gripper(end_effector)
    plan = Sequence(
        nodes=[
            close,
            HandledOnFailure(
                handled_node=GripperIsOpen(end_effector=end_effector),
                failure_handling_strategy=RecoveryByInsertion(
                    maximum_attempts=1, transformation=transformation
                ),
            ),
        ]
    )

    run_plan(plan, [*extensions, PlanRewriting(transformations=[recording])])

    assert recording.offered_actions == [close, *transformation.inserted_actions]


def test_a_step_on_the_top_level_cannot_be_recovered_by_inserting(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    end_effector = robot.right_arm.end_effector
    handled = HandledOnFailure(
        handled_node=StepFailingItsFirstRuns(failing_runs=1),
        failure_handling_strategy=RecoveryByInsertion(
            maximum_attempts=1,
            transformation=CloseGripperBeforeAFailingStep(end_effector=end_effector),
        ),
    )

    with pytest.raises(StatechartAlreadyCompiledError):
        run_plan(handled, extensions)
