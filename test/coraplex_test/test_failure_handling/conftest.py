from copy import deepcopy
from dataclasses import dataclass, field

import pytest
from krrood.entity_query_language.factories import a
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from typing_extensions import Optional, Type

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.failures import MotionDidNotFinish
from coraplex.execution_environment import simulated_robot
from coraplex.failure_handling.failure_handler import FailureHandler
from coraplex.failure_handling.failure_handling_strategy import (
    FailureHandlingStrategy,
    FailureResolution,
    Propagate,
    RetryNode,
)
from coraplex.language import CodeNode, SequentialNode
from coraplex.plans.factories import code, execute_single, sequential
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan_node import PlanNode, UnderspecifiedNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction

# %% stand-in nodes


@dataclass(eq=False)
class FailingLeaf(CodeNode):
    """
    A leaf that fails a configurable number of times before it succeeds.

    Every failure is attributed to the leaf itself, making the leaf a drop-in for any
    work that fails and later recovers.
    """

    failure_type: Type[PlanFailure] = field(kw_only=True, default=PlanFailure)
    """
    The type of failure the leaf raises.
    """

    remaining_failures: Optional[int] = field(kw_only=True, default=None)
    """
    How many failures the leaf still raises; None makes it fail on every run.
    """

    executions: int = field(init=False, default=0)
    """
    How often the leaf has run.
    """

    def notify(self):
        self.executions += 1
        if self.remaining_failures is None:
            raise self.failure_type(node=self)
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise self.failure_type(node=self)
        return super().notify()


# %% stub handlers and strategies


@dataclass
class ConsultationCountingHandler(FailureHandler):
    """
    Counts how often the performing nodes consult the handler.
    """

    consultations: int = field(default=0, init=False)
    """
    How many failures this handler was consulted for.
    """

    def handle(self, failure: PlanFailure) -> FailureResolution:
        self.consultations += 1
        return super().handle(failure)


@dataclass
class SequenceRetryingStrategy(FailureHandlingStrategy):
    """
    Retries the nearest enclosing sequence, which never runs in a perform frame of its
    own.
    """

    def resolve(self, failure: PlanFailure) -> FailureResolution:
        for ancestor in failure.node.path:
            if isinstance(ancestor, SequentialNode):
                return RetryNode(failure=failure, target_node=ancestor)
        return Propagate(failure=failure)


# %% plan-construction helpers


def context_with(handler: FailureHandler) -> Context:
    """
    :param handler: The failure handler the plan under test should consult.
    :return: A world-free context carrying the handler.
    """
    return Context(world=None, robot=None, failure_handler=handler)


def child_of(parent: PlanNode) -> CodeNode:
    """
    :param parent: The node to hang a fresh leaf under.
    :return: The new leaf.
    """
    child = CodeNode(code=lambda: None)
    parent.add_child(child)
    return child


def sequence_over_a_leaf(context: Context) -> tuple[SequentialNode, CodeNode]:
    """
    :param context: The context the plan is built in.
    :return: A sequence and the leaf below it, neither of which the other performs in a
        frame of its own.
    """
    leaf = CodeNode(code=lambda: None)
    root = sequential([leaf], context)
    return root, leaf


# %% world-free node fixtures


@pytest.fixture
def code_node() -> CodeNode:
    return code(lambda: None)


@pytest.fixture
def underspecified_node() -> UnderspecifiedNode:
    return execute_single(a(NavigateAction)(target_location=Pose()))


# %% sabotaged manipulation scenario


def milk_pick_up(world, view) -> PickUpAction:
    """
    :param world: The apartment world holding the milk.
    :param view: The robot view whose left end effector grasps.
    :return: The pick-up of the milk every manipulation scenario in this package starts
        from.
    """
    return PickUpAction(
        target_object=world.get_semantic_annotations_by_type(Milk)[0],
        arm=Arms.LEFT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.end_effector,
        ),
    )


def move_the_robot_out_of_reach(view) -> None:
    """
    Drive the robot away from the objects, so nothing it could act on is reachable.
    """
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.0, 2, 0
    )


@dataclass
class PerformedMotionFailure:
    """
    The outcome of performing one sabotaged pick-up, shared by every test that inspects
    the attributed failure.
    """

    root: SequentialNode
    """
    The plan root that was performed.
    """

    failure: MotionDidNotFinish
    """
    The failure the sabotaged pick-up raised.
    """

    context: Context
    """
    The context the plan performed in.
    """


@pytest.fixture(scope="module")
def attributed_motion_failure(pr2_apartment_world) -> PerformedMotionFailure:
    """
    Perform one pick-up whose motions cannot succeed, because the robot is placed out of
    reach of the object it grasps.
    """
    world = deepcopy(pr2_apartment_world)
    view = world.get_semantic_annotations_by_type(PR2)[0]
    context = Context(world, view)
    pick_up = milk_pick_up(world, view)
    move_the_robot_out_of_reach(view)
    context.evaluate_conditions = False
    root = sequential([pick_up], context)

    with pytest.raises(MotionDidNotFinish) as raised:
        with simulated_robot:
            root.perform()

    return PerformedMotionFailure(root=root, failure=raised.value, context=context)
