"""
The PR2 picks up the milk and recovers from two failures on the way, one with each way of
handling a failure.

- Its gripper is closed when the plan needs it open. A plan transformation inserts the
  missing step into the running plan, and the step that failed runs again after it.
- Its grasp fails because a reading of the gripper joints is lost. The whole pick-up runs
  again, up to a limited number of attempts.

Runs in kinematic simulation against the apartment, so nothing on the network is needed.
Lower :attr:`FailureHandlingDemonstration.maximum_attempts` below
:attr:`FailureHandlingDemonstration.lost_readings` to see the plan give up.
"""

import os
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import groupby

from coraplex.datastructures.enums import InsertionPosition
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.plans.plan_transformation import (
    PlanRewriting,
    RecoveryByInsertion,
    SequentialInsertion,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import GraspingAction, PickUpAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.data_types import ObservationStateValues
from cramph.failure_handling import HandledOnFailure, RetryFailedNode
from cramph.node import FailsOnObservingFalse, StatechartNode, SucceedsOnObservingTrue
from cramph.statechart import Statechart
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World

# %% what the scene is built from


class SceneFile(StrEnum):
    """
    The files the demonstration builds its scene from, under the coraplex resources.
    """

    APARTMENT = os.path.join("worlds", "apartment.urdf")
    MILK = os.path.join("objects", "milk.stl")

    @property
    def path(self) -> str:
        """
        :return: Where the file is read from.
        """
        return os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", self.value
        )


class ApartmentBody(StrEnum):
    """
    The apartment's own bodies the demonstration reads.
    """

    ROOT = "apartment_root"
    """
    The apartment's root, which marks whether the scene was already populated.
    """


# %% nodes that can fail


@dataclass(eq=False, repr=False, kw_only=True)
class JointReading(SucceedsOnObservingTrue, FailsOnObservingFalse, StatechartNode):
    """
    Reads the joints of the robot once it runs, and succeeds if they stand at
    :attr:`joint_state` or fails otherwise.
    """

    joint_state: JointState
    """
    The joint positions this node expects to read.
    """

    tolerance: float = 0.01
    """
    How far a joint may stand from its expected position and still count as being there.
    """

    def on_tick(self) -> ObservationStateValues:
        joints_at_their_positions = all(
            abs(connection.position - target_value) < self.tolerance
            for connection, target_value in zip(
                self.joint_state.connections, self.joint_state.target_values
            )
        )
        if joints_at_their_positions:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE


@dataclass(eq=False, repr=False, kw_only=True)
class UnreliableJointReading(JointReading):
    """
    A :class:`JointReading` whose first :attr:`lost_readings` runs fail, whatever the
    joints stand at, as they do when a reading never arrives.
    """

    lost_readings: int
    """
    How many runs fail before the joints are read.
    """

    _runs: int = field(default=0, init=False, repr=False)
    """
    How often this node started.
    """

    def on_start(self) -> None:
        self._runs += 1

    def on_tick(self) -> ObservationStateValues:
        if self._runs <= self.lost_readings:
            return ObservationStateValues.FALSE
        return super().on_tick()


# %% plan transformations


@dataclass
class ConfirmGraspAfterClosing(SequentialInsertion[GraspingAction]):
    """
    Reads the closed gripper right after a grasp closed it, unreliably.

    A pick-up moves the object under the gripper only after its grasp, so a failed
    reading leaves the object where it stands and the pick-up can simply run again.
    """

    lost_readings: int
    """
    How many readings are lost before one arrives.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.AFTER

    def is_applicable(self, plan_node: GraspingAction) -> bool:
        return True

    def anchor(self, plan_node: GraspingAction) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, plan_node: GraspingAction) -> list[StatechartNode]:
        return [
            UnreliableJointReading(
                name="grasp reading",
                joint_state=plan_node.arm.end_effector.default_configuration(
                    GripperState.CLOSE
                ).joint_state,
                lost_readings=self.lost_readings,
            )
        ]


@dataclass
class OpenGripperBeforeReading(SequentialInsertion[JointReading]):
    """
    Opens a gripper before the reading that expects it to be open.
    """

    end_effector: EndEffector
    """
    The end effector whose gripper is opened.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: JointReading) -> bool:
        return True

    def anchor(self, plan_node: JointReading) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, plan_node: JointReading) -> list[StatechartNode]:
        return [
            SetGripperAction(
                name="open gripper",
                configuration=self.end_effector.default_configuration(
                    GripperState.OPEN
                ),
            )
        ]


# %% the demonstration


@dataclass
class FailureHandlingDemonstration(RobotDemonstration):
    """
    The PR2 walks up to the counter and picks up the milk, recovering from the failures it
    meets on the way.
    """

    ros_node_name: str = "failure_handling_demo_node"

    event_segmentation: bool = False

    lost_readings: int = 2
    """
    How many readings of the closed gripper are lost, each of which fails the pick-up.
    """

    maximum_attempts: int = 2
    """
    How often the pick-up is run again after it failed.
    """

    robot_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            1.1, 2.5, 0
        )
    )
    """
    Where the PR2 stands before the plan begins.
    """

    milk_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.0345
        )
    )
    """
    Where the milk stands on the counter.
    """

    counter_stand_x: float = 1.5
    """
    Where along the x-axis of the world the PR2 stands to pick up the milk.
    """

    counter_stand_y: float = 2.4
    """
    Where along the y-axis of the world the PR2 stands to pick up the milk.
    """

    plan: Sequence | None = field(default=None, init=False)
    """
    The plan performed, None until it was built.
    """

    statechart: Statechart | None = field(default=None, init=False)
    """
    The statechart performing :attr:`plan`, None until it was built.
    """

    def scene_specification(self) -> WorldSpecification:
        """
        :return: The apartment.
        """
        return WorldSpecification.from_urdf(SceneFile.APARTMENT.path)

    def robot_specification(self) -> RobotSpecification:
        """
        :return: The PR2, placed at :attr:`robot_start` through its odom.
        """
        return RobotSpecification(
            semantic_annotation_type=self.used_robot, localization_pose=self.robot_start
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(ApartmentBody.ROOT)

    def populate_scene(self, world: World) -> None:
        """
        Put the milk on the counter.
        """
        name = Milk.__name__
        SemanticAnnotationWithRootSpecification(
            name=name,
            semantic_annotation_type=Milk,
            root_specification=BodySpecification.mesh(name, SceneFile.MILK.path),
        ).spawn(world, parent=None, parent_T_self=self.milk_start)

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [
            RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0]),
            StatementGrounding(),
            PlanRewriting(
                transformations=[
                    ConfirmGraspAfterClosing(lost_readings=self.lost_readings)
                ]
            ),
        ]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Walk up to the counter with a closed gripper, make sure the gripper is open, and
        pick up the milk.
        """
        world = context.world
        robot = context.require_extension(RobotAccess).robot
        arm = robot.right_arm
        end_effector = arm.end_effector
        milk = world.get_semantic_annotations_by_type(Milk)[0]
        self.plan = Sequence(
            nodes=[
                ParkArmsAction(arms=robot.all_arms),
                MoveTorsoAction(torso_state=TorsoState.HIGH),
                NavigateAction(
                    target_location=Pose.from_xyz_rpy(
                        self.counter_stand_x,
                        self.counter_stand_y,
                        0.0,
                        reference_frame=world.root,
                    )
                ),
                SetGripperAction(
                    name="close gripper",
                    configuration=end_effector.default_configuration(
                        GripperState.CLOSE
                    ),
                ),
                HandledOnFailure(
                    name="gripper is open",
                    handled_node=JointReading(
                        name="gripper reading",
                        joint_state=end_effector.default_configuration(
                            GripperState.OPEN
                        ).joint_state,
                    ),
                    failure_handling_strategy=RecoveryByInsertion(
                        maximum_attempts=1,
                        transformation=OpenGripperBeforeReading(
                            end_effector=end_effector
                        ),
                    ),
                ),
                HandledOnFailure(
                    name="milk is picked up",
                    handled_node=PickUpAction(
                        name="pick up milk",
                        grasp=milk.grasp_candidates()[0],
                        arm=arm,
                    ),
                    failure_handling_strategy=RetryFailedNode(
                        maximum_attempts=self.maximum_attempts
                    ),
                ),
            ],
        )
        self.statechart = Statechart(context=context)
        self.statechart.add_node(self.plan)
        return self.statechart

    def print_what_happened(self) -> None:
        """
        Print the steps the plan ended up with, including the one that joined it while it
        ran, and how each node that handled or caused a failure went through its life
        cycle.
        """
        print("The plan ended up with these steps:")
        for step in self.plan.nodes:
            print(f"  {step.name}: {step.life_cycle_state.name}")
        print("Life cycle of the nodes that failed and recovered:")
        for node in [
            *self.statechart.get_nodes_by_type(JointReading),
            *self.statechart.get_nodes_by_type(PickUpAction),
        ]:
            history = self.statechart.state_history.get_life_cycle_history_of_node(node)
            changes = [life_cycle.name for life_cycle, _ in groupby(history)]
            print(f"  {node.name}: {' -> '.join(changes)}")


def main() -> None:
    """
    Run the demonstration and print what happened.
    """
    demonstration = FailureHandlingDemonstration(used_robot=PR2)
    world = demonstration.run()
    demonstration.print_what_happened()
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    print(f"The milk now hangs below {milk.root.parent_connection.parent.name}.")


if __name__ == "__main__":
    main()
