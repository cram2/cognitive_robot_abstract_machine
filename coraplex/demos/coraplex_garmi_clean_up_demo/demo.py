"""
GARMI puts a milk carton from the kitchen counter into a drawer.

GARMI picks up the milk from the front with its right arm, grasps the drawer's handle
with its left while the right arm parks, opens the drawer while the right arm lays the
milk into it from above and withdraws straight up out of it, and closes the drawer again
while the right arm parks. Running with :class:`~coraplex.plans.executors.RealRobotPlanExecutor`
drives the actual robot and takes the world from the running world server, merging the
apartment into it unless the server holds it already. The default runs the whole plan in
simulation against a world built from the apartment's MuJoCo scene and GARMI's URDF, so
nothing on the network is needed.

With :attr:`GraspSelection.BY_CLOSENESS`, the grasp on the handle and the grasp on the
milk are left to be chosen while the plan runs, both in one parallel, so they are
grounded together.

Needs the ``iai_garmi_apartment`` and ``garmi_description`` packages built in the
workspace, since the scene and the robot description are read from their share
directories.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum, StrEnum

from ament_index_python.packages import get_package_share_directory

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor, PlanExecutor
from coraplex.querying.grasp_closeness import GraspsByCloseness
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import (
    FixateObjectAction,
    GraspingAction,
    PickUpAction,
)
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from cramph.composites import Parallel, Sequence, StartedWhenTrue
from cramph.context import ContextExtension, StatechartContext
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from cramph.world_modification_nodes import MoveBranch
from giskardpy.motion_statechart.tasks.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.tasks.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from giskardpy.motion_statechart.monitors.feature_monitors import DistanceMonitor
from giskardpy.motion_statechart.monitors.joint_monitors import JointPositionReached
from krrood.entity_query_language.factories import a, flat_variable, variable
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Milk,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from krrood_spatial_math.spatial_types import Point3, Pose
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% what the scene is built from


class SceneFile(StrEnum):
    """
    The files the demonstration builds its scene from.
    """

    APARTMENT = os.path.join("mjcf", "scene-bodies.xml")
    """
    The apartment scene GARMI acts in, in the ``iai_garmi_apartment`` package.
    """

    MILK = os.path.join("objects", "milk.stl")
    """
    The milk's mesh, under the coraplex resources.
    """

    @property
    def path(self) -> str:
        """
        :return: Where the file is read from.
        """
        if self is SceneFile.APARTMENT:
            return os.path.join(
                get_package_share_directory("iai_garmi_apartment"), self.value
            )
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "resources",
            self.value,
        )


class SceneBody(StrEnum):
    """
    The bodies of the scene the demonstration acts on.
    """

    MILK = "milk"
    """
    The milk put into the drawer.
    """

    STORAGE_DRAWER = "drawer_2"
    """
    The drawer the milk is put into.
    """

    APARTMENT = "apartment"
    """
    The apartment's origin, posed in the occupancy map frame the robot localizes in,
    which also marks whether the scene was already populated.
    """


# %% where things stand


class HandDistance(float, Enum):
    """
    Distances, in meters, the hands keep to what they work on.
    """

    HANDLE_CONTACT = 0.2
    """
    How close the left hand has to be to the drawer's handle before it may touch its
    surroundings; further away it avoids them.
    """

    HANDLE_RELEASE_CLEARANCE = 0.1
    """
    How far the left hand backs off the drawer's handle once it has let go of it.
    """

    MILK_ABOVE_DRAWER_BEFORE_PLACING = 0.25
    """
    How far above where it is laid down the milk is carried first, so it comes into the
    drawer from above its walls rather than across them.
    """

    RETRACT_ABOVE_PLACE = 0.15
    """
    How far straight up from where it let go of the milk the right hand withdraws, so it
    is clear of the drawer before the arms park.
    """


@dataclass(frozen=True)
class ScenePlacement:
    """
    A position and an orientation, given as a heading about the vertical and a tilt.
    """

    x: float
    y: float
    z: float
    yaw: float = 0.0
    """
    The heading about the vertical, in radian.
    """

    pitch: float = 0.0
    """
    The tilt about the y-axis, in radian.
    """

    @property
    def transform(self) -> HomogeneousTransformationMatrix:
        """
        :return: The placement as a transform.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            self.x, self.y, self.z, pitch=self.pitch, yaw=self.yaw
        )

    def pose(self, reference_frame: KinematicStructureEntity) -> Pose:
        """
        :param reference_frame: The frame the placement is given in.
        :return: The placement as a pose in `reference_frame`.
        """
        return Pose.from_xyz_rpy(
            self.x,
            self.y,
            self.z,
            pitch=self.pitch,
            yaw=self.yaw,
            reference_frame=reference_frame,
        )


class ScenePose(Enum):
    """
    Where GARMI and the milk start, and where they go.
    """

    GARMI_START = ScenePlacement(-4.8, 1.5, 0, yaw=math.pi)
    """
    Where GARMI starts, in its ``odom`` frame: facing the kitchen from the room.

    The simulated world's ``odom`` frame is the occupancy map frame the apartment is posed
    in, not the apartment's own frame the other placements are relative to.
    """

    GARMI_AT_COUNTER = ScenePlacement(-4.5, 1.75, 0, yaw=math.pi)
    """
    Where GARMI returns to once the milk is in the drawer, in its ``odom`` frame.
    """

    GARMI_AT_DRAWER = ScenePlacement(-5.0, 2.15, 0, yaw=math.pi + math.pi / 4)
    """
    Where GARMI stands while it picks up the milk and opens and closes the drawer, in its
    ``odom`` frame: within reach of both.
    """

    MILK_START = ScenePlacement(-6, 1.87, 1.028)
    """
    Where the milk starts, on the kitchen counter, in the world's root frame.
    """

    MILK_IN_DRAWER = ScenePlacement(-0.545, 6.885, 0.79, pitch=-math.pi / 2)
    """
    Where the milk is put into the drawer once that is open, relative to the apartment:
    lying on its side just above the drawer's bottom plate, so the fingertips, which
    reach below the carton, stay clear of it.

    The milk is grasped from the front, along its x-axis, so tipping that axis up turns
    the grasp into one from above, and the hand lays the milk down from above.
    """


# %% how the run is set up


class GraspSelection(Enum):
    """
    How the grasps on the milk and on the drawer's handle come about.
    """

    GIVEN = "given"
    """
    The plan holds one grasp on each, written into it.
    """

    BY_CLOSENESS = "by closeness"
    """
    The plan holds statements of the grasps, grounded while it runs to the grasps closest
    to where GARMI stands first. The two are in one parallel, so they are grounded
    together.
    """


# %% the demonstration


@dataclass
class GarmiCleanUpDemonstration(RobotDemonstration):
    """
    GARMI puts a milk carton from the kitchen counter into a drawer.
    """

    ros_node_name: str = "garmi_clean_up_demo_node"

    grasp_selection: GraspSelection = GraspSelection.GIVEN
    """
    How the grasps on the milk and on the drawer's handle come about.
    """

    def scene_specification(self) -> WorldSpecification:
        """
        The apartment's MuJoCo scene.

        The scene keeps its collision geometry in a file that is not loaded, so every
        geom has to stand in for it.
        """
        return WorldSpecification.from_mjcf(
            SceneFile.APARTMENT.path, use_visual_as_collision_backup=True
        )

    def robot_specification(self) -> RobotSpecification:
        """
        GARMI at its start.
        """
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            odometry_T_robot_start=ScenePose.GARMI_START.value.transform,
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(
            SceneBody.APARTMENT.value
        )

    def populate_scene(self, world: World) -> None:
        """
        Annotate the apartment's furniture, then add the milk.

        The furniture is annotated first, so the reasoner describes the apartment the
        plan navigates rather than the objects the plan already knows.
        """
        world_reasoner = WorldReasoner(world)
        inferred = world_reasoner.infer_semantic_annotations()
        with world.modify_world():
            world.add_semantic_annotations(inferred)

        # The milk is picked up and carried, and a pose can only be written to a
        # connection that has the degrees of freedom to carry it, so it hangs off a 6DoF
        # connection rather than the default fixed one.
        Milk.get_annotation_specification(
            SceneBody.MILK.value,
            BodySpecification.mesh(
                SceneBody.MILK.value,
                SceneFile.MILK.path,
                parent_T_self=ScenePose.MILK_START.value.transform,
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=world.root)

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [
            RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0]),
        ]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Pick up the milk with the right arm and grasp the drawer's handle with the left,
        open the drawer while the right arm lays the milk into it from above, then close
        the drawer while the right arm parks.
        """
        world = context.world
        robot = context.require_extension(RobotAccess).robot
        left_arm = robot.get_left_arm_if_specified()
        right_arm = robot.get_right_arm_if_specified()
        left_hand = left_arm.end_effector
        storage_drawer = world.get_body_by_name(SceneBody.STORAGE_DRAWER.value)
        apartment = world.get_body_by_name(SceneBody.APARTMENT.value)
        [drawer] = [
            drawer
            for drawer in world.get_semantic_annotations_by_type(Drawer)
            if drawer.root is storage_drawer
        ]
        drawer_mechanism = drawer.handle.root.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        [milk] = world.get_semantic_annotations_by_type(Milk)
        milk_grasp = GraspCandidate(
            graspable=milk,
            grasp_pose=Pose.from_xyz_quaternion(
                quat_z=1, quat_w=0, reference_frame=milk.root
            ),
        )
        handle_grasp = GraspCandidate(
            graspable=drawer.handle,
            grasp_pose=Pose.from_xyz_quaternion(
                quat_y=0.707107,
                quat_z=0.707107,
                quat_w=0,
                reference_frame=drawer.handle.root,
            ),
        )
        milk_in_drawer = ScenePose.MILK_IN_DRAWER.value.pose(apartment)

        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    # Parked before driving, since GARMI may start with its arms
                    # reaching out, and a simulated drive arrives at once.
                    ParkArmsAction(arms=robot.all_arms),
                    NavigateAction(
                        target_location=ScenePose.GARMI_AT_DRAWER.value.pose(world.root)
                    ),
                    *self._take_hold_of_the_milk_and_the_handle(
                        milk_grasp, handle_grasp, left_arm, right_arm
                    ),
                    # Left hand opens the drawer while the right hands drops the milk in the open
                    # drawer. the gripper is kept at the handle of the drawer the whole time
                    Parallel(
                        nodes=[
                            Sequence(
                                nodes=[
                                    ArticulateGraspedMechanism(
                                        tip_link=left_hand.tool_frame,
                                        environment_link=drawer.handle.root,
                                        goal_joint_state=drawer_mechanism.dof.limits.upper.position,
                                        holding_end_effector=left_hand,
                                    ),
                                    SetGripperAction(
                                        configuration=left_hand.default_configuration(
                                            GripperState.OPEN
                                        )
                                    ),
                                ]
                            ),
                            Sequence(
                                nodes=self._lay_the_milk_into_the_drawer(
                                    milk,
                                    milk_grasp,
                                    milk_in_drawer,
                                    storage_drawer,
                                    right_arm,
                                )
                            ),
                            StartedWhenTrue(
                                monitor=JointPositionReached(
                                    name="Drawer open",
                                    connection=drawer_mechanism,
                                    position=drawer_mechanism.dof.limits.upper.position,
                                ),
                                monitored_node=FixateObjectAction(
                                    end_effector=left_hand
                                ),
                            ),
                            UpdateTemporaryCollisionRules.for_end_effector(left_hand),
                        ]
                    ),
                    # Left hand closes the drawer while the right arm is parked
                    Parallel(
                        nodes=[
                            ParkArmsAction(arms=[right_arm]),
                            ArticulateGraspedMechanism(
                                tip_link=left_hand.tool_frame,
                                environment_link=drawer.handle.root,
                                goal_joint_state=drawer_mechanism.dof.limits.lower.position,
                                holding_end_effector=left_hand,
                            ),
                            UpdateTemporaryCollisionRules.for_end_effector(left_hand),
                        ]
                    ),
                    # Back off the handle the way the hand came in.
                    MoveManipulatorAction(
                        target_pose=HasApproachesGraspPoses.pre_grasp_pose(
                            handle_grasp.grasp_pose,
                            left_hand,
                            HandDistance.HANDLE_RELEASE_CLEARANCE.value,
                        ),
                        end_effector=left_hand,
                        allow_gripper_collision=True,
                    ),
                    Parallel(
                        nodes=[
                            ParkArmsAction(arms=robot.all_arms),
                            NavigateAction(
                                target_location=ScenePose.GARMI_AT_COUNTER.value.pose(
                                    world.root
                                )
                            ),
                        ]
                    ),
                ]
            )
        )
        return statechart

    def _take_hold_of_the_milk_and_the_handle(
        self,
        milk_grasp: GraspCandidate,
        handle_grasp: GraspCandidate,
        left_arm: Arm,
        right_arm: Arm,
    ) -> list[StatechartNode]:
        """
        :param milk_grasp: The grasp on the milk, if :attr:`grasp_selection` gives it.
        :param handle_grasp: The grasp on the handle, if :attr:`grasp_selection` gives
            it.
        :param left_arm: The arm that grasps the handle.
        :param right_arm: The arm that picks up the milk.
        :return: The steps after which the right hand holds the milk, the right arm is
            parked and the left hand holds the handle: for given grasps, the pick-up
            first and then the grasp on the handle while the right arm parks; for grasps
            chosen by closeness, the two in one parallel.
        """
        left_hand = left_arm.end_effector
        # Lets the left hand stay on the handle once it holds it, while the right arm may
        # still be moving, but only once it is near the handle.
        handle_contact = StartedWhenTrue(
            monitor=DistanceMonitor(
                name="Left hand near the handle",
                root_link=handle_grasp.graspable.root,
                tip_link=left_hand.tool_frame,
                reference_point=Point3(reference_frame=handle_grasp.graspable.root),
                tip_point=Point3(reference_frame=left_hand.tool_frame),
                lower_limit=0.0,
                upper_limit=HandDistance.HANDLE_CONTACT.value,
            ),
            monitored_node=UpdateTemporaryCollisionRules.for_end_effector(left_hand),
        )
        if self.grasp_selection is GraspSelection.GIVEN:
            return [
                PickUpAction(grasp=milk_grasp, arm=right_arm),
                Parallel(
                    nodes=[
                        GraspingAction(grasp=handle_grasp, arm=left_arm),
                        ParkArmsAction(arms=[right_arm]),
                        handle_contact,
                    ]
                ),
            ]
        return [
            Parallel(
                nodes=[
                    a(GraspingAction)(
                        grasp=self._grasps_by_closeness(
                            handle_grasp.graspable, left_arm
                        ),
                        arm=left_arm,
                    ),
                    Sequence(
                        nodes=[
                            a(PickUpAction)(
                                grasp=self._grasps_by_closeness(
                                    milk_grasp.graspable, right_arm
                                ),
                                arm=right_arm,
                            ),
                            ParkArmsAction(arms=[right_arm]),
                        ]
                    ),
                    handle_contact,
                ]
            )
        ]

    @staticmethod
    def _grasps_by_closeness(graspable: HasGraspCandidates, arm: Arm) -> Match:
        """
        :param graspable: The object to grasp.
        :param arm: The arm that grasps it.
        :return: The grasps `graspable` offers, the closest to where the robot stands
            when they are grounded first.
        """
        # A variable, so the grasps are ranked when the statement is grounded, from
        # where the robot stands then.
        where_the_robot_stands = variable(
            Pose, domain=[Pose(reference_frame=arm._robot.root)]
        )
        return flat_variable(
            GraspsByCloseness(
                graspable=graspable,
                standing_position=where_the_robot_stands,
                number_of_grasps=None,
            )
        )

    def _lay_the_milk_into_the_drawer(
        self,
        milk: Milk,
        milk_grasp: GraspCandidate,
        milk_in_drawer: Pose,
        storage_drawer: KinematicStructureEntity,
        right_arm: Arm,
    ) -> list[StatechartNode | Match]:
        """
        :param milk: The milk the right hand holds.
        :param milk_grasp: The grasp on the milk, if :attr:`grasp_selection` gives it.
        :param milk_in_drawer: Where the milk is laid down.
        :param storage_drawer: The drawer the milk is laid into.
        :param right_arm: The arm that holds the milk.
        :return: The steps laying the milk down, hanging it off the drawer and, for a
            given grasp, withdrawing the right hand straight up out of the drawer.
        """
        # A place leaves the milk under the world; under the drawer it moves along when
        # the drawer closes.
        hang_off_the_drawer = MoveBranch(
            branch_root=milk.root, new_parent=storage_drawer
        )
        if self.grasp_selection is GraspSelection.BY_CLOSENESS:
            # Grounded once it starts, when the milk is in hand, since a place reads the
            # grasp off the gripper holding it.
            return [
                a(PlaceAction)(
                    object_designator=milk,
                    target_location=milk_in_drawer,
                    retreat_distance=HandDistance.MILK_ABOVE_DRAWER_BEFORE_PLACING.value,
                ),
                hang_off_the_drawer,
            ]
        right_hand = right_arm.end_effector
        tool_at_place = right_hand.tool_frame_goal_for_grasp(
            milk_grasp.moved_to(milk_in_drawer)
        )
        tool_above_place = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                z=HandDistance.RETRACT_ABOVE_PLACE.value,
                reference_frame=milk_in_drawer.reference_frame,
            )
            @ tool_at_place.homogeneous_matrix
        ).pose
        return [
            PlaceAction(
                object_designator=milk,
                target_location=milk_in_drawer,
                retreat_distance=HandDistance.MILK_ABOVE_DRAWER_BEFORE_PLACING.value,
            ),
            hang_off_the_drawer,
            MoveManipulatorAction(
                target_pose=tool_above_place,
                end_effector=right_hand,
                allow_gripper_collision=False,
            ),
        ]


def main(
    executor_type: type[PlanExecutor] = KinematicallySimulatedPlanExecutor,
    grasp_selection: GraspSelection = GraspSelection.GIVEN,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it
        drives the real robot or simulates it.
    :param grasp_selection: How the grasps on the milk and the handle come about.
    """
    GarmiCleanUpDemonstration(
        used_robot=Garmi,
        executor_type=executor_type,
        collision_avoidance=True,
        grasp_selection=grasp_selection,
    ).run()


if __name__ == "__main__":
    main()
