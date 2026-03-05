from typing import Optional
from dataclasses import dataclass, field

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Goal, NodeArtifacts
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.goals.unlatch_door import UnlatchDoor
from giskardpy.motion_statechart.monitors.joint_monitors import JointPositionReached
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.context import MotionStatechartContext

from semantic_digital_twin.datastructures.joint_state import JointState

from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


@dataclass(eq=False, repr=False)
class OpenDoorGoal(Goal):
    """
    Use this, if you have grasped a door handle and want to open the door and handle

    :param tip_link: end effector that is grasping the handle
    :param handle_name: link that is grasped by the tip_link
    :param handle_limit: Limit for how far the handle will be opened
    :param hinge_limit: Limit for how far the door-hinge will be opened
    :param root_link: Root-link for alignment of gripper to handle
    :param tip_normal: Gripper axis that is to be aligned with goal_normal
    :param goal_normal: Handle axis that is to be aligned with tip_normal
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    handle_name: KinematicStructureEntity = field(kw_only=True)
    hinge_limit: float = field(kw_only=True)
    handle_limit: Optional[float] = field(default=None, kw_only=True)
    root_link: Optional[KinematicStructureEntity] = field(default=None, kw_only=True)
    tip_normal: Optional[Vector3] = field(default=None, kw_only=True)
    goal_normal: Optional[Vector3] = field(default=None, kw_only=True)

    def expand(self, context: MotionStatechartContext) -> None:
        if self.tip_normal is None:
            self.tip_normal = Vector3(x=1, y=0, z=0, reference_frame=self.tip_link)

        if self.goal_normal is None:
            self.goal_normal = Vector3(x=0, y=0, z=-1, reference_frame=self.handle_name)

        if self.root_link is None:
            self.root_link = context.world.root

        handle_connection = self.handle_name.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        link_id = handle_connection.parent
        door_hinge_connection = link_id.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )

        if door_hinge_connection.dof.has_position_limits():
            min_limit_hinge = door_hinge_connection.dof.limits.lower.position
            max_limit_hinge = door_hinge_connection.dof.limits.upper.position
        else:
            min_limit_hinge = -3.14
            max_limit_hinge = 3.14

        if self.hinge_limit is None:
            limit_hinge = min_limit_hinge
        else:
            limit_hinge = max(min_limit_hinge, self.hinge_limit)

        unlatch_door = UnlatchDoor(
            tip_link=self.tip_link,
            handle_name=self.handle_name,
            handle_limit=self.handle_limit,
        )
        self.add_node(unlatch_door)

        jpl = JointPositionList(
            goal_state=JointState.from_mapping(
                {door_hinge_connection: max_limit_hinge}
            ),
            weight=DefaultWeights.WEIGHT_ABOVE_CA,
            name="DoorHinge",
        )
        jpl.end_condition = unlatch_door.observation_variable
        self.add_node(jpl)

        apl = AlignPlanes(
            root_link=self.root_link,
            tip_link=self.tip_normal.reference_frame,
            goal_normal=self.goal_normal,
            tip_normal=self.tip_normal,
            name="AlignBaseWithDoor",
        )
        apl.start_condition = unlatch_door.observation_variable
        self.add_node(apl)

        open_goal2 = Open(
            tip_link=self.tip_link,
            environment_link=link_id,
            goal_joint_state=limit_hinge,
            name="OpenHinge",
        )
        open_goal2.start_condition = unlatch_door.observation_variable
        self.add_node(open_goal2)

        hinge_state_monitor = JointPositionReached(
            connection=door_hinge_connection, position=limit_hinge, name="HingeMonitor"
        )
        hinge_state_monitor.start_condition = unlatch_door.observation_variable
        self.add_node(hinge_state_monitor)

        self._hinge_state_monitor = hinge_state_monitor

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self._hinge_state_monitor.observation_variable)
