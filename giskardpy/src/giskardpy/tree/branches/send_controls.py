from typing import List

from giskardpy.tree.behaviors.joint_group_vel_controller_publisher import (
    JointGroupVelController,
)
from giskardpy.tree.behaviors.joint_vel_controller_publisher import JointVelController
from giskardpy.tree.behaviors.send_cmd_vel import SendCmdVelTwist
from giskardpy.tree.composites.running_selector import RunningSelector
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    OmniDrive,
)


class SendControls(RunningSelector):
    def __init__(self, name: str = "send controls"):
        super().__init__(name, memory=False)

    def add_joint_velocity_controllers(self, namespaces: List[str]):
        self.add_child(JointVelController(namespaces=namespaces))

    def add_joint_velocity_group_controllers(
        self, cmd_topic: str, connections: List[ActiveConnection1DOF]
    ):
        self.add_child(
            JointGroupVelController(cmd_topic=cmd_topic, connections=connections)
        )

    def add_send_cmd_velocity(self, topic_name: str, joint: OmniDrive = None):
        self.add_child(SendCmdVelTwist(topic_name=topic_name, joint=joint))
