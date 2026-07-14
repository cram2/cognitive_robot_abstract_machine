from typing import Optional, TYPE_CHECKING

from giskardpy.model.world_config import WorldWithFixedRobot
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.xarm5 import XArm5

if TYPE_CHECKING:
    pass


class WorldWithXArm5Config(WorldWithFixedRobot):
    """Minimal XArm5 world config analogous to WorldWithTracyConfig.

    - Fixed-base robot (no drive joint)
    - Accepts URDF via argument
    """

    def __init__(self, urdf: Optional[str] = None):
        super().__init__(urdf=urdf, root_name=PrefixedName("map"), urdf_view=XArm5)

    def setup_world(self, robot_name: Optional[str] = None) -> None:
        super().setup_world()
        self.robot = self.world.get_semantic_annotations_by_type(XArm5)[0]


class XArm5StandAloneRobotInterfaceConfig(StandAloneRobotInterfaceConfig):
    def __init__(self):
        super().__init__(
            [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
            ]
        )
