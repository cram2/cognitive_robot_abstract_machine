from dataclasses import dataclass, field

from giskardpy.model.world_config import WorldWithMobileRobot
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2


@dataclass
class WorldWithPR2Config(WorldWithMobileRobot):
    """
    A world containing only the PR2, which drives on an omnidirectional base.
    """

    odometry_body_name: PrefixedName = PrefixedName("odom_combined")
    """
    Name of the body the robot drives relative to.
    """

    robot_class: type[PR2] = field(kw_only=True, default=PR2, init=False)
    """
    The robot class that is applied to the parsed description.
    """
