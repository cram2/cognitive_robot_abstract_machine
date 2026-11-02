from dataclasses import dataclass, field

import pytest
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.hsr.configs import (
    WorldWithHSRConfig,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.middleware.ros2.utils.utils_for_tests import GiskardTester
from cramph.composites import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.qp_controller_config import QPControllerConfig
from numpy import pi
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@pytest.fixture()
def default_joint_state(default_hsr_pose: dict[str, float]) -> dict[str, float]:
    return default_hsr_pose


@dataclass
class HSRTester(GiskardTester):
    base_footprint: KinematicStructureEntity = field(init=False)
    map: KinematicStructureEntity = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.base_footprint = self.api.world.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        self.map = self.api.world.root

    def setup_giskard(self) -> Giskard:
        robot_desc = load_xacro("package://hsr_description/robots/hsrb4s.urdf.xacro")
        return Giskard(
            world_config=WorldWithHSRConfig(urdf=robot_desc),
            robot_interface_config=StandAloneRobotInterfaceConfig(),
            server_config=GiskardServerConfig(
                debug_mode=True,
                plot_gantt_chart=True,
                plot_trajectory=True,
            ),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )


@pytest.fixture()
def robot():
    c = HSRTester()
    try:
        yield c
    finally:
        print("tear down")
        c.close()


# %% motions through the action server


def test_move_base(giskard: HSRTester) -> None:
    statechart = Statechart(context=giskard.api.context)
    statechart.add_node(
        node := Sequence(
            nodes=[
                SetOdometry(
                    base_pose=HomogeneousTransformationMatrix.from_xyz_axis_angle(
                        x=1.0,
                        y=1.0,
                        axis=Vector3.Z(),
                        angle=pi / 3,
                        reference_frame=giskard.map,
                    ),
                ),
                CartesianPose(
                    root_link=giskard.default_root,
                    tip_link=giskard.base_footprint,
                    goal_pose=Pose.from_xyz_axis_angle(
                        x=1.0,
                        axis=Vector3.Z(),
                        angle=pi,
                        reference_frame=giskard.map,
                    ),
                ),
            ]
        )
    )
    statechart.add_node(EndMotion.when_true(node))
    giskard.api.compile(statechart)
    giskard.api.execute()
