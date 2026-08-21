"""
Panda parks its arm above a ground plane holding two cubes, in MuJoCo.

Runs entirely in a local MuJoCo simulation built from the Franka Emika Panda MJCF scene
resolved through the ``iai_franka_panda_description`` ROS package -- no controller or
perception pipeline needed.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

from ament_index_python.packages import get_package_share_directory
from typing_extensions import ClassVar

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoBody, MujocoSim
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

CUBE_SIZE = 0.05
"""
Edge length of both cubes, in meters.
"""

CUBE_TO_PICK_NAME = "cube_to_pick"
"""
Name of the cube the plan acts on, which also marks whether the scene was already
spawned.
"""


@dataclass
class PandaSimpleDemo(RobotDemonstration):
    """
    Panda parks its arm above a ground plane holding two cubes.
    """

    ros_node_name: ClassVar[str] = "panda_demo_node"

    multi_sim: MujocoSim | None = field(init=False, default=None)
    """
    The MuJoCo simulation backing this run, started once the scene is fully populated.
    """

    def build_simulated_world(self) -> World:
        """
        Parse the Panda MJCF scene from the ``iai_franka_panda_description`` package.
        """
        scene_path = (
            Path(get_package_share_directory("iai_franka_panda_description"))
            / "mjcf"
            / "panda.xml"
        )
        world = MJCFParser(str(scene_path)).parse()
        Panda.from_world(world)
        return world

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(CUBE_TO_PICK_NAME)

    def populate_scene(self, world: World) -> None:
        """
        Add a ground plane and two cubes, then start the MuJoCo simulation.

        The simulation is built from the world's state at construction time, so it can
        only start once every body the scene needs is already in the world.
        """
        with world.modify_world():
            ground_plane = Body(name=PrefixedName("ground_plane"))
            ground_plane_geometry = ShapeCollection(
                [
                    Box(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            reference_frame=ground_plane
                        ),
                        scale=Scale(2.0, 2.0, 0.02),
                        color=Color(0.6, 0.6, 0.6, 1.0),
                    )
                ],
                reference_frame=ground_plane,
            )
            ground_plane.collision, ground_plane.visual = (
                ground_plane_geometry,
                ground_plane_geometry,
            )
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=ground_plane,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=-0.01, reference_frame=world.root
                    ),
                )
            )

            cube_bottom = Body(name=PrefixedName("cube_bottom"))
            cube_bottom_geometry = ShapeCollection(
                [
                    Box(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            reference_frame=cube_bottom
                        ),
                        scale=Scale(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                        color=Color(0.9, 0.3, 0.3, 1.0),
                    )
                ],
                reference_frame=cube_bottom,
            )
            cube_bottom.collision, cube_bottom.visual = (
                cube_bottom_geometry,
                cube_bottom_geometry,
            )
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=world,
                    parent=world.root,
                    child=cube_bottom,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.40, y=0.10, z=0.06, reference_frame=world.root
                    ),
                )
            )

            cube_to_pick = Body(name=PrefixedName(CUBE_TO_PICK_NAME))
            cube_to_pick_geometry = ShapeCollection(
                [
                    Box(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            reference_frame=cube_to_pick
                        ),
                        scale=Scale(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                        color=Color(0.3, 0.9, 0.3, 1.0),
                    )
                ],
                reference_frame=cube_to_pick,
            )
            cube_to_pick.collision, cube_to_pick.visual = (
                cube_to_pick_geometry,
                cube_to_pick_geometry,
            )
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=world,
                    parent=world.root,
                    child=cube_to_pick,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.40, y=-0.14, z=0.06, reference_frame=world.root
                    ),
                )
            )

        panda = world.get_semantic_annotations_by_type(self.used_robot)[0]
        arm = panda.get_arms()[0]
        for connection in arm.active_connections:
            connection.child.simulator_additional_properties.append(
                MujocoBody(gravitation_compensation_factor=1.0)
            )

        headless = os.environ.get("CI", "false").lower() == "true"
        self.multi_sim = MujocoSim(world=world, headless=headless, step_size=1e-3)
        self.multi_sim.start_simulation()

    def build_context(self, world: World) -> Context:
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            evaluate_conditions=False,
        )

    def build_plan(self, context: Context) -> PlanNode:
        return sequential([ParkArmsAction(Arms.BOTH)], context=context)

    def tear_down(self) -> None:
        """
        Let the simulation settle for a moment before stopping it, so a non-headless
        viewer sees the final pose rather than the window closing mid-motion.
        """
        if self.multi_sim is not None:
            time.sleep(2.0)
            self.multi_sim.stop_simulation()
        super().tear_down()


def main(execution_type: ExecutionType = ExecutionType.SIMULATED) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot or simulate it.
    """
    PandaSimpleDemo(
        used_robot=Panda,
        execution_type=execution_type,
        real_time_factor=1.0,
        prediction_horizon=20,
    ).run()


if __name__ == "__main__":
    main()
