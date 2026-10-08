#!/usr/bin/env python3
"""
PR2 Full PyCRAM Demo -- RobotDemonstration via ROS 1/2 Bridge
============================================================

Demonstrates executing PyCRAM action and motion plans on the PR2 using the
RobotDemonstration base class, with communication through the ROS 1/2 bridge.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import ClassVar, Type

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

import coraplex.alternative_motion_mappings.pr2_motion_mapping  # noqa: F401 -- registers PR2 bridge handlers

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pr2_full_pycram_demo")


# %% PR2FullPycramDemonstration


@dataclass
class PR2FullPycramDemonstration(RobotDemonstration):
    """
    Demonstration exercising PyCRAM plans on the PR2.
    """

    used_robot: Type[AbstractRobot] = PR2
    """
    The robot used for the demonstration.
    """

    ros_node_name: ClassVar[str] = "pr2_full_pycram_demo"
    """
    ROS node name registered for real runs.
    """

    execution_type: ExecutionType = ExecutionType.SIMULATED
    """
    Execution mode: simulated or real (via hardware bridge).
    """

    def build_simulated_world(self) -> World:
        """
        Load the PR2 model into the world from the iai_pr2 package.
        """
        logger.info("Loading PR2 description from iai_pr2 package...")
        world = URDFParser.from_file(
            "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
        ).parse()
        logger.info("PR2 world loaded successfully.")
        return world

    def is_scene_populated(self, world: World) -> bool:
        """
        The demonstration uses the robot model directly without extra scene objects.
        """
        return True

    def populate_scene(self, world: World) -> None:
        """
        No additional scene population required.
        """
        pass

    def build_context(self, world: World) -> Context:
        """
        Build the plan context for the PR2 in ``world``.
        """
        return Context(
            world=world,
            robot=PR2.from_world(world),
            ros_node=self.ros_node,
            alternative_motion_mappings=self.alternative_motion_mappings,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Build the sequential torso manipulation plan.
        """
        logger.info("Building PyCRAM torso plan (Torso HIGH -> LOW)...")
        return sequential(
            [
                MoveTorsoAction(TorsoState.HIGH),
                MoveTorsoAction(TorsoState.LOW),
            ],
            context=context,
        )


# %% Entry point


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PR2 Full PyCRAM Demo using RobotDemonstration base class"
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "real"],
        default="sim",
        help="Execution mode: 'sim' for local simulation, 'real' for bridge execution",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of times to run the demonstration plan",
    )
    args = parser.parse_args()

    mode_map = {
        "sim": ExecutionType.SIMULATED,
        "real": ExecutionType.REAL,
    }
    execution_type = mode_map[args.mode]

    logger.info("Starting PR2 Full PyCRAM Demonstration (mode=%s)...", args.mode)

    demo = PR2FullPycramDemonstration(
        used_robot=PR2,
        execution_type=execution_type,
        repetitions=args.repetitions,
    )
    demo.run()

    logger.info("PR2 Full PyCRAM Demonstration finished.")


if __name__ == "__main__":
    main()
