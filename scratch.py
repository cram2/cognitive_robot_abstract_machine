import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from giskardpy.executor import Executor
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from krrood.entity_query_language.conclusion import Add
from krrood.entity_query_language.entity import entity, let, set_of, inference
from krrood.entity_query_language.quantify_entity import an, a
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.procthor.procthor_semantic_annotations import Milk
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.reasoning.predicates_base import CausesOpening
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Container,
    Door,
    Drawer,
    Fridge,
    Handle,
)
from semantic_digital_twin.semantic_annotations.task_effect_motion import (
    Effect,
    OpenEffect,
    ClosedEffect,
    Motion,
)
from semantic_digital_twin.spatial_types.spatial_types import TransformationMatrix
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)


class ContainerDemo:
    """
    Demonstrates finding and opening container-like objects in a semantic world.
    """

    def __init__(self):
        self.resource_dir = os.path.join(
            os.path.dirname(__file__), "pycram", "resources"
        )
        self.world = self._setup_world()
        self._reason_about_world()
        self._setup_ros()

    def _setup_world(self):
        """
        Sets up the world by loading URDFs, STLs, and merging them.
        """
        pr2_sem_world = URDFParser.from_file(
            os.path.join(self.resource_dir, "robots", "pr2_calibrated_with_ft.urdf")
        ).parse()

        apartment_world = URDFParser.from_file(
            os.path.join(self.resource_dir, "worlds", "apartment.urdf")
        ).parse()

        milk_world = STLParser(
            os.path.join(self.resource_dir, "objects", "milk.stl")
        ).parse()

        cereal_world = STLParser(
            os.path.join(self.resource_dir, "objects", "breakfast_cereal.stl")
        ).parse()

        # Merge worlds
        apartment_world.merge_world(milk_world)
        apartment_world.merge_world(cereal_world)

        with apartment_world.modify_world():
            pr2_root = pr2_sem_world.get_body_by_name("base_footprint")
            apartment_root = apartment_world.root
            c_root_bf = OmniDrive.create_with_dofs(
                parent=apartment_root, child=pr2_root, world=apartment_world
            )
            apartment_world.merge_world(pr2_sem_world, c_root_bf)
            c_root_bf.origin = TransformationMatrix.from_xyz_rpy(1.5, 2.5, 0)

        # Set positions for objects
        apartment_world.get_body_by_name("milk.stl").parent_connection.origin = (
            TransformationMatrix.from_xyz_rpy(
                2.37, 2, 1.05, reference_frame=apartment_world.root
            )
        )
        apartment_world.get_body_by_name(
            "breakfast_cereal.stl"
        ).parent_connection.origin = TransformationMatrix.from_xyz_rpy(
            2.37, 1.8, 1.05, reference_frame=apartment_world.root
        )

        # Add semantic annotation for milk
        milk_view = Milk(body=apartment_world.get_body_by_name("milk.stl"))
        with apartment_world.modify_world():
            apartment_world.add_semantic_annotation(milk_view)

        return apartment_world

    def _reason_about_world(self):
        """
        Runs the WorldReasoner to infer semantic annotations.
        """
        with self.world.modify_world():
            world_reasoner = WorldReasoner(self.world)
            world_reasoner.reason()

    def _setup_ros(self):
        """
        Initializes ROS 2 node and publishers.
        """
        print("Initializing ROS 2 node for visualization...")
        rclpy.init()
        self.node = rclpy.create_node("container_query_node")
        self.viz_publisher = VizMarkerPublisher(node=self.node, world=self.world)
        self.path_publisher = self.node.create_publisher(
            Marker, "/semworld/handle_path", 10
        )

    def find_containers(self) -> List[SemanticAnnotation]:
        """
        Queries the world for Drawer, Fridge, and Door annotations and returns ContainerInfo objects.
        """
        drawers = list(
            an(
                entity(d := let(Drawer, domain=self.world.semantic_annotations))
            ).evaluate()
        )
        fridges = list(
            an(
                entity(f := let(Fridge, domain=self.world.semantic_annotations))
            ).evaluate()
        )
        doors = list(
            an(
                entity(d := let(Door, domain=self.world.semantic_annotations))
            ).evaluate()
        )

        results = drawers + fridges + doors

        return results

    def run(self):
        """
        Main execution loop: finds containers, defines an opening effect for each container, adds that to the world
        and then search for all effects that can be caused by a motion in the given environment.
        Negative example: does not return an effect if the container cannot be opened to the goal value
        """
        containers = self.find_containers()
        print(f"Found {len(containers)} openable objects:")

        effects = []
        for container in containers[:3]:
            if isinstance(container, Drawer) or isinstance(container, Fridge):
                property_getter = (
                    lambda obj: obj.container.body.parent_connection.position
                )
            elif isinstance(container, Door):
                property_getter = lambda obj: obj.body.parent_connection.position
            else:
                continue

            effect_open = OpenEffect(
                target_object=container, goal_value=0.2, property_getter=property_getter
            )
            close_effect = ClosedEffect(
                target_object=container, goal_value=0.0, property_getter=property_getter
            )
            effects.append(effect_open)
            effects.append(close_effect)

        with self.world.modify_world():
            self.world.add_semantic_annotations(effects)

        effect = let(Effect, domain=self.world.semantic_annotations)
        # motion = let(Motion, domain=None)
        # query = an(
        #     entity(
        #         motion,
        #         CausesOpening(effect=effect_open, motion=motion, environment=self.world),
        #     )
        # )
        # motion = let(Motion, domain=None)
        # query = an(
        #     entity(
        #         motion,
        #         predicate := CausesOpening(effect=effect_open, motion=motion, environment=self.world),
        #     )
        # )
        #
        # with query:
        #     Add(motion, predicate.motion)
        #
        # print(list(query.evaluate()))

        motion = let(Motion, domain=None)
        effect = let(Effect, domain=[effect_open])
        causes_opening = CausesOpening(effect=effect, environment=self.world)
        motion = an(entity(causes_opening.motion, causes_opening))
        print(list(motion.evaluate()))

        # causes_opening = CausesOpening(effect=effect_open, environment=self.world)
        # assert causes_opening.__call__() is True
        # print(causes_opening.motion)

    def cleanup(self):
        """
        Shuts down ROS node.
        """
        self.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    demo = ContainerDemo()
    try:
        demo.run()
    finally:
        demo.cleanup()
