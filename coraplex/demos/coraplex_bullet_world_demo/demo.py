import os

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    VisualizationBackend,
)
from coraplex.datastructures.grasp import GraspDescription

from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.visualization import WorldVisualization
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction

from coraplex.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Spoon,
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color
from coraplex.datastructures.enums import VisualizationBackend
from coraplex.visualization import WorldVisualization

world = setup_world()

visualization = WorldVisualization.from_environment(
    world, default_backend=VisualizationBackend.CRAMERA
).start()
spoon = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "spoon.stl"
    )
).parse()
bowl = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
    )
).parse()

# Distinct colors so the objects read clearly against the apartment.
world.get_body_by_name("milk.stl").visual.shapes[0].color = Color(0.9, 0.93, 1.0)
bowl.root.visual.shapes[0].color = Color(0.9, 0.75, 0.5)
spoon.root.visual.shapes[0].color = Color(0.6, 0.6, 0.65)

with world.modify_world():
    world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            2.4, 2.2, 1, reference_frame=world.root
        ),
    )
    connection = FixedConnection(
        parent=world.get_body_by_name("cabinet10_drawer_top"),
        child=spoon.root,
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            -0.05, -0.05, 0
        ),
    )
    world.merge_world(spoon, connection)


visualization = WorldVisualization.from_environment(
    world, default_backend=VisualizationBackend.CRAMERA
).start()

pr2 = PR2.from_world(world)
context = Context(world=world, robot=pr2, _debug=False, ros_node=visualization.ros_node)

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()
    world.add_semantic_annotations(
        [
            Bowl(root=world.get_body_by_name("bowl.stl")),
            Spoon(root=world.get_body_by_name("spoon.stl")),
        ]
    )
    world.add_semantic_annotation_recursively(
        Drawer(
            root=world.get_body_by_name("cabinet10_drawer_top"),
            handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
        )
    )

context.evaluate_conditions = False

plan = sequential(
    [
        ParkArmsAction(Arms.BOTH),
        MoveTorsoAction(TorsoState.HIGH),
        TransportAction(
            world.get_body_by_name("milk.stl"),
            Pose.from_xyz_rpy(4.9, 3.3, 0.8, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
        ),
        TransportAction(
            world.get_body_by_name("bowl.stl"),
            Pose.from_xyz_rpy(5, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
        ),
        TransportAction(
            world.get_body_by_name("spoon.stl"),
            Pose.from_xyz_rpy(5.1, 3.3, 0.75, yaw=1.57, reference_frame=world.root),
            Arms.LEFT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.TOP,
                pr2.left_arm.end_effector,
            ),
        ),
    ],
    context=context,
).plan
visualization.attach_plan(plan)

with simulated_robot:
    plan.perform()

