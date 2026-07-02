from __future__ import annotations

from demo_source_paths import add_workspace_source_paths
#from test.conftest import tracy_world

add_workspace_source_paths(__file__)

import rclpy

from coraplex.plans.plan import Plan
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world import World

from coraplex.datastructures.dataclasses import Context
from coraplex.motion_executor import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import PickAndPlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from robokudo_msgs.action import Query

def query_colored_block_poses_from_robokudo(node) -> dict[str, PoseStamped]:
      action_client = ActionClient(node, Query, "/robokudo/query")

      if not action_client.wait_for_server(timeout_sec=5.0):
          raise RuntimeError("RoboKudo query action server is not available.")

      goal = Query.Goal()
      goal.obj.type = "block"

      send_future = action_client.send_goal_async(goal)
      rclpy.spin_until_future_complete(node, send_future)

      goal_handle = send_future.result()
      if not goal_handle.accepted:
          raise RuntimeError("RoboKudo rejected the block query.")

      result_future = goal_handle.get_result_async()
      rclpy.spin_until_future_complete(node, result_future)

      result = result_future.result().result
      poses_by_color: dict[str, PoseStamped] = {}

      for object_designator in result.res:
          if not object_designator.pose:
              continue

          for color in object_designator.color:
              if color in {"red", "yellow", "blue"}:
                  poses_by_color[color] = object_designator.pose[0]

      missing_colors = {"red", "yellow", "blue"} - set(poses_by_color)
      if missing_colors:
          raise RuntimeError(f"RoboKudo did not detect blocks with colors:{sorted(missing_colors)}")

      return poses_by_color


def load_tracy_world() -> World:
    # project_root: Path = Path(__file__).resolve().parents[3]
    # tracy_path: Path = project_root / "semantic_digital_twin" / "resources" / "urdf" / "tracy.urdf"

    # if not tracy_path.is_file():
    #    raise FileNotFoundError("Tracy URDF file not found at: {}".format(tracy_path))

    # return URDFParser.from_file(str(tracy_path)).parse()
    return URDFParser.from_file(Tracy.get_ros_file_path()).parse()

def spawn_free_box(
        spawn_world: World,
        name: str = "box",
        position: tuple = (0.0, 0.0, 1.5),
        scale: Scale = Scale(0.1, 0.1, 0.1),
        color: Color = Color(1.0, 1.0, 0.0, 1.0)
) -> Body:
    spawn_body = Body(name=PrefixedName(name))

    box = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            reference_frame=spawn_body,
        ),
        scale=scale,
        color=color,
    )
    spawn_body.collision = ShapeCollection([box], reference_frame=spawn_body)

    with spawn_world.modify_world():
        connection = Connection6DoF.create_with_dofs(
            parent=spawn_world.root,
            child=spawn_body,
            world=spawn_world,
        )
        spawn_world.add_connection(connection)

        # Set the initial world pose of the box via the 6-DoF DoF state.
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=position[0],
            y=position[1],
            z=position[2],
            reference_frame=spawn_body,
        )

    return spawn_body

def setup_world(block_poses: dict[str, PoseStamped]) -> World:

    tracy_world: World = load_tracy_world()

    spawn_free_box(tracy_world, "box1", pose_to_position(block_poses["red"]), color=Color(1.0, 0.0, 0.0, 1.0))
    spawn_free_box(tracy_world, "box2", pose_to_position(block_poses["yellow"]), color=Color(1.0, 1.0, 0.0, 1.0))
    spawn_free_box(tracy_world, "box3", pose_to_position(block_poses["blue"]), color=Color(0.0, 0.0, 1.0, 1.0))

    return tracy_world

def pose_to_position(pose_stamped: PoseStamped) -> tuple[float, float, float]:
      position = pose_stamped.pose.position
      return position.x, position.y, position.z


def build_plan(world: World, tracy: Tracy, context: Context) -> Plan | None:
    return sequential(
        [
            ParkArmsAction(Arms.BOTH),
            PickAndPlaceAction(
                world.get_body_by_name("box3"),
                Pose.from_xyz_rpy(0.6, 0.0, 0.93, reference_frame=world.root),
                Arms.RIGHT,
                GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP, tracy.right_arm.end_effector),
            ),
            PickAndPlaceAction(
                world.get_body_by_name("box1"),
                Pose.from_xyz_rpy(0.6, 0.0, 1.03, reference_frame=world.root),
                Arms.LEFT,
                GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP, tracy.left_arm.end_effector),
            ),
            PickAndPlaceAction(
                world.get_body_by_name("box2"),
                Pose.from_xyz_rpy(0.6, 0.0, 1.13, reference_frame=world.root),
                Arms.RIGHT,
                GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP, tracy.right_arm.end_effector),
            ),
        ],
        context=context,
    ).plan

def run_simulation():
    node = rclpy.create_node("viz_marker")
    block_poses = query_colored_block_poses_from_robokudo(node)
    print(block_poses)
    world = setup_world(block_poses)#change2
    tracy = Tracy.from_world(world)

    context = Context(world=world, robot=tracy)
    context.evaluate_conditions = False

    VizMarkerPublisher(_world=world, node=node).with_tf_publisher()

    #plan: Plan | None = build_plan(world, tracy, context)

    #if plan is None:
        #print("No valid plan could be generated. Exiting.")
        #return

    #with simulated_robot:
        #plan.perform()

def main():
    rclpy.init()
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()