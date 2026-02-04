from pycram.designators.location_designator import *
from pycram.language import SequentialPlan
from pycram.datastructures.pose import PoseStamped
from pycram.robot_plans import NavigateActionDescription

arm_park = {
    "l_shoulder_pan_joint": 1.712,
    "l_shoulder_lift_joint": -0.264,
    "l_upper_arm_roll_joint": 1.38,
    "l_elbow_flex_joint": -2.12,
    "l_forearm_roll_joint": 16.996,
    "l_wrist_flex_joint": -0.073,
    "l_wrist_roll_joint": 0.0,
    "r_shoulder_pan_joint": -1.712,
    "r_shoulder_lift_joint": -0.256,
    "r_upper_arm_roll_joint": -1.463,
    "r_elbow_flex_joint": -2.12,
    "r_forearm_roll_joint": 1.766,
    "r_wrist_flex_joint": -0.07,
    "r_wrist_roll_joint": 0.051,
}

logger = logging.getLogger("pycram")
logger.setLevel(logging.DEBUG)


def test_reachability_costmap_location(immutable_simple_pr2_world):
    world, robot, context = immutable_simple_pr2_world

    for name, state in arm_park.items():
        world.state[world.get_degree_of_freedom_by_name(name).id].position = state
    world.state[world.get_degree_of_freedom_by_name("torso_lift_joint").id].position = (
        0.3
    )
    world.notify_state_change()

    location_desig = CostmapLocation(
        world.get_body_by_name("milk.stl"), reachable_for=robot
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4
    # assertTrue(Arms.LEFT == location.reachable_arm or Arms.RIGHT == location.reachable_arm)


def test_reachability_pose_costmap_location(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    for name, state in arm_park.items():
        world.state[world.get_degree_of_freedom_by_name(name).id].position = state
    world.notify_state_change()
    location_desig = CostmapLocation(
        PoseStamped.from_list([-2.2, 0, 1], [0, 0, 0, 1], world.root),
        reachable_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_visibility_costmap_location(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    for name, state in arm_park.items():
        world.state[world.get_degree_of_freedom_by_name(name).id].position = state
    world.notify_state_change()
    location_desig = CostmapLocation(
        world.get_body_by_name("milk.stl"), visible_for=robot_view
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_visibility_pose_costmap_location(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    for name, state in arm_park.items():
        world.state[world.get_degree_of_freedom_by_name(name).id].position = state
    world.notify_state_change()
    location_desig = CostmapLocation(
        PoseStamped.from_list([-1, 0, 1.2], frame=world.root),
        visible_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_reachability_and_visibility_costmap_location(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    for name, state in arm_park.items():
        world.state[world.get_degree_of_freedom_by_name(name).id].position = state
    world.state[world.get_degree_of_freedom_by_name("torso_lift_joint").id].position = (
        0.3
    )
    world.notify_state_change()
    location_desig = CostmapLocation(
        world.get_body_by_name("milk.stl"),
        reachable_for=robot_view,
        visible_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_reachability_probabilistic_costmap_location(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    for name, state in arm_park.items():
        world.state[world.get_degree_of_freedom_by_name(name).id].position = state
    world.state[world.get_degree_of_freedom_by_name("torso_lift_joint").id].position = (
        0.3
    )
    world.notify_state_change()
    location_desig = ProbabilisticCostmapLocation(
        world.get_body_by_name("milk.stl"), reachable_for=robot_view
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4
    # assertTrue(Arms.LEFT == location.reachable_arm or Arms.RIGHT == location.reachable_arm)


def test_reachability_pose_probabilistic_costmap_location(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    for name, state in arm_park.items():
        world.state[world.get_degree_of_freedom_by_name(name).id].position = state
    location_desig = ProbabilisticCostmapLocation(
        PoseStamped.from_list([0.4, 0.6, 0.9], [0, 0, 0, 1], frame=world.root),
        reachable_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4
    # assertTrue(Arms.LEFT == location.reachable_arm or Arms.RIGHT == location.reachable_arm)


def test_visibility_probabilistic_costmap_location(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    for name, state in arm_park.items():
        world.state[world.get_degree_of_freedom_by_name(name).id].position = state
    location_desig = ProbabilisticCostmapLocation(
        world.get_body_by_name("milk.stl"), visible_for=robot_view
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_visibility_pose_probabilistic_costmap_location(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    for name, state in arm_park.items():
        world.state[world.get_degree_of_freedom_by_name(name).id].position = state
    location_desig = ProbabilisticCostmapLocation(
        PoseStamped.from_list([-1, 0, 1.2], frame=world.root),
        visible_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_reachability_and_visibility_probabilistic_costmap_location(
    immutable_simple_pr2_world,
):
    world, robot_view, context = immutable_simple_pr2_world
    for name, state in arm_park.items():
        world.state[world.get_degree_of_freedom_by_name(name).id].position = state
    world.state[world.get_degree_of_freedom_by_name("torso_lift_joint").id].position = (
        0.3
    )
    world.notify_state_change()
    location_desig = ProbabilisticCostmapLocation(
        world.get_body_by_name("milk.stl"),
        reachable_for=robot_view,
        visible_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_semantic_location(immutable_model_world):
    world, robot_view, context = immutable_model_world
    location_desig = SemanticCostmapLocation(
        world.get_body_by_name("island_countertop")
    )
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4

    location_desig = SemanticCostmapLocation(
        world.get_body_by_name("island_countertop"),
        for_object=world.get_body_by_name("milk.stl"),
    )
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_probabilistic_semantic_location(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    location_desig = ProbabilisticSemanticLocation(
        [world.get_body_by_name("box_2")], link_is_center_link=True
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4

    location_desig = ProbabilisticSemanticLocation(
        [world.get_body_by_name("box")],
        for_object=world.get_body_by_name("milk.stl"),
        link_is_center_link=True,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_accessing_location(immutable_model_world):
    world, robot_view, context = immutable_model_world
    location_desig = AccessingLocation(
        world.get_body_by_name("handle_cab10_t"),
        robot_desig=robot_view,
        arm=Arms.RIGHT,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    access_pose = location_desig.resolve()

    assert len(access_pose.position.to_list()) == 3
    assert len(access_pose.orientation.to_list()) == 4


def test_giskard_location_pose(immutable_model_world):
    world, robot_view, context = immutable_model_world
    location_desig = GiskardLocation(
        PoseStamped.from_list([2.1, 2, 1], frame=world.root), Arms.RIGHT
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))

    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_costmap_location_last_result(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    for name, state in arm_park.items():
        world.state[world.get_degree_of_freedom_by_name(name).id].position = state
    world.notify_state_change()
    location_desig = CostmapLocation(
        PoseStamped.from_list([-2.2, 0, 1], [0, 0, 0, 1], world.root),
        reachable_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    last_result = next(location_desig.last_result)

    assert len(last_result.position.to_list()) == 3
    assert len(last_result.orientation.to_list()) == 4
    assert location == last_result
    assert location is last_result
