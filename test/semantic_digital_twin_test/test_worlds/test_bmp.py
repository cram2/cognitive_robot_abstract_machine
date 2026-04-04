import random
import time
from copy import deepcopy

import pytest
import rclpy
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from krrood.entity_query_language.factories import variable, an, set_of
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.reasoning.predicates_base import (
    SatisfiesRequest,
    Causes,
    CanExecute,
)
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Door,
)
from semantic_digital_twin.semantic_annotations.task_effect_motion import (
    OpenedEffect,
    ClosedEffect,
    TaskRequest,
    Effect,
    Motion,
    RunMSCModel,
)
from semantic_digital_twin.world import World


@pytest.fixture(scope="function")
def mutable_model_world(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    return world


@pytest.fixture
def stretch_kitchen_world(stretch_world, kitchen_world):
    world = deepcopy(stretch_world)
    world.merge_world(deepcopy(kitchen_world))
    return world


@pytest.fixture
def tiago_kitchen_world(tiago_world, kitchen_world):
    world = deepcopy(tiago_world)
    world.merge_world(deepcopy(kitchen_world))
    return world


class TestBodyMotionProblem:
    @staticmethod
    def _get_effect_execution_model_for_open_goal(
        handle_body, actuator, goal_value
    ) -> RunMSCModel:
        """
        Create a motion statechart model for an open goal.
        """
        msc = MotionStatechart()
        goal = Open(
            tip_link=handle_body,
            environment_link=handle_body,
            goal_joint_state=goal_value,
        )
        msc.add_node(goal)
        msc.add_node(EndMotion.when_true(goal))

        return RunMSCModel(msc=msc, actuator=actuator, timeout=500)

    def _extend_world(
        self, world: World, only_drawers: bool = False, only_doors: bool = False
    ):
        """
        Extend the world with semantic annotations, effects, and motions for drawers and doors.
        """
        world_reasoner = WorldReasoner(world)
        inferred = world_reasoner.infer_semantic_annotations()
        with world.modify_world():
            world.add_semantic_annotations(inferred)

        drawers = [] if only_doors else world.get_semantic_annotations_by_type(Drawer)

        doors = [] if only_drawers else world.get_semantic_annotations_by_type(Door)

        annotations = drawers + doors
        print(f"len annotations: {len(annotations)}")
        print(f"len drawers: {len(drawers)}")

        # Define effects for the drawers
        effects = []
        motions = []
        property_getter = lambda obj: obj.root.parent_connection.position
        for annotation in annotations:
            act = annotation.root.parent_connection
            effect_open = OpenedEffect(
                target_object=annotation,
                goal_value=act.active_dofs[0].limits.upper.position,
                property_getter=property_getter,
            )
            close_effect = ClosedEffect(
                target_object=annotation,
                goal_value=act.active_dofs[0].limits.lower.position,
                property_getter=property_getter,
            )
            effects.append(effect_open)
            effects.append(close_effect)

            close_motion = Motion(
                trajectory=[],
                actuator=act,
                motion_model=self._get_effect_execution_model_for_open_goal(
                    annotation.handle,
                    act,
                    act.active_dofs[0].limits.lower.position,
                ),
            )
            open_motion = Motion(
                trajectory=[],
                actuator=act,
                motion_model=self._get_effect_execution_model_for_open_goal(
                    annotation.handle,
                    act,
                    act.active_dofs[0].limits.upper.position,
                ),
            )
            motions.append(open_motion)
            motions.append(close_motion)

        # Define simple TaskRequests
        open_task = TaskRequest(task_type="open", name="open_container")
        close_task = TaskRequest(task_type="close", name="close_container")
        return effects, motions, open_task, close_task, drawers

    def _extend_world2(self, world: World):
        """
        Extend the world with simplified semantic annotations, effects, and motions for drawers and doors.
        """
        world_reasoner = WorldReasoner(world)
        inferred = world_reasoner.infer_semantic_annotations()
        with world.modify_world():
            world.add_semantic_annotations(inferred)

        drawers = world.get_semantic_annotations_by_type(Drawer)

        doors = world.get_semantic_annotations_by_type(Door)

        annotations = drawers + doors
        property_getter = lambda obj: obj.root.parent_connection.position

        effects = []
        motions = []
        for a in annotations:
            act = a.root.parent_connection
            max_value = act.active_dofs[0].limits.upper.position
            if isinstance(a, Door):
                max_value /= 2
            open_effect = OpenedEffect(
                target_object=a,
                goal_value=max_value,
                property_getter=property_getter,
            )
            effects.append(open_effect)

            open_motion = Motion(
                trajectory=[],
                actuator=act,
                motion_model=self._get_effect_execution_model_for_open_goal(
                    a.handle,
                    act,
                    act.active_dofs[0].limits.upper.position,
                ),
            )
            motions.append(open_motion)

            open_task = TaskRequest(task_type="open", name="open_container")

        return effects, motions, open_task, None, drawers

    def test_query_motion_satisfying_task_request1(self, mutable_model_world):
        """
        Test whether a motion can be found that satisfies a given task request in the apartment world.
        """
        world = mutable_model_world
        effects, motions, open_task, close_task, _ = self._extend_world(world)

        # Define Krrood symbols
        task_sym = variable(TaskRequest, domain=[open_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        # Define Predicates for the query
        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(effect=effect_sym, motion=motion_sym, environment=world)

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                satisfies_request,
                causes_opening,
            )
        )
        results = list(query.evaluate())
        print(len(results))
        motion_key, effect_key, request_key = list(results[0].data.keys())
        print(
            f"A solution to fulfill the request {results[0].data[request_key].task_type} \n"
            f"is to achieve effect {results[0].data[effect_key].name} by executing the motion \n"
            f"trajectory {results[0].data[motion_key].trajectory} \n"
            f"on the DoF {results[0].data[motion_key].actuator.name}"
        )

    def test_query_motion_satisfying_task_request_not_all(self, mutable_model_world):
        """
        Test motion querying when some drawers are already open and not all effects can be achieved.
        """
        world = mutable_model_world
        effects, motions, open_task, close_task, _ = self._extend_world2(world)

        for drawer in world.get_semantic_annotations_by_type(Drawer):
            if random.randint(0, 5) == 4:
                max_position = drawer.root.parent_connection.active_dofs[
                    0
                ].limits.upper.position
                drawer.root.parent_connection.position = max_position
                print(f"drawer {drawer.name} moved to position {max_position}")

        print(f"len of effects: {len(effects)}, len of motions: {len(motions)}")

        # Define Krrood symbols
        task_sym = variable(TaskRequest, domain=[open_task, close_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        # Define Predicates for the query
        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(effect=effect_sym, motion=motion_sym, environment=world)

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                satisfies_request,
                causes_opening,
            )
        )
        results = list(query.evaluate())
        print(len(results))

        motion_key, effect_key, request_key = list(results[0].data.keys())
        print(
            f"A solution to fulfill the request {results[0].data[request_key].task_type} \n"
            f"is to achieve effect {results[0].data[effect_key].name} by executing the motion \n"
            f"trajectory {results[0].data[motion_key].trajectory} \n"
            f"on the DoF {results[0].data[motion_key].actuator.name}"
        )

    def test_query_task_and_effect_satisfying_motion(self, mutable_model_world: World):
        """
        Test whether the task request and effect can be found for a given motion in the apartment world.
        """
        apartment_world = mutable_model_world
        effects, _, open_task, close_task, drawers = self._extend_world(apartment_world)

        # Define a motion
        motion = Motion(
            trajectory=[0.0, 0.1, 0.2, 0.3, 0.4],
            actuator=drawers[0].root.parent_connection,
        )

        # Define Krrood symbols
        task_sym = variable(TaskRequest, domain=[open_task, close_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=[motion])

        # Define Predicates for the query
        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(
            effect=effect_sym, motion=motion_sym, environment=apartment_world
        )

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                satisfies_request, causes_opening
            )
        )
        results = list(query.evaluate())

        motion_key, effect_key, request_key = list(results[0].data.keys())
        print(
            f"The trajectory {results[0].data[motion_key].trajectory} \n"
            f"on the DoF {results[0].data[motion_key].actuator.name} \n"
            f"can be caused by effect {results[0].data[effect_key].name} \n"
            f"Which satisfies the request {results[0].data[request_key].task_type} \n"
        )

    def test_query_motion_if_drawers_open(self, mutable_model_world):
        """
        Test that querying for motions adapts to changes in the world state (drawers opening).
        """
        apartment_world = mutable_model_world
        effects, motions, open_task, close_task, drawers = self._extend_world(
            apartment_world, only_drawers=True
        )

        # Define Krrood symbols
        task_sym = variable(TaskRequest, domain=[open_task, close_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        # Define Predicates for the query
        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(
            effect=effect_sym, motion=motion_sym, environment=apartment_world
        )

        # Query for motions that can be causes in the current world based on defined task requests, effects
        # and the world state. Only OpenedEffects should be available, as all drawers are closed
        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                satisfies_request,
                causes_opening,
            )
        )
        results = list(query.evaluate())
        print(len(results))
        # assert that all entries are "open"
        assert all([res.data[task_sym].task_type == "open" for res in results])
        print("first query done with task type ", results[0].data[task_sym].task_type)

        # change the world state
        for drawer in drawers:
            drawer.root.parent_connection.position = (
                drawer.root.parent_connection.active_dofs[0].limits.upper.position
            )
        apartment_world.notify_state_change()

        # query again
        results = list(query.evaluate())
        print(len(results))
        print(len(drawers))
        assert all([res.data[task_sym].task_type == "close" for res in results])
        print("second query done with task type ", results[0].data[task_sym].task_type)

    def present_results(self, results, robot: AbstractRobot):
        """
        Print the results of a motion query to the console.
        """
        for r in results:
            task, motion, effect = r.values()
            print("-" * 20)
            print(f"Task: {task}")
            print(f"Robot: {robot.name}")
            print(
                f"Effect: {effect.__class__.__name__}  for target: {effect.target_object.__class__.__name__}(name={effect.target_object.name} body={effect.target_object.root.name} handle={effect.target_object.handle.name})"
            )

    def test_query_motion_satisfying_task_request_stretch(self, stretch_kitchen_world):
        """
        Test motion querying for a specific task request (oven door) using a Stretch robot in the kitchen world.
        """
        world = stretch_kitchen_world
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("viz_node")
        viz = VizMarkerPublisher(node=node, _world=world)
        viz.with_tf_publisher()

        effects, motions, open_task, close_task, drawers = self._extend_world(
            world, only_drawers=True
        )

        # task_sym = variable(
        #     TaskRequest,
        #     domain=[TaskRequest(task_type="open", name="oven_area_oven_door")],
        # )
        task_sym = variable(TaskRequest, domain=[open_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(effect=effect_sym, motion=motion_sym, environment=world)
        robot = Stretch.from_world(world)
        can_execute = CanExecute(motion=motion_sym, robot=robot)

        query = an(
            set_of(task_sym, motion_sym, effect_sym).where(
                satisfies_request, causes_opening, can_execute
            )
        )

        results = list(query.evaluate())
        # motion: Motion = results[0]
        self.present_results(results, robot)
        print(len(results))
        # print(motion)
        assert len(results) == len(drawers)
        names = []
        for r in results:
            task, motion, effect = r.values()
            names.append(effect.target_object.root.name.name)
        diff = set([(a.target_object.root.name.name) for a in effects]).difference(
            set(names)
        )
        print(diff)
        print(len(diff))

    def test_query_motion_satisfying_task_request_tiago(self, tiago_kitchen_world):
        """
        Test motion querying for a specific task request using a Tiago robot in the kitchen world.
        """
        world = tiago_kitchen_world
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("viz_node")
        viz = VizMarkerPublisher(_world=world, node=node)
        viz.with_tf_publisher()
        effects, motions, open_task, close_task, drawers = self._extend_world(
            world, only_doors=True
        )

        task_sym = variable(TaskRequest, domain=[open_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(effect=effect_sym, motion=motion_sym, environment=world)
        robot = Tiago.from_world(world)
        can_execute = CanExecute(motion=motion_sym, robot=robot)

        query = an(
            set_of(task_sym, motion_sym, effect_sym).where(
                satisfies_request, causes_opening, can_execute
            )
        )

        results = list(query.evaluate())
        # motion: Motion = results[0]
        self.present_results(results, robot)
        print(len(results))
        # print(motion)
        # assert len(results) == len(drawers)
        names = []
        for r in results:
            task, motion, effect = r.values()
            names.append(effect.target_object.root.name.name)
        diff = set([(a.target_object.root.name.name) for a in effects]).difference(
            set(names)
        )
        print(diff)
        print(len(diff))

    def test_query_task_and_effect_satisfying_motion_pr2(self, mutable_model_world):
        """
        Test querying for task and effect for a given motion using a PR2 robot in the apartment world.
        """
        world = mutable_model_world
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("viz_node")
        viz = VizMarkerPublisher(_world=world, node=node)
        viz.with_tf_publisher()

        effects, _, open_task, close_task, drawers = self._extend_world(world)

        # Define a motion
        motion = Motion(
            trajectory=[0.0, 0.1, 0.2, 0.3, 0.4],
            actuator=drawers[0].root.parent_connection,
        )

        # Define Krrood symbols
        task_sym = variable(TaskRequest, domain=[open_task, close_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=[motion])

        # Define Predicates for the query
        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(effect=effect_sym, motion=motion_sym, environment=world)
        robot = PR2.from_world(world)
        can_execute = CanExecute(motion=motion_sym, robot=robot)

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                satisfies_request, causes_opening, can_execute
            )
        )
        results = list(query.evaluate())

        motion_key, effect_key, request_key = list(results[0].data.keys())
        print(
            f"The trajectory {results[0].data[motion_key].trajectory} \n"
            f"on the DoF {results[0].data[motion_key].actuator.name} \n"
            f"can be caused by effect {results[0].data[effect_key].name} \n"
            f"Which satisfies the request {results[0].data[request_key].task_type} \n"
        )

    def test_satisfies_request(self, mutable_model_world):
        """
        Test the SatisfiesRequest predicate.
        """
        world = mutable_model_world
        effects, _, open_task, _, _ = self._extend_world(world)

        effect = next(e for e in effects if isinstance(e, OpenedEffect))
        predicate = SatisfiesRequest(task=open_task, effect=effect)
        assert predicate()

        close_task = TaskRequest(task_type="close", name="close_container")
        predicate_close = SatisfiesRequest(task=close_task, effect=effect)
        assert not predicate_close()

    def test_causes(self, mutable_model_world):
        """
        Test the Causes predicate.
        """
        world = mutable_model_world
        effects, motions, _, _, _ = self._extend_world(world)

        effect = effects[0]  # OpenedEffect for first drawer
        motion = motions[0]  # Open motion for first drawer

        predicate = Causes(effect=effect, motion=motion, environment=world)
        assert predicate()

        effect = effects[0]  # OpenedEffect for first drawer
        motion = motions[1]  # Open motion for second drawer

        predicate = Causes(effect=effect, motion=motion, environment=world)
        assert not predicate()

    def test_can_execute(self, mutable_model_world, rclpy_node):
        """
        Test the CanExecute predicate.
        """
        world = mutable_model_world
        viz = VizMarkerPublisher(_world=world, node=rclpy_node)
        viz.with_tf_publisher()

        effects, motions, _, _, _ = self._extend_world(world)

        motion = motions[0]
        # Need a trajectory for CanExecute to return True
        motion.trajectory = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        robot = PR2.from_world(world)

        predicate = CanExecute(motion=motion, robot=robot)
        assert predicate()


from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration
from rclpy.time import Time


def points_to_path_marker_array(
    points,  # iterable of (x, y, z)
    frame_id="map",
    marker_ns="path",
    marker_id=0,
    line_width=0.02,
    lifetime_sec=0.0,
    r=1.0,
    g=0.0,
    b=0.0,  # red path
) -> MarkerArray:
    """
    Convert a list of 3D points into a MarkerArray for RViz2.
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = Time().to_msg()
    marker.ns = marker_ns
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    # Identity pose, points are in the given frame directly
    marker.pose.orientation.w = 1.0

    # Line width (only x is used for LINE_STRIP)
    marker.scale.x = float(line_width)

    # Color (RGBA in [0, 1], alpha must be > 0)
    marker.color.r = float(r)
    marker.color.g = float(g)
    marker.color.b = float(b)
    marker.color.a = 1.0

    # Optional lifetime (0 = forever)
    marker.lifetime = Duration(
        sec=int(lifetime_sec), nanosec=int((lifetime_sec % 1.0) * 1e9)
    )

    # Fill points
    for x, y, z in points:
        p = Point()
        p.x, p.y, p.z = float(x), float(y), float(z)
        marker.points.append(p)

    marker_array = MarkerArray()
    marker_array.markers.append(marker)
    return marker_array
