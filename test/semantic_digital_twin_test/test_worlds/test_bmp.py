import os
from copy import deepcopy

from pkg_resources import resource_filename
from rdflib.plugins.sparql.parser import PrefixedName

from krrood.entity_query_language.conclusion import Add, Set
from krrood.entity_query_language.entity import entity, set_of, inference, variable
from krrood.entity_query_language.match import match
from krrood.entity_query_language.entity_result_processors import an, a
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.reasoning.predicates_base import (
    SatisfiesRequest,
    Causes,
    CanExecute,
)
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Container,
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
import pytest
import rclpy
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    ActiveConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    CollisionCheckingConfig,
)


@pytest.fixture(scope="function")
def mutable_model_world(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    return world


class TestBodyMotionProblem:
    @staticmethod
    def _get_effect_execution_model_for_open_goal(
        handle_body, actuator, goal_value
    ) -> RunMSCModel:
        msc = MotionStatechart()
        goal = Open(
            tip_link=handle_body,
            environment_link=handle_body,
            goal_joint_state=goal_value,
        )
        msc.add_node(goal)
        msc.add_node(EndMotion.when_true(goal))

        return RunMSCModel(msc=msc, actuator=actuator, timeout=500)

    def _extend_world(self, world: World):
        with world.modify_world():
            world_reasoner = WorldReasoner(world)
            world_reasoner.reason()

        drawers = world.get_semantic_annotations_by_type(Drawer)

        # Define effects for the drawers
        effects = []
        motions = []
        property_getter = lambda obj: obj.container.body.parent_connection.position
        for drawer in drawers:
            effect_open = OpenedEffect(
                target_object=drawer, goal_value=0.3, property_getter=property_getter
            )
            close_effect = ClosedEffect(
                target_object=drawer,
                goal_value=0.0,
                property_getter=property_getter,
            )
            effects.append(effect_open)
            effects.append(close_effect)
            close_motion = Motion(
                trajectory=[],
                actuator=drawer.container.body.parent_connection,
                motion_model=self._get_effect_execution_model_for_open_goal(
                    drawer.handle.body, drawer.container.body.parent_connection, 0.0
                ),
            )
            open_motion = Motion(
                trajectory=[],
                actuator=drawer.container.body.parent_connection,
                motion_model=self._get_effect_execution_model_for_open_goal(
                    drawer.handle.body, drawer.container.body.parent_connection, 0.3
                ),
            )
            motions.append(open_motion)
            motions.append(close_motion)

        # Define simple TaskRequests
        open_task = TaskRequest(task_type="open", name="open_drawer")
        close_task = TaskRequest(task_type="close", name="close_drawer")
        return effects, motions, open_task, close_task, drawers

    def test_query_motion_satisfying_task_request1(self, mutable_model_world):
        world = mutable_model_world
        effects, motions, open_task, close_task, _ = self._extend_world(world)

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

        motion_key, effect_key, request_key = list(results[0].data.keys())
        print(
            f"A solution to fulfill the request {results[0].data[request_key].task_type} \n"
            f"is to achieve effect {results[0].data[effect_key].name} by executing the motion \n"
            f"trajectory {results[0].data[motion_key].trajectory} \n"
            f"on the DoF {results[0].data[motion_key].actuator.name}"
        )

    def test_query_task_and_effect_satisfying_motion(self, mutable_model_world: World):
        apartment_world = mutable_model_world
        effects, _, open_task, close_task, drawers = self._extend_world(apartment_world)

        # Define a motion
        motion = Motion(
            trajectory=[0.0, 0.1, 0.2, 0.3, 0.4],
            actuator=drawers[0].container.body.parent_connection,
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
        apartment_world = mutable_model_world
        effects, motions, open_task, close_task, drawers = self._extend_world(
            apartment_world
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
            drawer.container.body.parent_connection.position = 0.3
        apartment_world.notify_state_change()

        # query again
        results = list(query.evaluate())
        print(len(results))
        print(len(drawers))
        assert all([res.data[task_sym].task_type == "close" for res in results])
        print("second query done with task type ", results[0].data[task_sym].task_type)

    def get_world(self):
        urdf_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "../..",
            "pycram",
            "resources",
            "robots",
        )
        pr2 = os.path.join(urdf_dir, "pr2_with_ft2_cableguide.urdf")
        pr2_parser = URDFParser.from_file(file_path=pr2)
        world_with_pr2 = pr2_parser.parse()
        # leere welt, die andere reinmergen im with block, wie bei giskard.py
        with world_with_pr2.modify_world():
            # DoF has hardware interface flag hier setzen. kann ich mir bei pr2_standalone_confign abschauen
            pr2_root = world_with_pr2.root
            localization_body = Body(name=PrefixedName("odom_combined"))
            world_with_pr2.add_kinematic_structure_entity(localization_body)
            c_root_bf = OmniDrive.create_with_dofs(
                parent=localization_body, child=pr2_root, world=world_with_pr2
            )
            world_with_pr2.add_connection(c_root_bf)
            robot = PR2.from_world(world_with_pr2)

        with world_with_pr2.modify_world():
            path_to_srdf = resource_filename(
                "giskardpy", "../../self_collision_matrices/iai/pr2.srdf"
            )
            world_with_pr2.load_collision_srdf(path_to_srdf)
            frozen_joints = ["r_gripper_l_finger_joint", "l_gripper_l_finger_joint"]
            for joint_name in frozen_joints:
                c: ActiveConnection = world_with_pr2.get_connection_by_name(joint_name)
                c.frozen_for_collision_avoidance = True

            for body in robot.bodies_with_collisions:
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.1, violated_distance=0.0
                )
                body.set_static_collision_config(collision_config)

            for joint_name in ["r_wrist_roll_joint", "l_wrist_roll_joint"]:
                connection: ActiveConnection = world_with_pr2.get_connection_by_name(
                    joint_name
                )
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.05,
                    violated_distance=0.0,
                    max_avoided_bodies=4,
                )
                connection.set_static_collision_config_for_direct_child_bodies(
                    collision_config
                )

            for joint_name in ["r_wrist_flex_joint", "l_wrist_flex_joint"]:
                connection: ActiveConnection = world_with_pr2.get_connection_by_name(
                    joint_name
                )
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.05,
                    violated_distance=0.0,
                    max_avoided_bodies=2,
                )
                connection.set_static_collision_config_for_direct_child_bodies(
                    collision_config
                )
            for joint_name in ["r_elbow_flex_joint", "l_elbow_flex_joint"]:
                connection: ActiveConnection = world_with_pr2.get_connection_by_name(
                    joint_name
                )
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.05,
                    violated_distance=0.0,
                    max_avoided_bodies=1,
                )
                connection.set_static_collision_config_for_direct_child_bodies(
                    collision_config
                )
            for joint_name in ["r_forearm_roll_joint", "l_forearm_roll_joint"]:
                connection: ActiveConnection = world_with_pr2.get_connection_by_name(
                    joint_name
                )
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.025,
                    violated_distance=0.0,
                    max_avoided_bodies=1,
                )
                connection.set_static_collision_config_for_direct_child_bodies(
                    collision_config
                )

            collision_config = CollisionCheckingConfig(
                buffer_zone_distance=0.2, violated_distance=0.1, max_avoided_bodies=2
            )
            robot.drive.set_static_collision_config_for_direct_child_bodies(
                collision_config
            )
        return world_with_pr2

    def test_query_motion_satisfying_task_request2(self, mutable_model_world: World):
        world = mutable_model_world
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("viz_node")
        VizMarkerPublisher(world=world, node=node)

        effects, motions, open_task, close_task, drawers = self._extend_world(world)

        task_sym = variable(TaskRequest, domain=[open_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions[0])

        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(effect=effect_sym, motion=motion_sym, environment=world)
        robot = PR2.from_world(world)
        can_execute = CanExecute(motion=motion_sym, robot=robot)

        query = an(
            set_of(task_sym, motion_sym).where(
                satisfies_request, causes_opening, can_execute
            )
        )

        results = list(query.evaluate())
        # motion: Motion = results[0]
        print(len(results))
        # print(motion)
        # assert len(results) == len(drawers)
