import os
import random
from copy import deepcopy
import time

from multiverse_simulator import MultiverseViewer
from pkg_resources import resource_filename

from krrood.entity_query_language.conclusion import Add, Set
from krrood.entity_query_language.entity import entity, set_of, inference, variable
from krrood.entity_query_language.match import match
from krrood.entity_query_language.entity_result_processors import an, a
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from pycram.datastructures.enums import GripperState
from pycram.robot_descriptions.pr2_states import (
    both_park as park_pr2,
    left_gripper_open,
    right_gripper_open,
)
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
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
    Container,
    Milk,
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
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
import pytest
import rclpy
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.simulator_to_world_state import (
    SimulatorToWorldStateSynchronizer,
)
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    ActiveConnection,
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    CollisionCheckingConfig,
)


@pytest.fixture(scope="function")
def mutable_model_world(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    return world


@pytest.fixture
def mutable_stretch_world(stretch_apartment_world):
    return deepcopy(stretch_apartment_world)


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

    def _extend_world(
        self, world: World, only_drawers: bool = False, only_doors: bool = False
    ):
        with world.modify_world():
            world_reasoner = WorldReasoner(world)
            world_reasoner.reason()

        drawers = [] if only_doors else world.get_semantic_annotations_by_type(Drawer)

        doors = [] if only_drawers else world.get_semantic_annotations_by_type(Door)

        annotations = drawers + doors
        print(f"len annotations: {len(annotations)}")
        print(f"len drawers: {len(drawers)}")

        # Define effects for the drawers
        effects = []
        motions = []
        drawer_property_getter = (
            lambda obj: obj.container.body.parent_connection.position
        )
        door_property_getter = lambda obj: obj.body.parent_connection.position
        for annotation in annotations:
            act = (
                annotation.container.body.parent_connection
                if isinstance(annotation, Drawer)
                else annotation.body.parent_connection
            )
            effect_open = OpenedEffect(
                target_object=annotation,
                goal_value=act.active_dofs[0].upper_limits.position,
                property_getter=(
                    drawer_property_getter
                    if isinstance(annotation, Drawer)
                    else door_property_getter
                ),
            )
            close_effect = ClosedEffect(
                target_object=annotation,
                goal_value=act.active_dofs[0].lower_limits.position,
                property_getter=(
                    drawer_property_getter
                    if isinstance(annotation, Drawer)
                    else door_property_getter
                ),
            )
            effects.append(effect_open)
            effects.append(close_effect)

            close_motion = Motion(
                trajectory=[],
                actuator=act,
                motion_model=self._get_effect_execution_model_for_open_goal(
                    annotation.handle.body,
                    act,
                    act.active_dofs[0].lower_limits.position,
                ),
            )
            open_motion = Motion(
                trajectory=[],
                actuator=act,
                motion_model=self._get_effect_execution_model_for_open_goal(
                    annotation.handle.body,
                    act,
                    act.active_dofs[0].upper_limits.position,
                ),
            )
            motions.append(open_motion)
            motions.append(close_motion)

        # Define simple TaskRequests
        open_task = TaskRequest(task_type="open", name="open_container")
        close_task = TaskRequest(task_type="close", name="close_container")
        return effects, motions, open_task, close_task, drawers

    def _extend_world2(self, world: World):
        with world.modify_world():
            world_reasoner = WorldReasoner(world)
            world_reasoner.reason()
        drawers = world.get_semantic_annotations_by_type(Drawer)

        doors = world.get_semantic_annotations_by_type(Door)

        annotations = drawers + doors
        drawer_property_getter = (
            lambda obj: obj.container.body.parent_connection.position
        )
        door_property_getter = lambda obj: obj.body.parent_connection.position

        effects = []
        motions = []
        for a in annotations:
            act = (
                a.container.body.parent_connection
                if isinstance(a, Drawer)
                else a.body.parent_connection
            )
            max_value = act.active_dofs[0].upper_limits.position
            if isinstance(a, Door):
                max_value /= 2
            open_effect = OpenedEffect(
                target_object=a,
                goal_value=max_value,
                property_getter=(
                    drawer_property_getter
                    if isinstance(a, Drawer)
                    else door_property_getter
                ),
            )
            effects.append(open_effect)

            open_motion = Motion(
                trajectory=[],
                actuator=act,
                motion_model=self._get_effect_execution_model_for_open_goal(
                    a.handle.body,
                    act,
                    act.active_dofs[0].upper_limits.position,
                ),
            )
            motions.append(open_motion)

            open_task = TaskRequest(task_type="open", name="open_container")

        return effects, motions, open_task, None, drawers

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

    def test_query_motion_satisfying_task_request_not_all(self, mutable_model_world):
        world = mutable_model_world
        effects, motions, open_task, close_task, _ = self._extend_world2(world)

        for drawer in world.get_semantic_annotations_by_type(Drawer):
            if random.randint(0, 5) == 4:
                max_position = drawer.container.body.parent_connection.active_dofs[
                    0
                ].upper_limits.position
                drawer.container.body.parent_connection.position = max_position
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

    def get_apartment_world(self):
        #### apartment
        apartment_world = URDFParser.from_file(
            os.path.join(
                os.path.dirname(__file__),
                "../../..",
                "pycram",
                "resources",
                "worlds",
                "apartment.urdf",
            )
        ).parse()
        milk_world = STLParser(
            os.path.join(
                os.path.dirname(__file__),
                "../../..",
                "pycram",
                "resources",
                "objects",
                "milk.stl",
            )
        ).parse()
        cereal_world = STLParser(
            os.path.join(
                os.path.dirname(__file__),
                "../../..",
                "pycram",
                "resources",
                "objects",
                "breakfast_cereal.stl",
            )
        ).parse()
        apartment_world.merge_world_at_pose(
            milk_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                2.37, 2, 1.05, reference_frame=apartment_world.root
            ),
        )
        apartment_world.merge_world_at_pose(
            cereal_world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                2.37, 1.8, 1.05, reference_frame=apartment_world.root
            ),
        )
        milk_view = Milk(body=apartment_world.get_body_by_name("milk.stl"))
        with apartment_world.modify_world():
            apartment_world.add_semantic_annotation(milk_view)
        return apartment_world

    def get_kitchen_world(self):
        kitchen_world = URDFParser.from_file(
            os.path.join(
                os.path.dirname(__file__),
                "../../..",
                "pycram",
                "resources",
                "worlds",
                "kitchen.urdf",
            )
        ).parse()

        return kitchen_world

    def get_world(self, use_kitchen=False):
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
        world = World()

        # leere welt, die andere reinmergen im with block, wie bei giskard.py
        with world.modify_world():
            # DoF has hardware interface flag hier setzen. kann ich mir bei pr2_standalone_confign abschauen
            map = Body(name=PrefixedName("map"))
            odom = Body(name=PrefixedName("odom"))
            localization = Connection6DoF.create_with_dofs(
                parent=map, child=odom, world=world
            )
            world.add_connection(localization)

            world_with_robot = pr2_parser.parse()
            robot = PR2.from_world(world_with_robot)

            odom = OmniDrive.create_with_dofs(
                parent=odom,
                child=world_with_robot.root,
                translation_velocity_limits=0.2,
                rotation_velocity_limits=0.2,
                world=world,
            )

            world.merge_world(world_with_robot, odom)

            controlled_joints = [
                "torso_lift_joint",
                # "head_pan_joint",
                # "head_tilt_joint",
                "r_shoulder_pan_joint",
                "r_shoulder_lift_joint",
                "r_upper_arm_roll_joint",
                "r_forearm_roll_joint",
                "r_elbow_flex_joint",
                "r_wrist_flex_joint",
                "r_wrist_roll_joint",
                "l_shoulder_pan_joint",
                "l_shoulder_lift_joint",
                "l_upper_arm_roll_joint",
                "l_forearm_roll_joint",
                "l_elbow_flex_joint",
                "l_wrist_flex_joint",
                "l_wrist_roll_joint",
                odom.name,
            ]
            for joint_name in controlled_joints:
                connection: ActiveConnection = world.get_connection_by_name(joint_name)
                if not isinstance(connection, ActiveConnection):
                    raise Exception(
                        f"{joint_name} is not an active connection and cannot be controlled."
                    )
                connection.has_hardware_interface = True

        with world.modify_world():
            path_to_srdf = resource_filename(
                "giskardpy", "../../self_collision_matrices/iai/pr2.srdf"
            )
            world.load_collision_srdf(path_to_srdf)
            frozen_joints = ["r_gripper_l_finger_joint", "l_gripper_l_finger_joint"]
            for joint_name in frozen_joints:
                c: ActiveConnection = world.get_connection_by_name(joint_name)
                c.frozen_for_collision_avoidance = True

            for body in robot.bodies_with_collisions:
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.01, violated_distance=0.0
                )
                body.set_static_collision_config(collision_config)

            for joint_name in ["r_wrist_roll_joint", "l_wrist_roll_joint"]:
                connection: ActiveConnection = world.get_connection_by_name(joint_name)
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.005,
                    violated_distance=0.0,
                    max_avoided_bodies=4,
                    disabled=True,
                )
                connection.set_static_collision_config_for_direct_child_bodies(
                    collision_config
                )

            for joint_name in ["r_wrist_flex_joint", "l_wrist_flex_joint"]:
                connection: ActiveConnection = world.get_connection_by_name(joint_name)
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.005,
                    violated_distance=0.0,
                    max_avoided_bodies=2,
                )
                connection.set_static_collision_config_for_direct_child_bodies(
                    collision_config
                )
            for joint_name in ["r_elbow_flex_joint", "l_elbow_flex_joint"]:
                connection: ActiveConnection = world.get_connection_by_name(joint_name)
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.005,
                    violated_distance=0.0,
                    max_avoided_bodies=1,
                )
                connection.set_static_collision_config_for_direct_child_bodies(
                    collision_config
                )
            for joint_name in ["r_forearm_roll_joint", "l_forearm_roll_joint"]:
                connection: ActiveConnection = world.get_connection_by_name(joint_name)
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.0025,
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

        # apartment_world = self.get_apartment_world()
        apartment_world = (
            self.get_kitchen_world() if use_kitchen else self.get_apartment_world()
        )

        world.merge_world(apartment_world)
        world.get_body_by_name("base_footprint").parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                *([0, 0, 0] if use_kitchen else [1.5, 2, 0])
            )
        )
        robot = PR2.from_world(world)
        for j in robot.joint_states:
            if j.state_type == "Park":
                j.apply_to_world(world)
        left_gripper_open.apply_to_world(world)
        right_gripper_open.apply_to_world(world)

        return world

    def get_stretch_world(self, use_kitchen=False):
        urdf_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "../..",
            "pycram",
            "resources",
            "robots",
        )
        stretch = os.path.join(urdf_dir, "stretch_description.urdf")
        stretch_parser = URDFParser.from_file(file_path=stretch)
        world = World()

        # leere welt, die andere reinmergen im with block, wie bei giskard.py
        with world.modify_world():
            # DoF has hardware interface flag hier setzen. kann ich mir bei pr2_standalone_confign abschauen
            map = Body(name=PrefixedName("map"))
            odom = Body(name=PrefixedName("odom"))
            localization = Connection6DoF.create_with_dofs(
                parent=map, child=odom, world=world
            )
            world.add_connection(localization)

            world_with_robot = stretch_parser.parse()
            # robot = Stretch.from_world(world_with_robot)

            odom = OmniDrive.create_with_dofs(
                parent=odom,
                child=world_with_robot.root,
                translation_velocity_limits=0.2,
                rotation_velocity_limits=0.2,
                world=world,
            )
            robot = Stretch.from_world(world_with_robot)
            world.merge_world(world_with_robot, odom)

            controlled_joints = [
                "joint_gripper_finger_left",
                "joint_gripper_finger_right",
                # "joint_right_wheel",
                # "joint_left_wheel",
                "joint_lift",
                "joint_arm_l3",
                "joint_arm_l2",
                "joint_arm_l1",
                "joint_arm_l0",
                "joint_wrist_yaw",
                "joint_head_pan",
                "joint_head_tilt",
                odom.name,
            ]
            for joint_name in controlled_joints:
                connection: ActiveConnection = world.get_connection_by_name(joint_name)
                if not isinstance(connection, ActiveConnection):
                    raise Exception(
                        f"{joint_name} is not an active connection and cannot be controlled."
                    )
                connection.has_hardware_interface = True

        with world.modify_world():
            path_to_srdf = resource_filename(
                "giskardpy", "../../self_collision_matrices/iai/stretch.srdf"
            )
            world.load_collision_srdf(path_to_srdf)
            frozen_joints = ["joint_gripper_finger_left", "joint_gripper_finger_right"]
            for joint_name in frozen_joints:
                c: ActiveConnection = world.get_connection_by_name(joint_name)
                c.frozen_for_collision_avoidance = True

            for body in robot.bodies_with_collisions:
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.1, violated_distance=0.0
                )
                body.set_static_collision_config(collision_config)

            for joint_name in frozen_joints:
                c: ActiveConnection = world.get_connection_by_name(joint_name)
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.005,
                    violated_distance=0.0,
                    max_avoided_bodies=4,
                    disabled=True,
                )
                c.set_static_collision_config_for_direct_child_bodies(collision_config)

        # apartment_world = self.get_apartment_world()
        apartment_world = (
            self.get_kitchen_world() if use_kitchen else self.get_apartment_world()
        )

        world.merge_world(apartment_world)
        world.get_body_by_name("base_link").parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                *([0, 0, 0] if use_kitchen else [1.5, 2, 0])
            )
        )

        robot = Stretch.from_world(world)
        for j in robot.joint_states:
            if j.state_type == "Park":  # or j.state_type == "Open":
                j.apply_to_world(world)

        return world

    def get_tiago_world(self, use_kitchen=False):
        urdf_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "../..",
            "pycram",
            "resources",
            "robots",
        )
        tiago = os.path.join(urdf_dir, "tiago_dual.urdf")
        tiago_parser = URDFParser.from_file(file_path=tiago)
        world = World()

        # leere welt, die andere reinmergen im with block, wie bei giskard.py
        with world.modify_world():
            # DoF has hardware interface flag hier setzen. kann ich mir bei pr2_standalone_confign abschauen
            map = Body(name=PrefixedName("map"))
            odom = Body(name=PrefixedName("odom"))
            localization = Connection6DoF.create_with_dofs(
                parent=map, child=odom, world=world
            )
            world.add_connection(localization)

            world_with_robot = tiago_parser.parse()
            # robot = Stretch.from_world(world_with_robot)

            odom = OmniDrive.create_with_dofs(
                parent=odom,
                child=world_with_robot.root,
                translation_velocity_limits=0.2,
                rotation_velocity_limits=0.2,
                world=world,
            )
            robot = Tiago.from_world(world_with_robot)
            world.merge_world(world_with_robot, odom)
            controlled_joints = [
                "torso_lift_joint",
                "head_1_joint",
                "head_2_joint",
                "arm_left_1_joint",
                "arm_left_2_joint",
                "arm_left_3_joint",
                "arm_left_4_joint",
                "arm_left_5_joint",
                "arm_left_6_joint",
                "arm_left_7_joint",
                "arm_right_1_joint",
                "arm_right_2_joint",
                "arm_right_3_joint",
                "arm_right_4_joint",
                "arm_right_5_joint",
                "arm_right_6_joint",
                "arm_right_7_joint",
                "gripper_right_left_finger_joint",
                "gripper_right_right_finger_joint",
                "gripper_left_left_finger_joint",
                "gripper_left_right_finger_joint",
                odom.name,
            ]
            for joint_name in controlled_joints:
                connection: ActiveConnection = world.get_connection_by_name(joint_name)
                if not isinstance(connection, ActiveConnection):
                    raise Exception(
                        f"{joint_name} is not an active connection and cannot be controlled."
                    )
                connection.has_hardware_interface = True

        with world.modify_world():
            path_to_srdf = resource_filename(
                "giskardpy", "../../self_collision_matrices/iai/tiago_dual.srdf"
            )
            world.load_collision_srdf(path_to_srdf)
            frozen_joints = [
                "gripper_right_left_finger_joint",
                "gripper_right_right_finger_joint",
                "gripper_left_left_finger_joint",
                "gripper_left_right_finger_joint",
            ]
            for joint_name in frozen_joints:
                c: ActiveConnection = world.get_connection_by_name(joint_name)
                c.frozen_for_collision_avoidance = True

            for body in robot.bodies_with_collisions:
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.005, violated_distance=0.0
                )
                body.set_static_collision_config(collision_config)

            for joint_name in frozen_joints:
                c: ActiveConnection = world.get_connection_by_name(joint_name)
                collision_config = CollisionCheckingConfig(
                    buffer_zone_distance=0.005,
                    violated_distance=0.0,
                    max_avoided_bodies=4,
                    disabled=True,
                )
                c.set_static_collision_config_for_direct_child_bodies(collision_config)

        apartment_world = (
            self.get_kitchen_world() if use_kitchen else self.get_apartment_world()
        )

        world.merge_world(apartment_world)
        world.get_body_by_name("base_footprint").parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                *([0, 0, 0] if use_kitchen else [1.5, 2, 0])
            )
        )

        robot = Tiago.from_world(world)
        for j in robot.joint_states:
            if j.state_type == "Park":  # or j.state_type == "Open":
                j.apply_to_world(world)

        return world

    def present_results(self, results, robot: AbstractRobot):
        for r in results:
            task, motion, effect = r.values()
            print("-" * 20)
            print(f"Task: {task}")
            print(f"Robot: {robot.name}")
            print(
                f"Effect: {effect.__class__.__name__}  for target: {effect.target_object.__class__.__name__}(name={effect.target_object.name} body={effect.target_object.body.name if isinstance(effect.target_object, Door) else effect.target_object.container.body.name} handle={effect.target_object.handle.body.name})"
            )

    def test_query_motion_satisfying_task_request2(self):
        world = self.get_world(use_kitchen=True)
        # if not rclpy.ok():
        #     rclpy.init()
        # node = rclpy.create_node("viz_node")
        # VizMarkerPublisher(world=world, node=node, throttle_state_updates=1)

        effects, motions, open_task, close_task, drawers = self._extend_world(
            world, only_doors=True
        )

        task_sym = variable(
            TaskRequest,
            domain=[TaskRequest(task_type="open", name="oven_area_oven_door")],
        )
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(effect=effect_sym, motion=motion_sym, environment=world)
        robot = PR2.from_world(world)
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
            names.append(
                effect.target_object.body.name.name
                if isinstance(effect.target_object, Door)
                else effect.target_object.container.body.name.name
            )
        diff = set(
            [
                (
                    a.target_object.body.name.name
                    if isinstance(a.target_object, Door)
                    else a.target_object.container.body.name.name
                )
                for a in effects
            ]
        ).difference(set(names))
        print(diff)
        print(len(diff))

    def test_query_motion_satisfying_task_request_stretch(self):
        world = self.get_stretch_world(use_kitchen=True)
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("viz_node")
        VizMarkerPublisher(world=world, node=node, throttle_state_updates=1)

        effects, motions, open_task, close_task, drawers = self._extend_world(
            world, only_doors=True
        )

        task_sym = variable(
            TaskRequest,
            domain=[TaskRequest(task_type="open", name="oven_area_oven_door")],
        )
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
        # assert len(results) == len(drawers)
        names = []
        for r in results:
            task, motion, effect = r.values()
            names.append(
                effect.target_object.body.name.name
                if isinstance(effect.target_object, Door)
                else effect.target_object.container.body.name.name
            )
        diff = set(
            [
                (
                    a.target_object.body.name.name
                    if isinstance(a.target_object, Door)
                    else a.target_object.container.body.name.name
                )
                for a in effects
            ]
        ).difference(set(names))
        print(diff)
        print(len(diff))

    def test_query_motion_satisfying_task_request_tiago(self):
        world = self.get_tiago_world(use_kitchen=True)
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("viz_node")
        VizMarkerPublisher(world=world, node=node, throttle_state_updates=30)

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
            names.append(
                effect.target_object.body.name.name
                if isinstance(effect.target_object, Door)
                else effect.target_object.container.body.name.name
            )
        diff = set(
            [
                (
                    a.target_object.body.name.name
                    if isinstance(a.target_object, Door)
                    else a.target_object.container.body.name.name
                )
                for a in effects
            ]
        ).difference(set(names))
        print(diff)
        print(len(diff))

    def test_query_task_and_effect_satisfying_motion_pr2(self):
        world = self.get_world()
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("viz_node")
        VizMarkerPublisher(world=world, node=node, throttle_state_updates=5)

        effects, _, open_task, close_task, drawers = self._extend_world(world)

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

    def test_what_needs_to_be_done(self):
        scene_path = os.path.join(
            "/home/malte/libs/semantic_digital_twin_demo/assets/apartment.xml"
        )
        world = MJCFParser(scene_path).parse()
        viewer = MultiverseViewer()

        box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.6, y=2, z=1, roll=0, pitch=0, yaw=0, reference_frame=world.root
        )
        box = Box(
            scale=Scale(1.0, 1.0, 2),
            color=Color(
                1.0,
                0.0,
                0.0,
                1.0,
            ),
        )
        collision = ShapeCollection([box])
        visual = ShapeCollection([box])
        body = Body(
            name=PrefixedName("my first body", "my first prefix"),
            visual=visual,
            collision=collision,
        )

        with world.modify_world():
            world.add_body(body)
            con = FixedConnection(
                parent=world.root, child=body, parent_T_connection_expression=box_origin
            )
            world.add_connection(con)

        with world.modify_world():
            world_reasoner = WorldReasoner(world)
            world_reasoner.reason()

        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("viz_node")
        VizMarkerPublisher(world=world, node=node)

        headless = (
            os.environ.get("CI", "false").lower() == "true"
        )  # headless in CI environments
        multi_sim = MujocoSim(
            world=world,
            viewer=viewer,
            headless=headless,
            step_size=5e-3,
            integrator="IMPLICITFAST",
        )
        multi_sim.start_simulation()

        time.sleep(1)
        # Initialize state feedback from simulator to world and perform on-demand sync
        sync = SimulatorToWorldStateSynchronizer(
            world=world, sim=multi_sim, poll_period_s=0.1
        )
        sync.initialize_subscriptions()

        for i in range(101):
            viewer.write_objects = {
                "cabinet11_drawer1_joint": {"joint_angular_position": [0.003 * i]},
            }
            time.sleep(0.1)

        # Pull the final simulator state into the world before asserting
        sync.synchronize_once()

        assert world.get_connection_by_name(
            "cabinet11_drawer1_joint"
        ).position == pytest.approx(0.3, abs=0.01)


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
    Convert a list of 3D points into a MarkerArray containing
    a single LINE_STRIP marker for RViz2.
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
