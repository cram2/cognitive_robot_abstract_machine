from itertools import combinations

import pytest
import urdf_parser_py.urdf as up
from rustworkx import NoPathFound

import semantic_world.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.middleware import set_middleware
from giskardpy.model.utils import hacky_urdf_parser_fix
from giskardpy.model.world_config import EmptyWorld, WorldWithOmniDriveRobot
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.user_interface import GiskardWrapper
from giskardpy.utils.utils import suppress_stderr
from semantic_world.connections import FixedConnection, PrismaticConnection, OmniDrive, ActiveConnection, \
    RevoluteConnection
from semantic_world.geometry import Box, Scale, Color
from semantic_world.prefixed_name import PrefixedName
from semantic_world.robots import AbstractRobot, Manipulator
from semantic_world.world import World
from semantic_world.world_entity import Body


try:
    import rospy
    from giskardpy_ros1.ros1.ros_msg_visualization import ROSMsgVisualization

    rospy.init_node('tests')
    vis = ROSMsgVisualization(tf_frame='map')
    rospy.sleep(1)
except ImportError as e:
    pass

try:
    from giskardpy_ros.ros2 import rospy
    from giskardpy_ros.ros2.rospy import ROS2Wrapper

    set_middleware(ROS2Wrapper())
    rospy.init_node('giskard')
except ImportError as e:
    pass


def visualize():
    god_map.world.notify_state_change()
    god_map.collision_scene.sync()
    # vis.publish_markers()


@pytest.fixture()
def empty_world() -> World:
    config = EmptyWorld()
    config.setup()
    god_map.tmp_folder = 'tmp'
    return config.world


@pytest.fixture()
def fixed_box_world() -> World:
    class WorldWithFixedBox(EmptyWorld):
        box_name = PrefixedName('box')
        joint_name = PrefixedName('box_joint')

        def setup(self) -> None:
            super().setup()
            with self.world.modify_world():
                box_geometry = Box(scale=Scale(1, 1, 1), color=Color(1, 0, 0, 1))
                box = Body(name=self.box_name, visual=[box_geometry], collision=[box_geometry])
                connection = FixedConnection(parent=self.world.root, child=box)
                # self.world.add_body(box)
                self.world.add_connection(connection)

    config = WorldWithFixedBox()
    # collision_avoidance = DefaultCollisionAvoidanceConfig()
    config.setup()
    # collision_avoidance.setup()
    return config.world


@pytest.fixture()
def box_world_prismatic() -> World:
    class WorldWithPrismaticBox(EmptyWorld):
        box_name = PrefixedName('box')

        def setup(self) -> None:
            super().setup()
            with self.world.modify_world():
                box_geometry = Box(scale=Scale(1, 1, 1), color=Color(1, 0, 0, 1))
                box = Body(name=self.box_name, visual=[box_geometry], collision=[box_geometry])
                joint = PrismaticConnection(parent=self.world.root, child=box, axis=cas.UnitVector3.X(),
                                            _world=self.world)
                joint.dof.lower_limits.position = -1
                joint.dof.lower_limits.velocity = -1
                joint.dof.upper_limits.position = 1
                joint.dof.upper_limits.velocity = 1
                self.world.add_connection(joint)

    config = WorldWithPrismaticBox()
    # collision_avoidance = DefaultCollisionAvoidanceConfig()
    config.setup()
    # collision_avoidance.setup()
    return config.world


@pytest.fixture()
def box_world():
    class WorldWithOmniBox(EmptyWorld):
        box_name = PrefixedName('box')
        joint_name = PrefixedName('box_joint')

        def setup(self) -> None:
            super().setup()
            with self.world.modify_world():
                box = Body(self.box_name)
                box_geometry = Box(scale=Scale(1, 1, 1), color=Color(1, 0, 0, 1))
                box.collision.append(box_geometry)
                box.visual.append(box_geometry)
                self.world.add_body(box)
                joint = OmniDrive(parent=self.world.root, child=box,
                                  translation_velocity_limits=1,
                                  rotation_velocity_limits=1)
                self.world.add_connection(joint)

    config = WorldWithOmniBox()
    collision_avoidance = DefaultCollisionAvoidanceConfig()
    config.setup()
    collision_avoidance.setup()
    assert config.box_name in config.world.bodies
    return config.world


@pytest.fixture()
def simple_two_arm_world() -> World:
    urdf = open('urdfs/simple_two_arm_robot.urdf', 'r').read()
    config = WorldWithOmniDriveRobot(urdf)
    with config.world.modify_world():
        config.setup()
    # todo move to controller
    # controlled_joints = ControlledConnections(config.world.search_for_connections_of_type(ActiveConnection))
    # config.world.add_view(controlled_joints)
    # config.world.register_controlled_joints(config.world.movable_joint_names)
    # collision_avoidance = DefaultCollisionAvoidanceConfig()
    # collision_avoidance.setup()
    return config.world


@pytest.fixture()
def pr2_world() -> World:
    urdf = open('urdfs/pr2.urdf', 'r').read()
    config = WorldWithOmniDriveRobot(urdf=urdf)
    with config.world.modify_world():
        config.setup()
    pr2: AbstractRobot = config.world.get_view_by_name('robot')
    pr2.controlled_connections.connections = config.world.search_for_connections_of_type(ActiveConnection)
    return config.world


@pytest.fixture()
def giskard_pr2() -> GiskardWrapper:
    urdf = open('urdfs/pr2.urdf', 'r').read()
    giskard = GiskardWrapper(world_config=WorldWithOmniDriveRobot(urdf=urdf),
                             collision_avoidance_config=DisableCollisionAvoidanceConfig(),
                             qp_controller_config=QPControllerConfig())
    return giskard


class TestWorld:
    def test_empty_world(self, empty_world: World):
        assert len(empty_world.connections) == 0

    def test_fixed_box_world(self, fixed_box_world: World):
        assert len(fixed_box_world.connections) == 1
        assert len(fixed_box_world.bodies) == 2
        # visualize()

    def test_simple_two_arm_robot(self, simple_two_arm_world: World):
        simple_two_arm_world.state[PrefixedName('prismatic_joint', 'muh')].position = 0.4
        simple_two_arm_world.state[PrefixedName('r_joint_1', 'muh')].position = 0.2
        simple_two_arm_world.state[PrefixedName('r_joint_2', 'muh')].position = 0.2
        simple_two_arm_world.state[PrefixedName('r_joint_3', 'muh')].position = 0.2
        simple_two_arm_world.state[PrefixedName('l_joint_1', 'muh')].position = 0.2
        simple_two_arm_world.state[PrefixedName('l_joint_2', 'muh')].position = 0.2
        simple_two_arm_world.state[PrefixedName('l_joint_3', 'muh')].position = 0.2
        # visualize()

    def test_compute_fk(self, box_world_prismatic: World):
        connection = box_world_prismatic.connections[0]
        box = box_world_prismatic.bodies[-1]

        box_world_prismatic.state[connection.dof.name].position = 1
        box_world_prismatic.notify_state_change()
        fk = box_world_prismatic.compute_forward_kinematics_np(root=box_world_prismatic.root, tip=box)
        assert fk[0, 3] == 1
        # visualize()

    # def test_compute_self_collision_matrix(self, pr2_world: World):
    #     disabled_links = {pr2_world.search_for_link_name('br_caster_l_wheel_link'),
    #                       pr2_world.search_for_link_name('fr_caster_l_wheel_link')}
    #     reference_collision_scene = BetterPyBulletSyncer()
    #     reference_collision_scene.load_self_collision_matrix_from_srdf(
    #         'data/pr2_test.srdf', 'pr2')
    #     reference_reasons = reference_collision_scene.self_collision_matrix
    #     reference_disabled_links = reference_collision_scene.disabled_links
    #     collision_scene: CollisionWorldSynchronizer = god_map.collision_scene
    #     actual_reasons = collision_scene.compute_self_collision_matrix('pr2',
    #                                                                    number_of_tries_never=500)
    #     assert actual_reasons == reference_reasons
    #     assert reference_disabled_links == disabled_links

    def test_compute_chain_reduced_to_controlled_joints(self, simple_two_arm_world: World):
        root = simple_two_arm_world.get_body_by_name('r_eef')
        tip = simple_two_arm_world.get_body_by_name('l_eef')
        controlled_joints: ControlledConnections = simple_two_arm_world.views[1]
        link_a, link_b = controlled_joints.compute_chain_reduced_to_controlled_joints(root, tip)
        assert link_a == simple_two_arm_world.get_body_by_name('r_eef')
        assert link_b == simple_two_arm_world.get_body_by_name('l_link_3')

        tip = simple_two_arm_world.get_body_by_name('r_link_1')
        link_a, link_b = controlled_joints.compute_chain_reduced_to_controlled_joints(root, tip)
        assert link_a == simple_two_arm_world.get_body_by_name('r_eef')
        assert link_b == simple_two_arm_world.get_body_by_name('r_link_1')

    def test_group_pr2_hand(self, pr2_world: World):
        pr2: AbstractRobot = pr2_world.get_view_by_name('robot')
        pr2.manipulators.append(Manipulator(root=pr2_world.get_body_by_name('r_wrist_roll_link'),
                                            name=PrefixedName('r_hand'),
                                            tool_frame=pr2_world.get_body_by_name('r_gripper_tool_frame'),
                                            _world=pr2_world))
        view: AbstractRobot = pr2.manipulators[0]
        assert set(view.connections) == {
            pr2_world.get_connection_by_name('r_gripper_palm_joint'),
            pr2_world.get_connection_by_name('r_gripper_led_joint'),
            pr2_world.get_connection_by_name('r_gripper_motor_accelerometer_joint'),
            pr2_world.get_connection_by_name('r_gripper_tool_joint'),
            pr2_world.get_connection_by_name('r_gripper_motor_slider_joint'),
            pr2_world.get_connection_by_name('r_gripper_l_finger_joint'),
            pr2_world.get_connection_by_name('r_gripper_r_finger_joint'),
            pr2_world.get_connection_by_name('r_gripper_motor_screw_joint'),
            pr2_world.get_connection_by_name('r_gripper_l_finger_tip_joint'),
            pr2_world.get_connection_by_name('r_gripper_r_finger_tip_joint'),
            pr2_world.get_connection_by_name('r_gripper_joint')}
        assert set(view.bodies) == {
            pr2_world.get_body_by_name('r_wrist_roll_link'),
            pr2_world.get_body_by_name('r_gripper_palm_link'),
            pr2_world.get_body_by_name('r_gripper_led_frame'),
            pr2_world.get_body_by_name('r_gripper_motor_accelerometer_link'),
            pr2_world.get_body_by_name('r_gripper_tool_frame'),
            pr2_world.get_body_by_name('r_gripper_motor_slider_link'),
            pr2_world.get_body_by_name('r_gripper_motor_screw_link'),
            pr2_world.get_body_by_name('r_gripper_l_finger_link'),
            pr2_world.get_body_by_name('r_gripper_l_finger_tip_link'),
            pr2_world.get_body_by_name('r_gripper_r_finger_link'),
            pr2_world.get_body_by_name('r_gripper_r_finger_tip_link'),
            pr2_world.get_body_by_name('r_gripper_l_finger_tip_frame')}

    def test_compute_chain_of_connections(self, pr2_world: World):
        with suppress_stderr():
            urdf = open('urdfs/pr2.urdf', 'r').read()
            parsed_urdf = up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))

        root_link = 'base_footprint'
        tip_link = 'r_gripper_tool_frame'
        real = pr2_world.compute_chain_of_connections(root=pr2_world.get_body_by_name(root_link),
                                                      tip=pr2_world.get_body_by_name(tip_link))
        expected = parsed_urdf.get_chain(root_link, tip_link, joints=True, links=False, fixed=True)
        assert {x.name.name for x in real} == set(expected)

    def test_compute_chain_of_bodies(self, pr2_world: World):
        with suppress_stderr():
            urdf = open('urdfs/pr2.urdf', 'r').read()
            parsed_urdf = up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))

        root_link = 'base_footprint'
        tip_link = 'r_gripper_tool_frame'
        real = pr2_world.compute_chain_of_bodies(root=pr2_world.get_body_by_name(root_link),
                                                 tip=pr2_world.get_body_by_name(tip_link))
        expected = parsed_urdf.get_chain(root_link, tip_link, joints=False, links=True, fixed=False)
        assert {x.name.name for x in real} == set(expected)

    def test_get_chain2(self, pr2_world: World):
        root_link = pr2_world.get_body_by_name('l_gripper_tool_frame')
        tip_link = pr2_world.get_body_by_name('r_gripper_tool_frame')
        with pytest.raises(NoPathFound):
            pr2_world.compute_chain_of_connections(root_link, tip_link)

    def test_get_split_chain(self, pr2_world: World):
        root_link = pr2_world.get_body_by_name('l_gripper_r_finger_tip_link')
        tip_link = pr2_world.get_body_by_name('l_gripper_l_finger_tip_link')
        chain1, connection, chain2 = pr2_world.compute_split_chain_of_bodies(root_link, tip_link)
        chain1 = [n.name.name for n in chain1]
        connection = [n.name.name for n in connection]
        chain2 = [n.name.name for n in chain2]
        assert chain1 == ['l_gripper_r_finger_tip_link', 'l_gripper_r_finger_link']
        assert connection == ['l_gripper_palm_link']
        assert chain2 == ['l_gripper_l_finger_link', 'l_gripper_l_finger_tip_link']

    def test_get_joint_limits2(self, pr2_world: World):
        c: RevoluteConnection = pr2_world.get_connection_by_name('l_shoulder_pan_joint')
        assert c.dof.lower_limits.position == -0.564601836603
        assert c.dof.upper_limits.position == 2.1353981634

    # @pytest.skip(reason='not sure how to implement yet')
    # def test_get_controlled_parent_joint_of_link(self, pr2_world: World):
    #     with pytest.raises(KeyError) as e_info:
    #         pr2_world.get_controlled_parent_joint_of_link(pr2_world.search_for_link_name('odom_combined'))
    #     assert pr2_world.get_controlled_parent_joint_of_link(
    #         pr2_world.search_for_link_name('base_footprint')) == 'pr2/brumbrum'
    #
    # @pytest.skip(reason='not sure how to implement yet')
    # def test_get_parent_joint_of_joint(self, pr2_world: World):
    #     # TODO shouldn't this return a not found error?
    #     with pytest.raises(KeyError) as e_info:
    #         pr2_world.get_controlled_parent_joint_of_joint(PrefixedName('brumbrum', 'pr2'))
    #     with pytest.raises(KeyError) as e_info:
    #         pr2_world.search_for_parent_joint(pr2_world.search_for_joint_name('r_wrist_roll_joint'),
    #                                           stop_when=lambda x: False)
    #     assert pr2_world.get_controlled_parent_joint_of_joint(
    #         pr2_world.search_for_joint_name('r_torso_lift_side_plate_joint')) == 'pr2/torso_lift_joint'
    #     assert pr2_world.get_controlled_parent_joint_of_joint(
    #         pr2_world.search_for_joint_name('torso_lift_joint')) == 'pr2/brumbrum'
    #
    # @pytest.skip(reason='not sure how to implement yet')
    # def test_possible_collision_combinations(self, pr2_world: World):
    #     result = pr2_world.groups[pr2_world.robot_names[0]].possible_collision_combinations()
    #     reference = {pr2_world.sort_links(link_a, link_b) for link_a, link_b in
    #                  combinations(pr2_world.groups[pr2_world.robot_names[0]].link_names_with_collisions, 2) if
    #                  not pr2_world.are_linked(link_a, link_b)}
    #     assert result == reference
    #
    # @pytest.skip(reason='not sure how to implement yet')
    # def test_compute_chain_reduced_to_controlled_joints2(self, pr2_world: World):
    #     link_a, link_b = pr2_world.compute_chain_reduced_to_controlled_joints(
    #         pr2_world.search_for_link_name('l_upper_arm_link'),
    #         pr2_world.search_for_link_name('r_upper_arm_link'))
    #     assert link_a == 'pr2/l_upper_arm_roll_link'
    #     assert link_b == 'pr2/r_upper_arm_roll_link'
    #
    # @pytest.skip(reason='not sure how to implement yet')
    # def test_compute_chain_reduced_to_controlled_joints3(self, pr2_world: World):
    #     with pytest.raises(KeyError):
    #         pr2_world.compute_chain_reduced_to_controlled_joints(
    #             pr2_world.search_for_link_name('l_wrist_roll_link'),
    #             pr2_world.search_for_link_name('l_gripper_r_finger_link'))


class TestController:
    def test_joint_goal(self, giskard_pr2: GiskardWrapper):
        init = 'init'
        g1 = 'g1'
        g2 = 'g2'
        giskard_pr2.monitors.add_set_seed_configuration(seed_configuration={'r_wrist_roll_joint': 2},
                                                        name=init)
        giskard_pr2.motion_goals.add_joint_position({'r_wrist_roll_joint': -1}, name=g1,
                                                    start_condition=init,
                                                    end_condition=g1)
        giskard_pr2.motion_goals.add_joint_position({'r_wrist_roll_joint': 1}, name=g2,
                                                    start_condition=g1)
        giskard_pr2.monitors.add_end_motion(start_condition=g2)
        giskard_pr2.execute()

    def test_cart_goal(self, giskard_pr2: GiskardWrapper):
        init = 'init'
        g1 = 'g1'
        g2 = 'g2'
        root = god_map.world.get_body_by_name('map')
        init_goal1 = cas.TransformationMatrix(reference_frame=root)
        init_goal1.x = -0.5

        base_goal1 = cas.TransformationMatrix(reference_frame=root)
        base_goal1.x = 1.0

        base_goal2 = cas.TransformationMatrix(reference_frame=root)
        base_goal2.x = -1.0

        giskard_pr2.monitors.add_set_seed_odometry(base_pose=init_goal1, name=init)
        giskard_pr2.motion_goals.add_cartesian_pose(goal_pose=base_goal1, name=g1,
                                                    root_link='map',
                                                    tip_link='base_footprint',
                                                    start_condition=init,
                                                    end_condition=g1)
        giskard_pr2.motion_goals.add_cartesian_pose(goal_pose=base_goal2, name=g2,
                                                    root_link='map',
                                                    tip_link='base_footprint',
                                                    start_condition=g1)
        giskard_pr2.monitors.add_end_motion(start_condition=g2)
        giskard_pr2.execute(sim_time=20)

# import pytest
# pytest.main(['-s', __file__ + '::TestController::test_joint_goal_pr2'])
