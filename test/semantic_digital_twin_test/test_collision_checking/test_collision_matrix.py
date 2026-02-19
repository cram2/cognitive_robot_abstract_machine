from dataclasses import dataclass
from itertools import combinations

import pytest

from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
)
from semantic_digital_twin.collision_checking.collision_groups import (
    CollisionGroupConsumer,
)
from semantic_digital_twin.collision_checking.collision_manager import CollisionManager
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionMatrix,
    CollisionCheck,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
    AllowCollisionBetweenGroups,
    AllowNonRobotCollisions,
    AvoidAllCollisions,
    AllowCollisionRule,
    AvoidExternalCollisions,
    AllowSelfCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.collision_checking.pybullet_collision_detector import (
    BulletCollisionDetector,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


class TestCollisionRules:
    def test_get_distances(self, pr2_world_copy):
        with pr2_world_copy.modify_world():
            env = Body(
                name=PrefixedName("env"),
                collision=ShapeCollection([Sphere(radius=0.5)]),
            )
            root_C_env = FixedConnection(pr2_world_copy.root, env)
            pr2_world_copy.add_connection(root_C_env)

        base_link = pr2_world_copy.get_body_by_name("base_link")
        torso_lift_link = pr2_world_copy.get_body_by_name("torso_lift_link")
        head_pan_link = pr2_world_copy.get_body_by_name("head_pan_link")

        pr2 = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]

        collision_manager = CollisionManager(
            _world=pr2_world_copy,
            collision_detector=BulletCollisionDetector(_world=pr2_world_copy),
        )
        with pr2_world_copy.modify_world():
            collision_manager.add_default_rule(
                AvoidAllCollisions(
                    buffer_zone_distance=0.2,
                    violated_distance=0.05,
                )
            )
            collision_manager.add_default_rule(
                AvoidExternalCollisions(
                    buffer_zone_distance=0.1,
                    robot=pr2,
                    body_subset={torso_lift_link},
                )
            )
            collision_manager.add_default_rule(
                AvoidExternalCollisions(
                    buffer_zone_distance=0.05,
                    robot=pr2,
                    body_subset={head_pan_link},
                )
            )
        collision_manager.update_collision_matrix()
        # PR2 has a rule for base_link: buffer=0.2, violated=0.05
        # It's added to low_priority_rules
        assert collision_manager.get_buffer_zone_distance(base_link, env) == 0.2
        assert collision_manager.get_violated_distance(base_link, env) == 0.05

        # Test with a body that only has the general PR2 rule (buffer=0.1, violated=0.0)
        assert collision_manager.get_buffer_zone_distance(torso_lift_link, env) == 0.1
        assert collision_manager.get_violated_distance(torso_lift_link, env) == 0.0

        # Add a high priority rule to override
        override_rule = AvoidAllCollisions(
            buffer_zone_distance=0.5, violated_distance=0.1
        )
        with pr2_world_copy.modify_world():
            collision_manager.add_temporary_rule(override_rule)
        collision_manager.update_collision_matrix()
        assert collision_manager.get_buffer_zone_distance(base_link, env) == 0.5
        assert collision_manager.get_violated_distance(base_link, env) == 0.1

    def test_get_distances_no_rule(self, cylinder_bot_world):
        collision_manager = cylinder_bot_world.collision_manager

        body = cylinder_bot_world.bodies_with_collision[0]
        body2 = cylinder_bot_world.bodies_with_collision[1]

        collision_manager.default_rules = []
        collision_manager.temporary_rules = []
        collision_manager.ignore_collision_rules = []

        with pytest.raises(ValueError):
            collision_manager.get_buffer_zone_distance(body, body2)

    def test_AvoidCollisionBetweenGroups(self, pr2_world_state_reset):
        pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

        collision_matrix = CollisionMatrix()

        rule = AvoidCollisionBetweenGroups(
            buffer_zone_distance=0.05,
            violated_distance=0.0,
            body_group_a=pr2.left_arm.bodies,
            body_group_b=pr2.right_arm.bodies,
        )
        rule.update(pr2_world_state_reset)
        rule.apply_to_collision_matrix(collision_matrix)
        # -1 because torso is in both chains
        assert (
            len(collision_matrix.collision_checks)
            == len(pr2.left_arm.bodies) * len(pr2.right_arm.bodies) - 1
        )

    def test_AvoidCollisionBetweenGroups2(self, cylinder_bot_world):
        env1 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment")
        env2 = cylinder_bot_world.get_kinematic_structure_entity_by_name("environment2")
        robot = cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0]

        collision_manager = cylinder_bot_world.collision_manager
        collision_manager.temporary_rules.extend(
            [
                AvoidCollisionBetweenGroups(
                    buffer_zone_distance=10,
                    violated_distance=0.0,
                    body_group_a=[robot.root],
                    body_group_b=[env1],
                ),
                AvoidCollisionBetweenGroups(
                    buffer_zone_distance=15,
                    violated_distance=0.23,
                    body_group_a=[robot.root],
                    body_group_b=[env2],
                ),
            ]
        )

        collision_manager.update_collision_matrix()
        # -1 because torso is in both chains
        assert collision_manager.get_buffer_zone_distance(robot.root, env1) == 10
        assert collision_manager.get_buffer_zone_distance(robot.root, env2) == 15

    def test_AllowCollisionBetweenGroups(self, pr2_world_state_reset):
        pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

        collision_matrix = CollisionMatrix()

        rule1 = AvoidCollisionBetweenGroups(
            buffer_zone_distance=0.05,
            violated_distance=0,
            body_group_a=pr2.left_arm.bodies,
            body_group_b=pr2.right_arm.bodies,
        )
        rule2 = AllowCollisionBetweenGroups(
            body_group_a=pr2.left_arm.bodies, body_group_b=pr2.right_arm.bodies
        )

        rule1.apply_to_collision_matrix(collision_matrix)
        rule2.apply_to_collision_matrix(collision_matrix)
        assert len(collision_matrix.collision_checks) == 0

    def test_AllowNonRobotCollisions(self, pr2_apartment_world):
        pr2 = pr2_apartment_world.get_semantic_annotations_by_type(PR2)[0]
        pr2_body1 = pr2.bodies_with_collision[0]
        pr2_body2 = pr2.bodies_with_collision[2]

        apartment_body1 = pr2_apartment_world.get_body_by_name("handle_cab3_door_top")
        apartment_body2 = pr2_apartment_world.get_body_by_name("cabinet6_drawer_top")

        collision_matrix = CollisionMatrix()
        avoid_all = AvoidAllCollisions(
            buffer_zone_distance=0.05,
            violated_distance=0,
        )
        avoid_all.update(pr2_apartment_world)
        avoid_all.apply_to_collision_matrix(collision_matrix)
        # collisions between pr2 bodies and between apartment bodies should be avoided
        assert (
            CollisionCheck.create_and_validate(
                body_a=apartment_body1, body_b=apartment_body2, distance=0.0
            )
            in collision_matrix.collision_checks
        )
        assert (
            CollisionCheck.create_and_validate(
                body_a=pr2_body1, body_b=pr2_body2, distance=0.0
            )
            in collision_matrix.collision_checks
        )

        rule = AllowNonRobotCollisions()
        rule._update(pr2_apartment_world)
        rule.apply_to_collision_matrix(collision_matrix)
        # collisions between apartment bodies should be allowed
        assert (
            CollisionCheck.create_and_validate(
                body_a=apartment_body1, body_b=apartment_body2, distance=0.0
            )
            not in collision_matrix.collision_checks
        )
        assert (
            CollisionCheck.create_and_validate(
                body_a=pr2_body1, body_b=pr2_body2, distance=0.0
            )
            in collision_matrix.collision_checks
        )

    def test_AvoidExternalCollisions(self, pr2_apartment_world):
        pr2 = pr2_apartment_world.get_semantic_annotations_by_type(PR2)[0]
        collision_matrix = CollisionMatrix()
        rule = AvoidExternalCollisions(
            buffer_zone_distance=1, violated_distance=0.1, robot=pr2
        )
        rule.update(pr2_apartment_world)
        rule.apply_to_collision_matrix(collision_matrix)
        pr2_bodies = set(pr2.bodies_with_collision)
        for collision_check in collision_matrix.collision_checks:
            body_a_is_robot = collision_check.body_a in pr2_bodies
            body_b_is_robot = collision_check.body_b in pr2_bodies
            assert (
                body_a_is_robot
                and not body_b_is_robot
                or not body_a_is_robot
                and body_b_is_robot
            )

    def test_pr2_collision_config(self, pr2_world_state_reset):
        pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]
        collision_manager = pr2_world_state_reset.collision_manager
        collision_manager.update_collision_matrix()
        collision_matrix = collision_manager.collision_matrix
        rule: AllowCollisionRule
        for rule in collision_manager.ignore_collision_rules:
            assert (
                rule.allowed_collision_pairs & collision_matrix.collision_checks
                == set()
            )
            for check in collision_matrix.collision_checks:
                assert check.body_a not in rule.allowed_collision_bodies
                assert check.body_b not in rule.allowed_collision_bodies

        assert len(collision_matrix.collision_checks) > 0
        assert (
            collision_manager.get_max_avoided_bodies(
                pr2_world_state_reset.get_body_by_name("base_link")
            )
            == 2
        )
        assert (
            collision_manager.get_max_avoided_bodies(
                pr2_world_state_reset.get_body_by_name("torso_lift_link")
            )
            == 1
        )
        assert (
            collision_manager.get_max_avoided_bodies(
                pr2_world_state_reset.get_body_by_name("r_gripper_palm_link")
            )
            == 4
        )

    def test_AllowCollisionForAdjacentPairs(self, pr2_world_state_reset):
        pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]
        collision_manager = pr2_world_state_reset.collision_manager
        collision_manager.update_collision_matrix()
        expected_collision_matrix = collision_manager.collision_matrix

        hard_ware_interface_cache = {}

        with pr2_world_state_reset.modify_world():
            for dof in pr2_world_state_reset.degrees_of_freedom:
                hard_ware_interface_cache[dof.name] = dof.has_hardware_interface
                dof.has_hardware_interface = False

        collision_manager.update_collision_matrix()
        empty_collision_matrix = collision_manager.collision_matrix
        assert len(empty_collision_matrix.collision_checks) == 0

        with pr2_world_state_reset.modify_world():
            for dof in pr2_world_state_reset.degrees_of_freedom:
                dof.has_hardware_interface = hard_ware_interface_cache[dof.name]

        collision_manager.update_collision_matrix()
        assert collision_manager.collision_matrix == expected_collision_matrix

    def test_allow_self_collision(self, pr2_apartment_world):
        robot = pr2_apartment_world.get_semantic_annotations_by_type(PR2)[0]
        collision_matrix = CollisionMatrix()
        rule1 = AvoidAllCollisions()
        rule1._update(pr2_apartment_world)
        rule1.apply_to_collision_matrix(collision_matrix)
        number_of_all_checks = len(collision_matrix.collision_checks)
        allow_self_collision = AllowSelfCollisions(robot=robot)
        allow_self_collision._update(pr2_apartment_world)
        avoid_self_collision = AvoidSelfCollisions(robot=robot)
        avoid_self_collision._update(pr2_apartment_world)

        allow_self_collision.apply_to_collision_matrix(collision_matrix)
        number_after_allow_self_collision = len(collision_matrix.collision_checks)
        assert number_after_allow_self_collision < number_of_all_checks
        avoid_self_collision.apply_to_collision_matrix(collision_matrix)
        assert len(collision_matrix.collision_checks) == number_of_all_checks


class TestCollisionGroups:

    @dataclass
    class MockCollisionGroupConsumer(CollisionGroupConsumer):
        def on_compute_collisions(self, collision_results: CollisionCheckingResult): ...
        def on_collision_matrix_update(self): ...

    @pytest.mark.parametrize(
        "fix_name", ["pr2_world_state_reset", "cylinder_bot_world"]
    )
    def test_collision_groups(self, fix_name, request):
        world = request.getfixturevalue(fix_name)
        robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
        collision_manager = world.collision_manager
        collision_manager.collision_consumers = [
            collision_group_consumer := self.MockCollisionGroupConsumer()
        ]
        world._notify_model_change()

        # there should be groups
        assert len(collision_group_consumer.collision_groups) > 0

        # there should be fewer groups than bodies with collisions
        assert len(collision_group_consumer.collision_groups) <= len(
            world.bodies_with_collision
        )

        # no group should be in the bodies of another group
        for group1, group2 in combinations(
            collision_group_consumer.collision_groups, 2
        ):
            assert group1.root not in group2.bodies
            assert group2.root not in group1.bodies

        # no group should be empty if the root has no collision
        for group in collision_group_consumer.collision_groups:
            try:
                assert (
                    len(group.bodies) > 0 or group.root in robot.bodies_with_collision
                )
            except AssertionError:
                pass

        # the parent connection of every group is controlled, unless it's the root group
        for group in collision_group_consumer.collision_groups:
            if group.root == world.root:
                continue
            assert (
                group.root.parent_connection.is_controlled
            ), f"parent of group root {group.root.name} is not controlled"
            for body in group.bodies:
                assert not body.parent_connection.is_controlled

        # no group body should be in another group body
        for group1, group2 in combinations(
            collision_group_consumer.collision_groups, 2
        ):
            for body1 in group1.bodies:
                for body2 in group2.bodies:
                    assert body1 != body2

        # ever body with a collision should be in a group
        for body in robot.bodies_with_collision:
            collision_group_consumer.get_collision_group(body)
