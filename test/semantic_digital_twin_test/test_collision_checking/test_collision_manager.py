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
)
from semantic_digital_twin.robots.pr2 import PR2


class TestCollisionRules:
    def test_AvoidCollisionBetweenGroups(self, pr2_world_state_reset):
        pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

        collision_matrix = CollisionMatrix()

        rule = AvoidCollisionBetweenGroups(
            buffer_zone_distance=0.05,
            violated_distance=0.0,
            body_group1=pr2.left_arm.bodies,
            body_group2=pr2.right_arm.bodies,
        )

        rule.apply_to_collision_matrix(collision_matrix)
        # -1 because torso is in both chains
        assert (
            len(collision_matrix.collision_checks)
            == len(pr2.left_arm.bodies) * len(pr2.right_arm.bodies) - 1
        )

    def test_AllowCollisionBetweenGroups(self, pr2_world_state_reset):
        pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]

        collision_matrix = CollisionMatrix()

        rule1 = AvoidCollisionBetweenGroups(
            buffer_zone_distance=0.05,
            violated_distance=0,
            body_group1=pr2.left_arm.bodies,
            body_group2=pr2.right_arm.bodies,
        )
        rule2 = AllowCollisionBetweenGroups(
            body_group1=pr2.left_arm.bodies, body_group2=pr2.right_arm.bodies
        )

        rule1.apply_to_collision_matrix(collision_matrix)
        rule2.apply_to_collision_matrix(collision_matrix)
        assert len(collision_matrix.collision_checks) == 0

    def test_AllowNonRobotCollisions(self, pr2_apartment_world):
        pr2 = pr2_apartment_world.get_semantic_annotations_by_type(PR2)[0]
        pr2_body1 = pr2.bodies_with_collisions[0]
        pr2_body2 = pr2.bodies_with_collisions[2]

        apartment_body1 = pr2_apartment_world.get_body_by_name("handle_cab3_door_top")
        apartment_body2 = pr2_apartment_world.get_body_by_name("cabinet6_drawer_top")

        collision_matrix = CollisionMatrix()
        avoid_all = AvoidAllCollisions(
            buffer_zone_distance=0.05,
            violated_distance=0,
            bodies=pr2_apartment_world.bodies_with_collision,
        )
        avoid_all.apply_to_collision_matrix(collision_matrix)
        # collisions between pr2 bodies and between apartment bodies should be avoided
        assert (
            CollisionCheck(body_a=apartment_body1, body_b=apartment_body2, distance=0.0)
            in collision_matrix.collision_checks
        )
        assert (
            CollisionCheck(body_a=pr2_body1, body_b=pr2_body2, distance=0.0)
            in collision_matrix.collision_checks
        )

        rule = AllowNonRobotCollisions()
        rule.update(pr2_apartment_world)
        rule.apply_to_collision_matrix(collision_matrix)
        # collisions between apartment bodies should be allowed
        assert (
            CollisionCheck(body_a=apartment_body1, body_b=apartment_body2, distance=0.0)
            not in collision_matrix.collision_checks
        )
        assert (
            CollisionCheck(body_a=pr2_body1, body_b=pr2_body2, distance=0.0)
            in collision_matrix.collision_checks
        )

    def test_pr2_collision_config(self, pr2_world_state_reset):
        pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]
        collision_manager = CollisionManager(pr2_world_state_reset)
        collision_manager.low_priority_rules.extend(pr2.default_collision_rules)
        collision_matrix = collision_manager.create_collision_matrix()
        pass
