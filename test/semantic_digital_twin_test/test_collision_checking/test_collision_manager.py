import numpy as np

from krrood.symbolic_math.float_variable_data import (
    FloatVariableData,
)
from krrood.symbolic_math.symbolic_math import Vector, VariableParameters, FloatVariable
from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
    AvoidSelfCollisions,
)
from semantic_digital_twin.collision_checking.collision_variable_managers import (
    ExternalCollisionVariableManager,
    SelfCollisionVariableManager,
)
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World


class TestExternalCollisionExpressionManager:
    def test_simple(self, cylinder_bot_world):
        float_variable_data = FloatVariableData()
        float_variable_data.add_variable(FloatVariable("muh"))

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
        collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(2, {robot.root})
        )
        collision_manager.add_collision_consumer(
            external_collisions := ExternalCollisionVariableManager(float_variable_data)
        )
        external_collisions.register_group_of_body(robot.root)
        collision_manager.update_collision_matrix()
        collisions = collision_manager.compute_collisions()

        group = external_collisions.get_collision_group(robot.root)

        # test point on a
        point1 = external_collisions.get_group_a_P_point_on_a_symbol(group, 0)
        assert np.allclose(point1.evaluate(), np.array([0.0, 0.05, 0.499, 1.0]))
        point2 = external_collisions.get_group_a_P_point_on_a_symbol(group, 1)
        assert np.allclose(
            point2.evaluate(), np.array([0.05, 0.0, 0.499, 1.0]), atol=1e-4
        )

        # test contact normal
        contact_normal1 = external_collisions.get_root_V_contact_normal_symbol(group, 0)
        assert np.allclose(contact_normal1.evaluate(), np.array([0.0, -1.0, 0.0, 0.0]))
        contact_normal2 = external_collisions.get_root_V_contact_normal_symbol(group, 1)
        assert np.allclose(
            contact_normal2.evaluate(), np.array([-1, 0.0, 0.0, 0.0]), atol=1e-4
        )

        # test buffer distance
        buffer_distance1 = external_collisions.get_buffer_distance_symbol(group, 0)
        assert np.allclose(buffer_distance1.evaluate()[0], 15)
        buffer_distance2 = external_collisions.get_buffer_distance_symbol(group, 1)
        assert np.allclose(buffer_distance2.evaluate()[0], 10)

        # test contact distance
        contact_distance1 = external_collisions.get_contact_distance_symbol(group, 0)
        assert np.allclose(contact_distance1.evaluate()[0], 0.2)
        contact_distance2 = external_collisions.get_contact_distance_symbol(group, 1)
        assert np.allclose(contact_distance2.evaluate()[0], 0.7)

        # test violated distance
        violated_distance1 = external_collisions.get_violated_distance_symbol(group, 0)
        assert np.allclose(violated_distance1.evaluate()[0], 0.23)
        violated_distance2 = external_collisions.get_violated_distance_symbol(group, 1)
        assert np.allclose(violated_distance2.evaluate()[0], 0.0)

        # test full expr
        variables = external_collisions.float_variable_data.variables
        assert len(variables) == external_collisions.block_size * 2 + 1
        expression = Vector(variables)
        compiled_expression = expression.compile(
            VariableParameters.from_lists(variables)
        )
        result = compiled_expression(external_collisions.float_variable_data.data)
        assert np.allclose(result, external_collisions.float_variable_data.data)

        # test specific expression
        group_a_P_point_on_a = external_collisions.get_group_a_P_point_on_a_symbol(
            group, 0
        )
        root_b_V_contact_normal = external_collisions.get_root_V_contact_normal_symbol(
            group, 0
        )
        expr = root_b_V_contact_normal @ group_a_P_point_on_a.to_vector3()
        compiled_expression = expr.compile(VariableParameters.from_lists(variables))
        result = compiled_expression(external_collisions.float_variable_data.data)
        expected = (
            external_collisions.get_root_V_contact_normal_symbol(group, 0).evaluate()
            @ external_collisions.get_group_a_P_point_on_a_symbol(group, 0).evaluate()
        )
        assert np.allclose(result, expected)


class TestSelfCollisionExpressionManager:
    def test_simple(self, self_collision_bot_world):
        float_variable_data = FloatVariableData()
        float_variable_data.add_variable(FloatVariable("muh"))

        r_tip = self_collision_bot_world.get_kinematic_structure_entity_by_name("r_tip")
        l_tip = self_collision_bot_world.get_kinematic_structure_entity_by_name("l_tip")

        robot = self_collision_bot_world.get_semantic_annotations_by_type(MinimalRobot)[
            0
        ]
        collision_manager = self_collision_bot_world.collision_manager
        collision_manager.temporary_rules.extend(
            [
                AvoidSelfCollisions(
                    buffer_zone_distance=10,
                    violated_distance=0.23,
                    robot=robot,
                ),
            ]
        )
        collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(2, {robot.root})
        )
        collision_manager.add_collision_consumer(
            self_collisions := SelfCollisionVariableManager(float_variable_data)
        )
        self_collisions.register_groups_of_body_combination(l_tip, r_tip)
        assert len(self_collisions.registered_group_combinations) == 1
        group_a, group_b = list(self_collisions.registered_group_combinations.keys())[0]

        collision_manager.update_collision_matrix()
        collisions = collision_manager.compute_collisions()

        expected_l_tip_point = np.array([0.0, -0.15, 0.1, 1.0])
        expected_r_tip_point = np.array([0.0, 0.15, 0.1, 1.0])
        expected_normal = np.array([0.0, 1.0, 0.0, 0.0])

        if r_tip.id < l_tip.id:
            expected_body_a_point = expected_r_tip_point
            expected_body_b_point = expected_l_tip_point
            expected_normal *= -1
        else:
            expected_body_a_point = expected_l_tip_point
            expected_body_b_point = expected_r_tip_point

        # test point on a
        point1 = self_collisions.get_group_a_P_point_on_a_symbol(
            group_a, group_b
        ).evaluate()
        assert np.allclose(point1, expected_body_a_point)
        point2 = self_collisions.get_group_b_P_point_on_b_symbol(
            group_a, group_b
        ).evaluate()
        assert np.allclose(point2, expected_body_b_point)

        # test contact normal
        contact_normal1 = self_collisions.get_group_b_V_contact_normal_symbol(
            group_a, group_b
        )
        assert np.allclose(contact_normal1.evaluate(), expected_normal)

        # test buffer distance
        buffer_distance1 = self_collisions.get_buffer_distance_symbol(group_a, group_b)
        assert np.allclose(buffer_distance1.evaluate()[0], 10)

        # test contact distance
        contact_distance1 = self_collisions.get_contact_distance_symbol(
            group_a, group_b
        )
        assert np.allclose(contact_distance1.evaluate()[0], 0.1)

        # test violated distance
        violated_distance1 = self_collisions.get_violated_distance_symbol(
            group_a, group_b
        )
        assert np.allclose(violated_distance1.evaluate()[0], 0.23)

        # test full expr
        variables = self_collisions.float_variable_data.variables
        assert len(variables) == self_collisions.block_size + 1
        expression = Vector(variables)
        compiled_expression = expression.compile(
            VariableParameters.from_lists(variables)
        )
        result = compiled_expression(self_collisions.float_variable_data.data)
        assert np.allclose(result, self_collisions.float_variable_data.data)

        # test specific expression
        group_b_P_point_on_b = self_collisions.get_group_b_P_point_on_b_symbol(
            group_a, group_b
        )
        group_b_V_contact_normal = self_collisions.get_group_b_V_contact_normal_symbol(
            group_a, group_b
        )
        expr = group_b_V_contact_normal @ group_b_P_point_on_b.to_vector3()
        compiled_expression = expr.compile(VariableParameters.from_lists(variables))
        result = compiled_expression(self_collisions.float_variable_data.data)
        expected = expr.evaluate()
        assert np.allclose(result, expected)

    def test_reset(self, self_collision_bot_world):
        float_variable_data = FloatVariableData()
        float_variable_data.add_variable(FloatVariable("muh"))

        r_tip = self_collision_bot_world.get_kinematic_structure_entity_by_name("r_tip")
        l_tip = self_collision_bot_world.get_kinematic_structure_entity_by_name("l_tip")
        robot = self_collision_bot_world.get_semantic_annotations_by_type(MinimalRobot)[
            0
        ]
        root = robot.root
        collision_manager = self_collision_bot_world.collision_manager
        collision_manager.temporary_rules.extend(
            [
                AvoidSelfCollisions(
                    buffer_zone_distance=10,
                    violated_distance=0.23,
                    robot=robot,
                ),
            ]
        )
        collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(2, {robot.root})
        )
        collision_manager.add_collision_consumer(
            self_collisions := SelfCollisionVariableManager(float_variable_data)
        )
        self_collisions.register_groups_of_body_combination(root, r_tip)
        # insert a variable between the collision entries to possibly mess with the internal indexing
        v = FloatVariable("muh2")
        index = float_variable_data.add_variable(v)
        float_variable_data.set_value(index, 23)
        self_collisions.register_groups_of_body_combination(l_tip, r_tip)

        self_collisions.reset_collision_data()
        # other data should not get overwritten
        assert v.evaluate() == 23

        group_a, group_b = list(self_collisions.registered_group_combinations.keys())[1]

        contact_distance1 = self_collisions.get_contact_distance_symbol(
            group_a, group_b
        )
        assert np.allclose(contact_distance1.evaluate()[0], 100)


def test_collision_rules_survive_merge(pr2_world_copy):
    expected = len(pr2_world_copy.collision_manager.rules)
    world = World()
    with world.modify_world():
        world.merge_world(pr2_world_copy)
    assert len(world.collision_manager.rules) == expected
