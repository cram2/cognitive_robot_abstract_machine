import numpy as np

from giskardpy.motion_statechart.variable_managers.collision_expressions import (
    ExternalCollisionVariableManager,
)
from giskardpy.motion_statechart.variable_managers.float_variable_manager import (
    FloatVariableData,
)
from krrood.symbolic_math.symbolic_math import Vector, VariableParameters, FloatVariable
from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.robots.minimal_robot import MinimalRobot


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
                    body_group1=[robot.root],
                    body_group2=[env1],
                ),
                AvoidCollisionBetweenGroups(
                    buffer_zone_distance=15,
                    violated_distance=0.23,
                    body_group1=[robot.root],
                    body_group2=[env2],
                ),
            ]
        )
        collision_manager.max_avoided_bodies_rules.append(
            MaxAvoidedCollisionsOverride(2, {robot.root})
        )
        collision_manager.add_collision_consumer(
            external_collisions := ExternalCollisionVariableManager(float_variable_data)
        )
        external_collisions.register_body(robot.root)
        collisions = collision_manager.compute_collisions()

        # test point on a
        point1 = external_collisions.get_group1_P_point_on_a_symbol(robot.root, 0)
        assert np.allclose(point1.evaluate(), np.array([0.0, 0.05, 0.499, 1.0]))
        point2 = external_collisions.get_group1_P_point_on_a_symbol(robot.root, 1)
        assert np.allclose(
            point2.evaluate(), np.array([0.05, 0.0, 0.499, 1.0]), atol=1e-4
        )

        # test contact normal
        contact_normal1 = external_collisions.get_group1_V_contact_normal_symbol(
            robot.root, 0
        )
        assert np.allclose(contact_normal1.evaluate(), np.array([0.0, -1.0, 0.0, 0.0]))
        contact_normal2 = external_collisions.get_group1_V_contact_normal_symbol(
            robot.root, 1
        )
        assert np.allclose(
            contact_normal2.evaluate(), np.array([-1, 0.0, 0.0, 0.0]), atol=1e-4
        )

        # test buffer distance
        buffer_distance1 = external_collisions.get_buffer_distance_symbol(robot.root, 0)
        assert np.allclose(buffer_distance1.evaluate()[0], 15)
        buffer_distance2 = external_collisions.get_buffer_distance_symbol(robot.root, 1)
        assert np.allclose(buffer_distance2.evaluate()[0], 10)

        # test contact distance
        contact_distance1 = external_collisions.get_contact_distance_symbol(
            robot.root, 0
        )
        assert np.allclose(contact_distance1.evaluate()[0], 0.2)
        contact_distance2 = external_collisions.get_contact_distance_symbol(
            robot.root, 1
        )
        assert np.allclose(contact_distance2.evaluate()[0], 0.7)

        # test violated distance
        violated_distance1 = external_collisions.get_violated_distance_symbol(
            robot.root, 0
        )
        assert np.allclose(violated_distance1.evaluate()[0], 0.23)
        violated_distance2 = external_collisions.get_violated_distance_symbol(
            robot.root, 1
        )
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
        group1_P_point_on_a = external_collisions.get_group1_P_point_on_a_symbol(
            robot.root, 0
        )
        group_1_V_contact_normal = (
            external_collisions.get_group1_V_contact_normal_symbol(robot.root, 0)
        )
        expr = group_1_V_contact_normal @ group1_P_point_on_a.to_vector3()
        compiled_expression = expr.compile(VariableParameters.from_lists(variables))
        result = compiled_expression(external_collisions.float_variable_data.data)
        expected = (
            external_collisions.get_group1_V_contact_normal_symbol(
                robot.root, 0
            ).evaluate()
            @ external_collisions.get_group1_P_point_on_a_symbol(
                robot.root, 0
            ).evaluate()
        )
        assert np.allclose(result, expected)
