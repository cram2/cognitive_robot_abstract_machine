import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field

from krrood.symbolic_math.float_variable_data import (
    FloatVariableData,
)
from krrood.symbolic_math.symbolic_math import Vector, VariableParameters, FloatVariable
from semantic_digital_twin.collision_checking.collision_manager import (
    CollisionConsumer,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    MaxAvoidedCollisionsOverride,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowAllCollisions,
    AvoidCollisionBetweenGroups,
    AvoidExternalCollisions,
    AvoidSelfCollisions,
)
from semantic_digital_twin.collision_checking.collision_variable_managers import (
    ExternalCollisionVariableManager,
    SelfCollisionVariableManager,
)
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world import World


class TestExternalCollisionExpressionManager:
    def test_simple(self, cylinder_bot_world):
        float_variable_data = FloatVariableData()
        float_variable_data.register_expression(FloatVariable("muh"))

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
        assert np.allclose(
            point1.evaluate(), np.array([0.0, 0.05, 0.249, 1.0]), atol=1e-3
        )
        point2 = external_collisions.get_group_a_P_point_on_a_symbol(group, 1)
        assert np.allclose(
            point2.evaluate(), np.array([0.05, 0.0, 0.249, 1.0]), atol=1e-3
        )
        point2 = external_collisions.get_group_a_P_point_on_a_symbol(group, 1)
        assert np.allclose(
            point2.evaluate(), np.array([0.05, 0.0, 0.249, 1.0]), atol=1e-3
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
        assert np.allclose(contact_distance1.evaluate()[0], 0.2, atol=1e-3)
        contact_distance2 = external_collisions.get_contact_distance_symbol(group, 1)
        assert np.allclose(contact_distance2.evaluate()[0], 0.7, atol=1e-3)

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
        float_variable_data.register_expression(FloatVariable("muh"))

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
        float_variable_data.register_expression(FloatVariable("muh"))

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
        float_variable_data.register_expression(v)
        float_variable_data.set_value(v, 23)
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


def test_a_copied_world_keeps_the_distances_its_collision_rules_were_given(
    _pr2_world_setup,
):
    """
    A copy is used to try a motion out before it is run, so a copy whose rules fell back
    to their default distances answers for a robot that keeps less clearance than the
    one that goes on to execute.
    """
    original = [
        (type(rule), rule.buffer_zone_distance, rule.violated_distance)
        for rule in _pr2_world_setup.collision_manager.default_rules
    ]

    copied = [
        (type(rule), rule.buffer_zone_distance, rule.violated_distance)
        for rule in deepcopy(_pr2_world_setup).collision_manager.default_rules
    ]

    assert copied == original


# %% rules only name bodies that can collide


def test_an_avoid_rule_leaves_out_subset_bodies_that_cannot_collide(pr2_world_copy):
    """
    A body with no geometry can never be the reason two things are kept apart, so naming
    one in a rule only misstates what that rule covers.
    """
    robot = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]
    without_geometry = pr2_world_copy.get_body_by_name("l_force_torque_adapter_link")
    with_geometry = pr2_world_copy.get_body_by_name("l_force_torque_link")
    assert not without_geometry.has_collision()
    assert with_geometry.has_collision()

    rule = AvoidExternalCollisions(
        robot=robot, body_subset={without_geometry, with_geometry}
    )

    assert rule.body_subset == {with_geometry}


def test_no_collision_rule_names_a_body_that_cannot_collide(pr2_world_copy):
    """
    Every rule the robot installs is built from bodies that carry geometry, and stays
    that way whichever rule is added next.
    """
    for rule in pr2_world_copy.collision_manager.rules:
        rule.update(pr2_world_copy)

    named = {
        body
        for rule in pr2_world_copy.collision_manager.rules
        for body in rule.referenced_bodies
    }

    assert sorted(str(body.name) for body in named if not body.has_collision()) == []


# %% whether a robot touches anything


def test_robot_is_in_collision_reports_a_contact_of_its_own(cylinder_bot_world):
    """
    The robot answers for its own bodies, under whichever rules the caller set.
    """
    robot = cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0]
    environment = cylinder_bot_world.get_kinematic_structure_entity_by_name(
        "environment"
    )
    collision_manager = cylinder_bot_world.collision_manager
    collision_manager.extend_temporary_rule(
        [
            AvoidCollisionBetweenGroups(
                buffer_zone_distance=10,
                violated_distance=10,
                body_group_a=[robot.root],
                body_group_b=[environment],
            )
        ]
    )
    collision_manager.update_collision_matrix()

    assert robot.is_in_collision


def test_robot_is_not_in_collision_while_it_keeps_its_distance(cylinder_bot_world):
    """
    A pair is watched from far enough away to see it coming, so being reported says only
    that the two are being watched.

    The robot counts as in collision once it comes within the distance the rule says
    must not be violated, and not before.
    """
    robot = cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0]
    environment = cylinder_bot_world.get_kinematic_structure_entity_by_name(
        "environment"
    )
    collision_manager = cylinder_bot_world.collision_manager
    collision_manager.extend_temporary_rule(
        [
            AvoidCollisionBetweenGroups(
                buffer_zone_distance=10,
                violated_distance=0.1,
                body_group_a=[robot.root],
                body_group_b=[environment],
            )
        ]
    )
    collision_manager.update_collision_matrix()
    [contact] = [
        contact
        for contact in collision_manager.compute_collisions().contacts
        if contact.body_a is environment or contact.body_b is environment
    ]

    assert contact.distance > 0.1
    assert not robot.is_in_collision


def test_robot_is_not_in_collision_when_nothing_is_checked(cylinder_bot_world):
    """
    With every collision of the robot allowed, no contact is reported for it.
    """
    robot = cylinder_bot_world.get_semantic_annotations_by_type(MinimalRobot)[0]
    collision_manager = cylinder_bot_world.collision_manager
    collision_manager.extend_temporary_rule([AllowAllCollisions()])
    collision_manager.update_collision_matrix()

    assert not robot.is_in_collision


# %% consumers stay out of the serialized model


@dataclass
class ConsumerHoldingSomethingUnserializable(CollisionConsumer):
    """
    A consumer carrying a value no JSON serializer knows, as a live ROS publisher does.
    """

    live_handle: object = field(default_factory=object)
    """
    Stands in for the node a visualization publisher keeps hold of.
    """

    def on_compute_collisions(self, collision_results):
        pass

    def on_world_model_update(self, world):
        pass

    def on_collision_matrix_update(self):
        pass


def test_a_consumer_does_not_have_to_be_serializable(pr2_world_copy):
    """
    Consumers are live observers, not part of the model, so attaching one must not make
    the world's modification history unserializable.
    """
    collision_manager = pr2_world_copy.collision_manager
    collision_manager.add_collision_consumer(ConsumerHoldingSomethingUnserializable())

    assert "collision_consumers" not in collision_manager.to_json()
