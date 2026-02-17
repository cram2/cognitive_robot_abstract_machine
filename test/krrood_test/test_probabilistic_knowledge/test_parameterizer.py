from __future__ import annotations

import unittest

from random_events.variable import Continuous
from random_events.product_algebra import SimpleEvent

from krrood.ormatic.dao import to_dao
from krrood.probabilistic_knowledge.parameterizer import Parameterizer, Parameterization
from test.krrood_test.dataset.example_classes import OptionalTestCase
from test.krrood_test.dataset.example_classes import Position, Pose, Orientation


def test_parameterize_position():
    """
    Test parameterization of the Position class.
    """
    position = Position(None, None, None)
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize_dao(to_dao(position), "Position")
    expected_variables = [
        Continuous("Position.x"),
        Continuous("Position.y"),
        Continuous("Position.z"),
    ]
    assert parameterization.variables == expected_variables


def test_parameterize_orientation():
    """
    Test parameterization of the Orientation class.
    """
    orientation = Orientation(None, None, None, None)
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize_dao(
        to_dao(orientation), "Orientation"
    )
    expected_variables = [
        Continuous("Orientation.x"),
        Continuous("Orientation.y"),
        Continuous("Orientation.z"),
    ]

    assert parameterization.variables == expected_variables


def test_parameterize_pose():
    """
    Test parameterization of the Pose class.
    """
    pose = Pose(
        position=Position(None, None, None),
        orientation=Orientation(None, None, None, None),
    )
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize_dao(to_dao(pose), "Pose")
    expected_variables = [
        Continuous("Pose.position.x"),
        Continuous("Pose.position.y"),
        Continuous("Pose.position.z"),
        Continuous("Pose.orientation.x"),
        Continuous("Pose.orientation.y"),
        Continuous("Pose.orientation.z"),
    ]

    assert parameterization.variables == expected_variables


def test_create_fully_factorized_distribution():
    """
    Test for a fully factorized distribution.
    """
    variables = [
        Continuous("Variable.A"),
        Continuous("Variable.B"),
    ]
    parameterization = Parameterization(variables)
    probabilistic_circuit = parameterization.create_fully_factorized_distribution()
    assert len(probabilistic_circuit.variables) == 2
    assert set(probabilistic_circuit.variables) == set(variables)


def test_parameterize_object():
    """
    Test parameterization of a single object via parameterize.
    """
    position = Position(x=1.0, y=2.0, z=3.0)
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(position, "Position")
    expected_names = {"Position.x", "Position.y", "Position.z"}
    assert {v.name for v in parameterization.variables} == expected_names
    assert len(parameterization.simple_event.variables) == 3


def test_parameterize_list():
    """
    Test parameterization of a list of objects via parameterize.
    """
    positions = [Position(x=1.0, y=2.0, z=3.0), Position(x=4.0, y=5.0, z=6.0)]
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(positions, "Positions")
    expected_names = {
        "Positions[0].x",
        "Positions[0].y",
        "Positions[0].z",
        "Positions[1].x",
        "Positions[1].y",
        "Positions[1].z",
    }
    variables = parameterization.variables
    event = parameterization.simple_event

    assert {v.name for v in variables} == expected_names
    assert len(event.variables) == 6
    # Check some values
    x0_var = next(v for v in variables if v.name == "Positions[0].x")
    x1_var = next(v for v in variables if v.name == "Positions[1].x")
    assert event[x0_var].simple_sets[0].lower == 1.0
    assert event[x1_var].simple_sets[0].lower == 4.0


def test_parameterize_nested_object():
    """
    Test parameterization of a nested object via parameterize.
    """
    pose = Pose(
        position=Position(x=1.0, y=2.0, z=3.0),
        orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    parameterizer = Parameterizer()
    parameterization = parameterizer.parameterize(pose, "Pose")

    # Position has x, y, z (3)
    # Orientation has x, y, z, w (4)
    # Total = 7
    assert len(parameterization.variables) == 7
    expected_names = {
        "Pose.position.x",
        "Pose.position.y",
        "Pose.position.z",
        "Pose.orientation.x",
        "Pose.orientation.y",
        "Pose.orientation.z",
        "Pose.orientation.w",
    }
    assert {v.name for v in parameterization.variables} == expected_names


class TestDAOParameterizer(unittest.TestCase):

    def setUp(self):
        self.parameterizer = Parameterizer()

    def test_parameterize_flat_dao(self):
        pos = Position(x=1.0, y=2.0, z=3.0)
        pos_dao = to_dao(pos)
        parameterization = self.parameterizer.parameterize_dao(pos_dao, "pos")
        variables = parameterization.variables
        event = parameterization.simple_event

        self.assertEqual(len(variables), 3)
        var_names = {v.name for v in variables}
        self.assertIn("pos.x", var_names)
        self.assertIn("pos.y", var_names)
        self.assertIn("pos.z", var_names)

        for var in variables:
            self.assertIsInstance(var, Continuous)
            self.assertEqual(
                event[var].simple_sets[0].lower,
                getattr(pos, var.name.split(".")[-1]),
            )

    def test_parameterize_nested_dao(self):
        pose = Pose(
            position=Position(x=1.0, y=2.0, z=3.0),
            orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        pose_dao = to_dao(pose)
        parameterization = self.parameterizer.parameterize_dao(pose_dao, "pose")
        variables = parameterization.variables
        event = parameterization.simple_event

        # Position has x, y, z (3)
        # Orientation has x, y, z, w (4)
        # Total = 7
        self.assertEqual(len(variables), 7)

        var_names = {v.name for v in variables}
        self.assertIn("pose.position.x", var_names)
        self.assertIn("pose.orientation.w", var_names)

        # Check values in event
        for var in variables:
            parts = var.name.split(".")
            if parts[1] == "position":
                val = getattr(pose.position, parts[2])
            else:
                val = getattr(pose.orientation, parts[2])
            self.assertEqual(event[var].simple_sets[0].lower, val)

    def test_parameterize_dao_set_value(self):
        optional = OptionalTestCase(1)
        optional_dao = to_dao(optional)
        parameterization = self.parameterizer.parameterize_dao(optional_dao, "optional")

        self.assertEqual(1, len(parameterization.variables))

    def test_parameterize_dao_none_value(self):
        optional = OptionalTestCase(None)
        optional_dao = to_dao(optional)
        parameterization = self.parameterizer.parameterize_dao(optional_dao, "optional")

        self.assertEqual(len(parameterization.variables), 1)

    def test_parameterize_dao_set_value_set_optional(self):
        optional = OptionalTestCase(1, Position(1.0, 2.0, 3.0))
        optional_dao = to_dao(optional)
        parameterization = self.parameterizer.parameterize_dao(optional_dao, "optional")

        self.assertEqual(len(parameterization.variables), 4)

    def test_parameterize_dao_none_value_underspecified_optional(self):
        optional = OptionalTestCase(None, Position(1.0, None, 3.0))
        optional_dao = to_dao(optional)
        parameterization = self.parameterizer.parameterize_dao(optional_dao, "optional")

        self.assertEqual(len(parameterization.variables), 4)

    def test_parameterize_dao_set_value_set_relationship(self):
        optional = OptionalTestCase(
            1, list_of_orientations=[Orientation(0.0, 0.0, 0.0, 1.0)]
        )
        optional_dao = to_dao(optional)
        parameterization = self.parameterizer.parameterize_dao(optional_dao, "optional")

        self.assertEqual(len(parameterization.variables), 5)

    def test_parameterize_dao_set_value_underspecified_relationship(self):
        optional = OptionalTestCase(
            1, list_of_orientations=[Orientation(0.0, 0.0, None, 1.0)]
        )
        optional_dao = to_dao(optional)
        parameterization = self.parameterizer.parameterize_dao(optional_dao, "optional")

        self.assertEqual(len(parameterization.variables), 5)

    def test_parameterize_dao_set_value_set_builtin(self):
        optional = OptionalTestCase(1, list_of_values=[0])
        optional_dao = to_dao(optional)
        parameterization = self.parameterizer.parameterize_dao(optional_dao, "optional")

        self.assertEqual(len(parameterization.variables), 2)

    def test_parameterize_dao_set_value_underspecified_builtin(self):
        optional = OptionalTestCase(1, list_of_values=[None])
        optional_dao = to_dao(optional)
        parameterization = self.parameterizer.parameterize_dao(optional_dao, "optional")

        self.assertEqual(len(parameterization.variables), 2)

    def test_parameterize_dao_with_optional_filled(self):
        # Orientation with w=None
        orient = Orientation(x=0.0, y=0.0, z=None, w=1.0)
        orient_dao = to_dao(orient)
        parameterization = self.parameterizer.parameterize_dao(orient_dao, "orient")
        variables = parameterization.variables
        event = parameterization.simple_event

        self.assertEqual(len(variables), 4)
        z_var = next(v for v in variables if v.name == "orient.z")
        # Since SimpleEvent fills missing variables with reals(), orient.w should be a full interval
        self.assertTrue(event[z_var].simple_sets[0].lower < -1e10)
        self.assertTrue(event[z_var].simple_sets[0].upper > 1e10)

    def test_parameterization_initialization(self):
        """
        Test the initialization of Parameterization with default values.
        """
        parameterization = Parameterization()
        self.assertEqual(parameterization.variables, [])
        self.assertEqual(parameterization.simple_event, SimpleEvent({}))

    def test_parameterization_update_variables(self):
        """
        Test updating variables in Parameterization.
        """
        parameterization = Parameterization()
        var_a = Continuous("A")
        parameterization.extend_variables([var_a])
        self.assertEqual(parameterization.variables, [var_a])

        var_b = Continuous("B")
        parameterization.extend_variables([var_b])
        self.assertEqual(parameterization.variables, [var_a, var_b])

    def test_parameterization_update_simple_event(self):
        """
        Test updating simple event in Parameterization.
        """
        var_a = Continuous("A")
        parameterization = Parameterization(variables=[var_a])
        event = SimpleEvent({var_a: 1.0})
        parameterization.update_simple_event(event)
        self.assertEqual(parameterization.simple_event[var_a].simple_sets[0].lower, 1.0)

    def test_parameterization_fill_missing_variables(self):
        """
        Test filling missing variables in Parameterization.
        """
        var_a = Continuous("A")
        var_b = Continuous("B")
        parameterization = Parameterization(variables=[var_a, var_b])
        parameterization.update_simple_event(SimpleEvent({var_a: 1.0}))

        self.assertNotIn(var_b, parameterization.simple_event.variables)
        parameterization.fill_missing_variables()
        self.assertIn(var_b, parameterization.simple_event.variables)

    def test_parameterization_update_parameterization(self):
        """
        Test updating a parameterization with another one.
        """
        var_a = Continuous("A")
        param_a = Parameterization(
            variables=[var_a], simple_event=SimpleEvent({var_a: 1.0})
        )

        var_b = Continuous("B")
        param_b = Parameterization(
            variables=[var_b], simple_event=SimpleEvent({var_b: 2.0})
        )

        param_a.merge_parameterization(param_b)

        self.assertEqual(len(param_a.variables), 2)
        self.assertIn(var_a, param_a.variables)
        self.assertIn(var_b, param_a.variables)
        self.assertEqual(param_a.simple_event[var_a].simple_sets[0].lower, 1.0)
        self.assertEqual(param_a.simple_event[var_b].simple_sets[0].lower, 2.0)


if __name__ == "__main__":
    unittest.main()
