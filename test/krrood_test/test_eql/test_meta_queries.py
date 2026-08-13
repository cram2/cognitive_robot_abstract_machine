"""
Tests for InferenceExplanation meta-query methods using real Drawer and Cabinet rules.

Each test group exercises one method against two rules so the expected counts can be
verified against the known rule structure.

Drawer rule (DoorsAndDrawersWorld):
    handle         = variable(Handle, world.bodies)
    prismatic_conn = variable(PrismaticConnection, world.connections)
    fixed_conn     = match_variable(FixedConnection, world.connections)(
                         parent=prismatic_conn.child, child=handle)
    rule           = inference(Drawer)(container=fixed_conn.parent, handle=fixed_conn.child)
  Conditions: fixed_conn.parent == prismatic_conn.child  AND  fixed_conn.child == handle

  Note on descendant traversal: because fixed_conn's match conditions embed prismatic_conn
  as a child node, node_descendants of EITHER comparator reaches both FixedConnection and
  PrismaticConnection variable nodes.  Methods that look for two distinct Connection-typed
  descendants therefore find 2 matching conditions (not 1).

Cabinet rule (InferredCabinetsWorld):
    drawer         = variable(Drawer, world.views)
    prismatic_conn = variable(PrismaticConnection, world.connections)
    rule           = entity(inference(Cabinet)(...)).where(prismatic_conn.child == drawer.container)
  Conditions: prismatic_conn.child == drawer.container
"""

import pytest

from krrood.entity_query_language.explanation.explanation import (
    InferenceExplanation,
    explain_inference,
)
from krrood.entity_query_language.factories import (
    entity,
    variable,
    match_variable,
    inference,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalOperator,
)
from ..dataset.semantic_world_like_classes import (
    Handle,
    Container,
    Body,
    Drawer,
    Cabinet,
    Connection,
    FixedConnection,
    PrismaticConnection,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def drawer_explanation(doors_and_drawers_world) -> InferenceExplanation:
    world = doors_and_drawers_world
    handle = variable(Handle, world.bodies)
    prismatic_conn = variable(PrismaticConnection, world.connections)
    fixed_conn = match_variable(FixedConnection, world.connections)(
        parent=prismatic_conn.child, child=handle
    )
    drawers = inference(Drawer)(
        container=fixed_conn.parent, handle=fixed_conn.child
    ).tolist()
    assert drawers, "Drawer rule produced no results – check the world fixture"
    explanation = explain_inference(drawers[0])
    assert explanation is not None
    return explanation


@pytest.fixture
def cabinet_explanation(inferred_cabinets_world) -> InferenceExplanation:
    world = inferred_cabinets_world
    drawer = variable(Drawer, world.views)
    prismatic_conn = variable(PrismaticConnection, world.connections)
    cabinets = (
        entity(inference(Cabinet)(container=prismatic_conn.parent, drawers=drawer))
        .where(prismatic_conn.child == drawer.container)
        .grouped_by(prismatic_conn.parent)
        .tolist()
    )
    assert cabinets, "Cabinet rule produced no results – check the world fixture"
    explanation = explain_inference(cabinets[0])
    assert explanation is not None
    return explanation


# ---------------------------------------------------------------------------
# get_satisfied_condition_expressions_for_the_instance
# ---------------------------------------------------------------------------


def test_drawer_satisfied_conditions_returns_comparators(drawer_explanation):
    # The method returns ALL satisfied condition expressions, including LogicalOperator
    # wrappers (AND nodes) produced by match_variable and the inference root.
    # The Drawer rule has 2 comparators plus 2 AND nodes from the condition tree.
    conditions = (
        drawer_explanation.get_satisfied_condition_expressions_for_the_instance().tolist()
    )
    assert len(conditions) >= 2
    assert all(isinstance(c, (Comparator, LogicalOperator)) for c in conditions)
    assert sum(1 for c in conditions if isinstance(c, Comparator)) >= 2


def test_cabinet_satisfied_conditions_returns_comparator(cabinet_explanation):
    conditions = (
        cabinet_explanation.get_satisfied_condition_expressions_for_the_instance().tolist()
    )
    assert len(conditions) == 1
    assert isinstance(conditions[0], Comparator)


def test_get_comparator_type_conditions(drawer_explanation):
    conditions = drawer_explanation.get_satisfied_comparator_conditions().tolist()
    assert len(conditions) == 2
    assert all(isinstance(c, Comparator) for c in conditions)


def test_get_comparators_between_attributes(drawer_explanation):
    conditions = (
        drawer_explanation.get_satisfied_comparator_conditions_between_attributes().tolist()
    )
    assert len(conditions) == 1
    assert isinstance(conditions[0], Comparator)


# ---------------------------------------------------------------------------
# get_variable_nodes_of_given_type
# ---------------------------------------------------------------------------


def test_drawer_connection_variable_nodes_count(drawer_explanation):
    nodes = drawer_explanation.get_variable_nodes_of_given_type(Connection).tolist()
    assert len(nodes) == 2
    node_types = {n._type_ for n in nodes}
    assert FixedConnection in node_types
    assert PrismaticConnection in node_types


def test_drawer_handle_variable_nodes_count(drawer_explanation):
    nodes = drawer_explanation.get_variable_nodes_of_given_type(Handle).tolist()
    assert len(nodes) == 1
    assert nodes[0]._type_ is Handle


def test_cabinet_connection_variable_nodes_count(cabinet_explanation):
    nodes = cabinet_explanation.get_variable_nodes_of_given_type(Connection).tolist()
    assert len(nodes) == 1
    assert nodes[0]._type_ is PrismaticConnection


def test_cabinet_drawer_variable_nodes_count(cabinet_explanation):
    nodes = cabinet_explanation.get_variable_nodes_of_given_type(Drawer).tolist()
    assert len(nodes) == 1
    assert nodes[0]._type_ is Drawer


# ---------------------------------------------------------------------------
# get_values_of_variable_nodes_of_given_type
# ---------------------------------------------------------------------------


def test_drawer_handle_value_matches_instance(drawer_explanation):
    handles = drawer_explanation.get_values_of_variable_nodes_of_given_type(
        Handle
    ).tolist()
    assert len(handles) == 1


def test_cabinet_drawer_values_are_in_cabinet(cabinet_explanation):
    drawers = cabinet_explanation.get_values_of_variable_nodes_of_given_type(
        Drawer
    ).tolist()
    assert len(drawers) >= 1
    cabinet = cabinet_explanation.instance
    # grouped_by aggregates the drawer instances into a single list binding, so each
    # element from the query is itself a list of Drawer instances.
    for d in drawers:
        drawer_list = d if isinstance(d, list) else [d]
        for drawer in drawer_list:
            assert isinstance(drawer, Drawer)
            assert drawer in cabinet.drawers


# ---------------------------------------------------------------------------
# get_conditions_that_relate_the_variables_of_type
# ---------------------------------------------------------------------------


def test_drawer_conditions_relating_connections(drawer_explanation):
    conditions = drawer_explanation.get_conditions_that_relate_the_variables_of_type(
        Connection
    ).tolist()
    assert len(conditions) == 1
    assert all(isinstance(c, Comparator) for c in conditions)
    assert conditions[0].left._owner_class_ is FixedConnection
    assert conditions[0].left._attribute_name_ == "parent"
    assert conditions[0].right._owner_class_ is PrismaticConnection
    assert conditions[0].right._attribute_name_ == "child"


def test_drawer_conditions_relating_bodies(drawer_explanation):
    # Both conditions have two Body-typed Attribute descendants:
    #   fixed_conn.parent (Body) and prismatic_conn.child (Body) in condition 1
    #   fixed_conn.child  (Body) and handle (Handle < Body)       in condition 2
    conditions = drawer_explanation.get_conditions_that_relate_the_variables_of_type(
        Body
    ).tolist()
    assert len(conditions) == 2


def test_cabinet_conditions_relating_bodies(cabinet_explanation):
    # prismatic_conn.child (Body) and drawer.container (Container < Body) are in the same condition.
    conditions = cabinet_explanation.get_conditions_that_relate_the_variables_of_type(
        Body
    ).tolist()
    assert len(conditions) == 1
    assert isinstance(conditions[0], Comparator)


def test_cabinet_no_conditions_relating_connections(cabinet_explanation):
    # There is only one Connection variable (prismatic_conn) in the cabinet rule,
    # so no condition can relate two distinct Connection nodes.
    conditions = cabinet_explanation.get_conditions_that_relate_the_variables_of_type(
        Connection
    ).tolist()
    assert len(conditions) == 0


# ---------------------------------------------------------------------------
# get_conditions_that_relate_variables_of_types (new method)
# ---------------------------------------------------------------------------


def test_drawer_relates_fixed_connection_to_prismatic(drawer_explanation):
    # FixedConnection.parent == PrismaticConnection.child
    conditions = drawer_explanation.get_conditions_that_relate_variables_of_types(
        FixedConnection, PrismaticConnection
    ).tolist()
    assert len(conditions) == 1


def test_drawer_relates_fixed_connection_to_handle(drawer_explanation):
    # fixed_conn.child == handle.
    conditions = drawer_explanation.get_conditions_that_relate_variables_of_types(
        FixedConnection, Handle
    ).tolist()
    assert len(conditions) == 1


def test_cabinet_relates_prismatic_connection_to_drawer(cabinet_explanation):
    conditions = cabinet_explanation.get_conditions_that_relate_variables_of_types(
        PrismaticConnection, Drawer
    ).tolist()
    assert len(conditions) == 1
    assert isinstance(conditions[0], Comparator)


def test_get_conditions_that_relate_variables_of_types_is_symmetric(drawer_explanation):
    ab = drawer_explanation.get_conditions_that_relate_variables_of_types(
        FixedConnection, PrismaticConnection
    ).tolist()
    ba = drawer_explanation.get_conditions_that_relate_variables_of_types(
        PrismaticConnection, FixedConnection
    ).tolist()
    assert {c._id_ for c in ab} == {c._id_ for c in ba}
