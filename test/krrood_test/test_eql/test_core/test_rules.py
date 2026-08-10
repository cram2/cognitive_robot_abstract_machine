import pytest

from krrood.entity_query_language.factories import (
    entity,
    variable,
    and_,
    inference,
    an,
    refinement,
    alternative,
    next_rule,
    deduced_variable,
    add,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.core.base_expressions import OperationResult
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.rules.conclusion import Add
from krrood.entity_query_language.rules.conclusion_selector import Refinement
from ...dataset.eql_rule_tree_doc_example import (
    ExampleConnection,
    ExampleView,
    ExampleFixedView,
    ExampleRevoluteView,
)
from ...dataset.semantic_world_like_classes import (
    Container,
    Handle,
    FixedConnection,
    PrismaticConnection,
    Drawer,
    View,
    Door,
    Body,
    RevoluteConnection,
    Wardrobe,
    Cabinet,
)


def test_generate_drawers_from_direct_condition(handles_and_containers_world):
    world = handles_and_containers_world
    container = variable(Container, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    prismatic_connection = variable(PrismaticConnection, domain=world.connections)
    drawers = variable(Drawer, domain=[])
    condition = and_(
        container == fixed_connection.parent,
        handle == fixed_connection.child,
        container == prismatic_connection.child,
    )

    with condition:
        Add(drawers, inference(Drawer)(handle=handle, container=container))

    assert condition._conditions_root_ is condition

    solutions_gen = condition.evaluate()
    all_solutions = list(solutions_gen)

    assert (
        len(all_solutions) == 2
    ), "Should generate components for two possible drawer."
    assert all(isinstance(d[drawers], Drawer) for d in all_solutions)
    assert all_solutions[0][drawers].handle.name == "Handle3"
    assert all_solutions[0][drawers].container.name == "Container3"
    assert all_solutions[1][drawers].handle.name == "Handle1"
    assert all_solutions[1][drawers].container.name == "Container1"


def test_conditions_root_resolves_for_a_condition_shared_by_two_queries():
    container = variable(Container, domain=[])
    fixed_connection = variable(FixedConnection, domain=[])
    shared_condition = container == fixed_connection.parent

    first_drawers = deduced_variable(Drawer)
    first_query = an(entity(first_drawers).where(shared_condition))
    first_query.build()

    second_drawers = deduced_variable(Drawer)
    second_query = an(entity(second_drawers).where(shared_condition))
    second_query.build()

    assert len(shared_condition._parents_) == 2, (
        "the condition must be a direct child of both queries' Where filters for this "
        "to exercise conditions-root resolution on a genuinely shared node"
    )
    assert shared_condition._conditions_root_ is shared_condition


def test_conditions_root_resolves_for_a_subexpression_shared_by_two_compound_conditions():
    body = variable(Body, domain=[])
    handle = variable(Handle, domain=[])
    fixed_connection = variable(FixedConnection, domain=[])
    shared_subexpression = handle == fixed_connection.child

    first_drawers = deduced_variable(Drawer)
    first_compound = and_(body == fixed_connection.parent, shared_subexpression)
    first_query = an(entity(first_drawers).where(first_compound))
    first_query.build()

    second_drawers = deduced_variable(Drawer)
    second_query = an(
        entity(second_drawers).where(and_(body.size > 1, shared_subexpression))
    )
    second_query.build()

    assert len(shared_subexpression._parents_) == 2, (
        "the subexpression must be a direct child of both queries' AND compounds for "
        "this to exercise conditions-root resolution on a node shared two hops below "
        "its owning filters"
    )
    assert shared_subexpression._conditions_root_ is first_compound, (
        "the subexpression's primary parent was fixed at its first attachment (to "
        "first_compound), so resolution must land on first_query's own AND compound, "
        "never second_query's"
    )


def test_conditions_root_resolves_for_a_rule_condition_reused_as_another_querys_filter():
    body = variable(Body, domain=[])
    handle = variable(Handle, domain=[])
    fixed_connection = variable(FixedConnection, domain=[])
    views = deduced_variable(View)
    query = an(entity(views).where(body == fixed_connection.parent))

    with query:
        Add(views, inference(Door)(handle=handle, body=body))
        refinement_condition = body.size > 1
        with refinement(refinement_condition):
            Add(views, inference(Door)(handle=handle, body=body))

    assert len(refinement_condition._parents_) == 1

    other_views = deduced_variable(View)
    other_query = an(entity(other_views).where(refinement_condition))
    other_query.build()

    assert len(refinement_condition._parents_) == 2, (
        "the refinement's condition must also be a direct child of the second query's "
        "Where filter for this to exercise conditions-root resolution on a rule "
        "condition reused outside its own rule tree"
    )
    assert refinement_condition._conditions_root_ is query._conditions_root_, (
        "the refinement condition's primary parent chain was fixed at its first "
        "attachment inside query's own rule tree, so resolution must land on query's "
        "own conditions root (its rule tree's Refinement), never other_query's"
    )


def test_conditions_root_resolves_after_insert_at_clones_an_already_parented_condition():
    body = variable(Body, domain=[])
    handle = variable(Handle, domain=[])
    fixed_connection = variable(FixedConnection, domain=[])

    views = deduced_variable(View)
    query = an(entity(views).where(body == fixed_connection.parent))
    with query:
        Add(views, inference(Door)(handle=handle, body=body))
    anchor = query._conditions_root_

    other_views = deduced_variable(View)
    other_query = an(entity(other_views).where(handle == fixed_connection.child))
    with other_query:
        Add(other_views, inference(Door)(handle=handle, body=body))
    already_parented_condition = other_query._conditions_root_

    # This is the live-growth API (the one insert_refinement/insert_alternative call) rather
    # than the with-refinement(...) DSL, so it exercises _node_for_new_position_'s cloning
    # branch directly: already_parented_condition already has a parent (other_query's Where),
    # so insert_at must splice in a clone rather than reusing the node in place.
    new_condition = Refinement.insert_at(anchor, already_parented_condition)

    assert new_condition is not already_parented_condition
    assert already_parented_condition._conditions_root_ is already_parented_condition, (
        "the original condition must be unaffected by the splice and still resolve "
        "within its own, untouched query"
    )
    assert new_condition._conditions_root_ is query._conditions_root_, (
        "the splice attaches new_condition into query's (the anchor's) own rule tree, "
        "so its primary parent chain must resolve to query's own (now-grown) "
        "conditions root"
    )


def test_generate_drawers_from_query(handles_and_containers_world):
    world = handles_and_containers_world
    container = variable(Container, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    prismatic_connection = variable(PrismaticConnection, domain=world.connections)
    drawers = deduced_variable(Drawer)
    query = an(
        entity(drawers).where(
            container == fixed_connection.parent,
            handle == fixed_connection.child,
            container == prismatic_connection.child,
        )
    )

    with query:
        Add(drawers, inference(Drawer)(handle=handle, container=container))

    solutions = query.evaluate()
    all_solutions = list(solutions)

    assert (
        len(all_solutions) == 2
    ), "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    assert all_solutions[0].handle.name == "Handle3"
    assert all_solutions[0].container.name == "Container3"
    assert all_solutions[1].handle.name == "Handle1"
    assert all_solutions[1].container.name == "Container1"


def test_rule_tree_with_a_refinement(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = variable(Body, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    drawers_and_doors = deduced_variable(View)
    query = an(
        entity(drawers_and_doors).where(
            body == fixed_connection.parent,
            handle == fixed_connection.child,
        )
    )

    with query:
        Add(drawers_and_doors, inference(Drawer)(handle=handle, container=body))
        with refinement(body.size > 1):
            Add(drawers_and_doors, inference(Door)(handle=handle, body=body))

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer and 1 door."
    assert isinstance(all_solutions[0], Door)
    assert all_solutions[0].handle.name == "Handle2"
    assert all_solutions[0].body.name == "Body2"
    assert isinstance(all_solutions[1], Drawer)
    assert all_solutions[1].handle.name == "Handle4"
    assert all_solutions[1].container.name == "Body4"
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].handle.name == "Handle1"
    assert all_solutions[2].container.name == "Container1"


def test_rule_tree_with_multiple_refinements(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = variable(Body, domain=world.bodies)
    container = variable(Container, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    revolute_connection = variable(RevoluteConnection, domain=world.connections)
    views = deduced_variable(View)
    query = an(
        entity(views).where(
            body == fixed_connection.parent,
            handle == fixed_connection.child,
        )
    )

    with query:
        Add(views, inference(Drawer)(handle=handle, container=body))
        with refinement(body.size > 1):
            Add(views, inference(Door)(handle=handle, body=body))
            with alternative(
                body == revolute_connection.child,
                container == revolute_connection.parent,
            ):
                Add(
                    views,
                    inference(Wardrobe)(handle=handle, body=body, container=container),
                )

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    assert isinstance(all_solutions[0], Door)
    assert all_solutions[0].handle.name == "Handle2"
    assert all_solutions[0].body.name == "Body2"
    assert isinstance(all_solutions[1], Wardrobe)
    assert all_solutions[1].handle.name == "Handle4"
    assert all_solutions[1].container.name == "Container2"
    assert all_solutions[1].body.name == "Body4"
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].handle.name == "Handle1"
    assert all_solutions[2].container.name == "Container1"


def test_rule_tree_with_an_alternative(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = variable(Body, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    revolute_connection = variable(RevoluteConnection, domain=world.connections)
    views = deduced_variable(View)
    query = an(
        entity(views)
        .where(
            body == fixed_connection.parent,
            handle == fixed_connection.child,
        )
        .distinct()
    )

    with query:
        Add(views, inference(Drawer)(handle=handle, container=body))
        with alternative(
            body == revolute_connection.parent, handle == revolute_connection.child
        ):
            Add(views, inference(Door)(handle=handle, body=body))

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 4, "Should generate 3 drawers, 1 door"
    assert isinstance(all_solutions[0], Drawer)
    assert all_solutions[0].handle.name == "Handle2"
    assert all_solutions[0].container.name == "Body2"
    assert isinstance(all_solutions[1], Door)
    assert all_solutions[1].handle.name == "Handle3"
    assert all_solutions[1].body.name == "Body3"
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].handle.name == "Handle4"
    assert all_solutions[2].container.name == "Body4"
    assert isinstance(all_solutions[3], Drawer)
    assert all_solutions[3].handle.name == "Handle1"
    assert all_solutions[3].container.name == "Container1"


def test_rule_tree_with_multiple_alternatives(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = variable(Body, domain=world.bodies)
    container = variable(Container, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    prismatic_connection = variable(PrismaticConnection, domain=world.connections)
    revolute_connection = variable(RevoluteConnection, domain=world.connections)
    views = deduced_variable(View)
    query = an(
        entity(views)
        .where(
            body == fixed_connection.parent,
            handle == fixed_connection.child,
            body == prismatic_connection.child,
        )
        .distinct()
    )

    with query:
        Add(views, inference(Drawer)(handle=handle, container=body))
        with alternative(
            revolute_connection.parent == body, revolute_connection.child == handle
        ):
            Add(views, inference(Door)(handle=handle, body=body))
        with alternative(
            fixed_connection.parent == body,
            fixed_connection.child == handle,
            body == revolute_connection.child,
            container == revolute_connection.parent,
        ):
            Add(
                views,
                inference(Wardrobe)(handle=handle, body=body, container=container),
            )

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {
        (Door, "Handle3", "Body3"),
        (Drawer, "Handle1", "Container1"),
        (Wardrobe, "Handle4", "Body4", "Container2"),
    }
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_optimized(doors_and_drawers_world):
    world = doors_and_drawers_world
    fixed_connection = variable(FixedConnection, domain=world.connections)
    prismatic_connection = variable(PrismaticConnection, domain=world.connections)
    revolute_connection = variable(RevoluteConnection, domain=world.connections)
    views = deduced_variable(View)
    query = an(
        entity(views)
        .where(
            HasType(fixed_connection.child, Handle),
            fixed_connection.parent == prismatic_connection.child,
        )
        .distinct()
    )

    with query:
        Add(
            views,
            inference(Drawer)(
                handle=fixed_connection.child, container=fixed_connection.parent
            ),
        )
        with alternative(HasType(revolute_connection.child, Handle)):
            Add(
                views,
                inference(Door)(
                    handle=revolute_connection.child, body=revolute_connection.parent
                ),
            )
        with alternative(
            fixed_connection,
            fixed_connection.parent == revolute_connection.child,
            HasType(revolute_connection.parent, Container),
        ):
            Add(
                views,
                inference(Wardrobe)(
                    handle=fixed_connection.child,
                    body=fixed_connection.parent,
                    container=revolute_connection.parent,
                ),
            )

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {
        (Door, "Handle3", "Body3"),
        (Drawer, "Handle1", "Container1"),
        (Wardrobe, "Handle4", "Body4", "Container2"),
    }
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_better_rule_tree(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = variable(Body, domain=world.bodies)
    container = variable(Container, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    prismatic_connection = variable(PrismaticConnection, domain=world.connections)
    revolute_connection = variable(RevoluteConnection, domain=world.connections)
    views = deduced_variable(View)
    query = an(
        entity(views)
        .where(
            body == fixed_connection.parent,
            handle == fixed_connection.child,
        )
        .distinct()
    )

    with query:
        with refinement(prismatic_connection.child == body):
            Add(views, inference(Drawer)(handle=handle, container=body))
            with alternative(
                body == revolute_connection.child,
                container == revolute_connection.parent,
            ):
                Add(
                    views,
                    inference(Wardrobe)(handle=handle, body=body, container=container),
                )
        with alternative(
            revolute_connection.parent == body, revolute_connection.child == handle
        ):
            Add(views, inference(Door)(handle=handle, body=body))

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {
        (Door, "Handle3", "Body3"),
        (Drawer, "Handle1", "Container1"),
        (Wardrobe, "Handle4", "Body4", "Container2"),
    }
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_better_rule_tree_optimized(
    doors_and_drawers_world,
):
    world = doors_and_drawers_world
    fixed_connection = variable(FixedConnection, domain=world.connections)
    prismatic_connection = variable(PrismaticConnection, domain=world.connections)
    revolute_connection = variable(RevoluteConnection, domain=world.connections)
    views = deduced_variable(View)
    query = an(
        entity(views)
        .where(
            HasType(fixed_connection.child, Handle),
        )
        .distinct()
    )

    with query:
        with refinement(prismatic_connection.child == fixed_connection.parent):
            Add(
                views,
                inference(Drawer)(
                    handle=fixed_connection.child, container=fixed_connection.parent
                ),
            )
            with alternative(
                fixed_connection.parent == revolute_connection.child,
                HasType(revolute_connection.parent, Container),
            ):
                Add(
                    views,
                    inference(Wardrobe)(
                        handle=fixed_connection.child,
                        body=fixed_connection.parent,
                        container=revolute_connection.parent,
                    ),
                )
        with next_rule(HasType(revolute_connection.child, Handle)):
            Add(
                views,
                inference(Door)(
                    handle=revolute_connection.child, body=revolute_connection.parent
                ),
            )

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {
        (Door, "Handle3", "Body3"),
        (Drawer, "Handle1", "Container1"),
        (Wardrobe, "Handle4", "Body4", "Container2"),
    }
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_with_grouped_by(inferred_cabinets_world):
    world = inferred_cabinets_world
    drawer = variable(Drawer, world.views)
    prismatic_connection = variable(PrismaticConnection, world.connections)
    cabinets = (
        entity(
            inference(Cabinet)(
                container=prismatic_connection.parent,
                drawers=drawer,
            )
        )
        .where(prismatic_connection.child == drawer.container)
        .grouped_by(prismatic_connection.parent)
        .tolist()
    )
    assert len(cabinets) == 2
    assert cabinets[0].container.name == "Container2"
    assert len(cabinets[0].drawers) == 2
    assert {d.handle.name for d in cabinets[0].drawers} == {"Handle1", "Handle3"}
    assert cabinets[1].container.name == "Container4"
    assert len(cabinets[1].drawers) == 1
    assert cabinets[1].drawers[0].handle.name == "Handle3"


@pytest.fixture
def rule_tree_doc_example_connections():
    return [
        ExampleConnection(1, "c1"),
        ExampleConnection(2, "c2"),
        ExampleConnection(3, "c3"),
        ExampleConnection(4, "m4"),
    ]


@pytest.mark.parametrize(
    ["alternative_code", "result_set"],
    [
        (
            2,
            og_set := {
                ExampleFixedView(ExampleConnection(1, "c1")),
                ExampleView(ExampleConnection(2, "c2")),
                ExampleView(ExampleConnection(3, "c3")),
            },
        ),
        (4, og_set | {ExampleRevoluteView(ExampleConnection(4, "m4"))}),
    ],
)
def test_doc_example(rule_tree_doc_example_connections, alternative_code, result_set):
    c = variable(ExampleConnection, domain=rule_tree_doc_example_connections)
    view = deduced_variable(ExampleView)

    # 1. Base query
    query = entity(view).where(c.name.startswith("c"))

    # 2. Rule Tree definition
    with query:
        # Default case:
        add(view, inference(ExampleView)(connection=c))

        # If type_code is 1, it's a ExampleFixedView
        with refinement(c.type_code == 1):
            add(view, inference(ExampleFixedView)(connection=c))

        # Otherwise, if type_code is 'alternative_code`, it's a ExampleRevoluteView
        with alternative(c.type_code == alternative_code):
            add(view, inference(ExampleRevoluteView)(connection=c))

    # 3. Execution
    results = query.tolist()
    assert len(results) == len(result_set)
    assert set(results) == result_set


def test_conclusions_of_type_returns_matching_conclusions(handles_and_containers_world):
    """
    ``conclusions_of_type`` returns the attached conclusions of the requested subtype.
    """
    world = handles_and_containers_world
    container = variable(Container, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    prismatic_connection = variable(PrismaticConnection, domain=world.connections)
    drawers = variable(Drawer, domain=[])
    condition = and_(
        container == fixed_connection.parent,
        handle == fixed_connection.child,
        container == prismatic_connection.child,
    )

    with condition:
        added = Add(drawers, inference(Drawer)(handle=handle, container=container))

    assert condition.conclusions_of_type(Add) == [added]


def test_conclusions_of_type_is_empty_without_matching_conclusions(
    handles_and_containers_world,
):
    """
    ``conclusions_of_type`` returns an empty list on an expression with no such
    conclusions.
    """
    world = handles_and_containers_world
    fixed_connection = variable(FixedConnection, domain=world.connections)

    assert fixed_connection.conclusions_of_type(Add) == []


def test_unwrapped_value_strips_literal_wrapper(handles_and_containers_world):
    """
    ``unwrapped_value`` returns the raw value behind a :class:`Literal` right-hand side.
    """
    world = handles_and_containers_world
    container = variable(Container, domain=world.bodies)
    drawers = variable(Drawer, domain=[])
    literal = Literal(_value_="drawer-value")
    condition = container == container

    with condition:
        added = Add(drawers, literal)

    assert added.unwrapped_value == "drawer-value"


def test_unwrapped_value_returns_non_literal_right_unchanged(
    handles_and_containers_world,
):
    """
    ``unwrapped_value`` returns the right-hand expression unchanged when it is not a
    literal.
    """
    world = handles_and_containers_world
    container = variable(Container, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    drawers = variable(Drawer, domain=[])
    condition = container == container

    with condition:
        conclusion_value = inference(Drawer)(handle=handle, container=container)
        added = Add(drawers, conclusion_value)

    assert added.unwrapped_value is conclusion_value


def test_rule_tree_anchors_when_where_condition_is_reused_in_a_sibling():
    """
    A node used as the bare WHERE condition and reused inside a sibling branch must
    still anchor.

    ``drawer.correct`` is a shared node: it is the WHERE condition and also appears in
    the ``alternative`` condition ``drawer.correct == False``. Building that comparator
    adds it as an extra parent of the shared node. Its primary ``_parent_`` must stay
    the structural (WHERE) parent so rule-tree splicing still finds the anchor; when the
    reuse overwrote ``_parent_`` the splice navigated from the comparator instead and
    failed.
    """
    correct_drawer = Drawer(
        handle=Handle("Handle1"), container=Container("Container1"), correct=True
    )
    incorrect_drawer = Drawer(
        handle=Handle("Handle2"), container=Container("Container2"), correct=False
    )
    drawer = variable(Drawer, domain=[correct_drawer, incorrect_drawer])
    views = deduced_variable(View)
    query = an(entity(views).where(drawer.correct))

    with query:
        add(views, inference(Door)(handle=drawer.handle, body=drawer.container))
        with alternative(drawer.correct == False):
            add(views, inference(Door)(handle=drawer.handle, body=drawer.container))

    all_solutions = list(query.evaluate())
    assert (
        len(all_solutions) == 2
    ), "The base branch and its reused-condition alternative must both fire."
    assert {(door.handle.name, door.body.name) for door in all_solutions} == {
        ("Handle1", "Container1"),
        ("Handle2", "Container2"),
    }


# %% shared-anchor splice parentage


def test_splice_keeps_an_earlier_branchs_comparator_operand_intact():
    """
    Splicing under a shared anchor must not rewire an unrelated earlier operand.

    ``drawer.correct`` is a shared-identity singleton. Using it first inside
    ``alternative(drawer.correct == False)`` makes that ``Comparator`` its primary
    ``_parent_``, so a nested refinement anchored on the later bare
    ``refinement(drawer.correct)`` must splice above the parent that branch owns it by,
    not above the comparator.
    """
    drawer = variable(Drawer, domain=[])
    views = deduced_variable(View)
    query = an(entity(views).where(drawer.container))

    with query:
        add(views, inference(Door)(handle=drawer.handle, body=drawer.container))
        with alternative(drawer.correct == False) as earlier_comparator:
            add(views, inference(Door)(handle=drawer.handle, body=drawer.container))
        with refinement(drawer.correct):
            add(views, inference(Door)(handle=drawer.handle, body=drawer.container))
            with refinement(drawer.handle):
                add(views, inference(Door)(handle=drawer.handle, body=drawer.container))

    assert earlier_comparator.left is drawer.correct


def test_nested_refinement_fires_when_its_anchor_was_attached_in_an_earlier_branch():
    """
    A refinement nested under a reused anchor must stay in the rule tree and fire.

    The splice that drags the nested branch above an earlier sibling's comparator
    detaches it from the rule tree, so its conclusion never replaces the one it refines.
    """
    correct_drawer = Drawer(
        handle=Handle("Handle1"), container=Container("Container1"), correct=True
    )
    incorrect_drawer = Drawer(
        handle=Handle("Handle2"), container=Container("Container2"), correct=False
    )
    drawer = variable(Drawer, domain=[correct_drawer, incorrect_drawer])
    views = deduced_variable(View)
    query = an(entity(views).where(drawer.container))

    with query:
        add(views, inference(Door)(handle=drawer.handle, body=drawer.container))
        with alternative(drawer.correct == False):
            add(views, inference(Door)(handle=drawer.handle, body=drawer.container))
        with refinement(drawer.correct):
            add(
                views,
                inference(Drawer)(handle=drawer.handle, container=drawer.container),
            )
            with refinement(drawer.handle.name == "Handle1"):
                add(views, inference(Cabinet)(container=drawer.container))

    all_solutions = list(query.evaluate())

    assert any(isinstance(view, Cabinet) for view in all_solutions)


def test_conclusions_fire_without_an_active_evaluation_context(
    handles_and_containers_world,
):
    """
    A conclusion must still fire when no ``EvaluationContext`` is active.

    ``_evaluate_conclusions_and_update_bindings_`` is normally only reached from inside
    ``_evaluate_``, which has already set one up. But real-world callers can drive
    evaluation from a code path where no context was ever created for the current thread
    (for example, resuming a query from a thread that does not share the caller's
    ``contextvars.Context`` -- Python's ``ContextVar`` values do not propagate into a
    plain ``threading.Thread`` by default). This calls the raw, double-underscore
    ``_evaluate__`` directly (bypassing ``_evaluate_``'s context setup entirely) to
    prove the conclusion-firing check falls back to a purely structural one instead of
    assuming a context always exists.
    """
    world = handles_and_containers_world
    container = variable(Container, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    drawers = variable(Drawer, domain=[])
    condition = and_(
        container == fixed_connection.parent,
        handle == fixed_connection.child,
    )

    with condition:
        Add(drawers, inference(Drawer)(handle=handle, container=container))

    assert condition._conditions_root_ is condition

    raw_result = next(
        result
        for result in condition._evaluate__(OperationResult({}))
        if not result.is_false
    )

    processed_result = condition._evaluate_conclusions_and_update_bindings_(raw_result)

    assert drawers._id_ in processed_result.bindings
    assert isinstance(processed_result.bindings[drawers._id_], Drawer)
