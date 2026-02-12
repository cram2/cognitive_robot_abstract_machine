from krrood.entity_query_language.conclusion import Add
from krrood.entity_query_language.entity import variable, entity, and_, inference
from krrood.entity_query_language.entity_result_processors import an
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.rule import refinement, alternative, next_rule
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
)


def test_generate_drawers(handles_and_containers_world):
    world = handles_and_containers_world
    container = variable(Container, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    prismatic_connection = variable(PrismaticConnection, domain=world.connections)
    drawers = variable(Drawer, domain=None)
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


def test_add_conclusion(handles_and_containers_world):
    world = handles_and_containers_world

    container = variable(Container, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    prismatic_connection = variable(PrismaticConnection, domain=world.connections)
    drawers = variable(Drawer, domain=None)
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
    # all_drawers = list(drawers._evaluate_())
    # assert len(all_drawers) == 2, "Should generate components for two possible drawer."


def test_rule_tree_with_a_refinement(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = variable(Body, domain=world.bodies)
    handle = variable(Handle, domain=world.bodies)
    fixed_connection = variable(FixedConnection, domain=world.connections)
    drawers_and_doors = variable(View, domain=None, inferred=True)
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

    # query._render_tree_()

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
    views = variable(View, domain=None, inferred=True)
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

    # query._render_tree_()
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
    views = variable(View, domain=None, inferred=True)
    query = an(
        entity(views).where(
            body == fixed_connection.parent,
            handle == fixed_connection.child,
        )
    )

    with query:
        Add(views, inference(Drawer)(handle=handle, container=body))
        with alternative(
            body == revolute_connection.parent, handle == revolute_connection.child
        ):
            Add(views, inference(Door)(handle=handle, body=body))

    query.visualize()

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
    views = variable(View, domain=None, inferred=True)
    query = an(
        entity(views).where(
            body == fixed_connection.parent,
            handle == fixed_connection.child,
            body == prismatic_connection.child,
        )
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

    # query._render_tree_()

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
    views = variable(View, domain=None, inferred=True)
    query = an(
        entity(views).where(
            HasType(fixed_connection.child, Handle),
            fixed_connection.parent == prismatic_connection.child,
        )
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

    # query._render_tree_()

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
    views = variable(View, domain=None, inferred=True)
    query = an(
        entity(views).where(
            body == fixed_connection.parent,
            handle == fixed_connection.child,
        )
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

    # query._render_tree_()

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
    views = variable(View, domain=None, inferred=True)
    query = an(
        entity(views).where(
            HasType(fixed_connection.child, Handle),
        )
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

    # query._render_tree_()

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
