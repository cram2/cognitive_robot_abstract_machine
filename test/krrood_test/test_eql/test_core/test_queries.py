from dataclasses import dataclass
from math import factorial

import pytest

from krrood.entity_query_language.entity import (
    and_,
    not_,
    contains,
    in_,
    entity,
    set_of,
    let,
    or_,
    exists,
    flatten,
    count,
)
import krrood.entity_query_language.entity as eql
from krrood.entity_query_language.quantify_entity import an, a, the
from krrood.entity_query_language.failures import (
    MultipleSolutionFound,
    UnsupportedNegation,
    GreaterThanExpectedNumberOfSolutions,
    LessThanExpectedNumberOfSolutions, NonPositiveLimitValue, LiteralConditionError,
)
from krrood.entity_query_language.predicate import (
    HasType,
    symbolic_function,
    Predicate,
)
from krrood.entity_query_language.result_quantification_constraint import (
    ResultQuantificationConstraint,
    Exactly,
    AtLeast,
    AtMost,
    Range,
)
from ...dataset.semantic_world_like_classes import (
    Handle,
    Body,
    Container,
    FixedConnection,
    PrismaticConnection,
    World,
    Connection,
    FruitBox,
    ContainsType,
    Apple, Drawer,
)


def test_empty_conditions(handles_and_containers_world, doors_and_drawers_world):
    world = handles_and_containers_world
    world2 = doors_and_drawers_world

    query = an(entity(body := let(type_=Body, domain=world.bodies)))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."


def test_empty_conditions_and_no_domain(
    handles_and_containers_world, doors_and_drawers_world
):
    world = handles_and_containers_world
    world2 = doors_and_drawers_world

    query = an(entity(body := let(type_=Body, domain=None), body.world == world))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."


def test_empty_conditions_without_using_entity(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(let(type_=Body, domain=world.bodies)))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."


def test_reevaluation_of_simple_query(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(entity(body := let(type_=Body, domain=world.bodies)))
    assert len(list(query.evaluate())) == len(world.bodies), "Should generate 6 bodies."
    assert len(list(query.evaluate())) == len(
        world.bodies
    ), "Re-eval: Should generate 6 bodies."


def test_filtering_connections_without_joining_with_parent_or_child_queries(
    handles_and_containers_world,
):
    world = handles_and_containers_world

    query = an(
        entity(
            connection := let(Connection, world.connections),
            HasType(connection.parent, Container),
            connection.parent.name == "Container1",
            HasType(connection.child, Handle),
        )
    )

    results = list(query.evaluate())
    assert len(results) == 1, "Should generate 1 connections."
    assert results[0].parent.name == "Container1"
    assert results[0].child.name == "Handle1"


def test_generate_with_using_attribute_and_callables(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    def generate_handles():

        yield from an(
            entity(body := let(Body, world.bodies), body.name.startswith("Handle"))
        ).evaluate()

    handles = list(generate_handles())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(
        isinstance(h, Handle) for h in handles
    ), "All generated items should be of type Handle."


def test_generate_with_using_contains(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    query = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            contains(body.name, "Handle"),
        )
    )

    handles = list(query.evaluate())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(
        isinstance(h, Handle) for h in handles
    ), "All generated items should be of type Handle."


def test_generate_with_using_in(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    query = an(
        entity(
            body := let(name="body", type_=Body, domain=world.bodies),
            in_("Handle", body.name),
        )
    )

    handles = list(query.evaluate())
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(
        isinstance(h, Handle) for h in handles
    ), "All generated items should be of type Handle."


def test_generate_with_using_and(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    query = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            contains(body.name, "Handle") & contains(body.name, "1"),
        )
    )

    handles = list(query.evaluate())
    assert len(handles) == 1, "Should generate at least one handle."
    assert all(
        isinstance(h, Handle) for h in handles
    ), "All generated items should be of type Handle."


def test_generate_with_using_or(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    query = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            contains(body.name, "Handle1") | contains(body.name, "Handle2"),
        )
    )

    handles = list(query.evaluate())
    assert len(handles) == 2, "Should generate at least one handle."
    assert all(
        isinstance(h, Handle) for h in handles
    ), "All generated items should be of type Handle."


def test_generate_with_using_multi_or(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    generate_handles_and_container1 = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            contains(body.name, "Handle1")
            | contains(body.name, "Handle2")
            | contains(body.name, "Container1"),
        )
    )

    handles_and_container1 = list(generate_handles_and_container1.evaluate())
    assert len(handles_and_container1) == 3, "Should generate at least one handle."


def test_generate_with_or_and(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_handles_and_container1():

        yield from an(
            entity(
                body := let(type_=Body, domain=world.bodies),
                or_(
                    and_(contains(body.name, "Handle"), contains(body.name, "1")),
                    and_(contains(body.name, "Container"), contains(body.name, "1")),
                ),
            )
        ).evaluate()

    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 2, "Should generate at least one handle."


def test_reevaluation_of_or_and_query(handles_and_containers_world):
    world = handles_and_containers_world

    query = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            or_(
                and_(contains(body.name, "Handle"), contains(body.name, "1")),
                and_(contains(body.name, "Container"), contains(body.name, "1")),
            ),
        )
    )

    handles_and_container1 = list(query.evaluate())
    assert (
        len(handles_and_container1) == 2
    ), "Should generate one handle and one container."
    handles_and_container1 = list(query.evaluate())
    assert (
        len(handles_and_container1) == 2
    ), "Re-eval: Should generate one handle and one container."


def test_generate_with_and_or(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_handles_and_container1():

        query = an(
            entity(
                body := let(type_=Body, domain=world.bodies),
                or_(contains(body.name, "Handle"), contains(body.name, "1")),
                or_(contains(body.name, "Container"), contains(body.name, "1")),
            )
        )
        # query._render_tree_()
        yield from query.evaluate()

    handles_and_container1 = list(generate_handles_and_container1())
    assert len(handles_and_container1) == 2, "Should generate at least one handle."


def test_generate_with_multi_and(handles_and_containers_world):
    world = handles_and_containers_world

    def generate_container1():

        query = an(
            entity(
                body := let(type_=Body, domain=world.bodies),
                contains(body.name, "n"),
                contains(body.name, "1"),
                contains(body.name, "C"),
            )
        )

        # query._render_tree_()
        yield from query.evaluate()

    all_solutions = list(generate_container1())
    assert len(all_solutions) == 1, "Should generate one container."
    assert isinstance(
        all_solutions[0], Container
    ), "The generated item should be of type Container."
    assert all_solutions[0].name == "Container1"


def test_reevaluate_with_multi_and(handles_and_containers_world):
    world = handles_and_containers_world

    query = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            contains(body.name, "n"),
            contains(body.name, "1"),
            contains(body.name, "C"),
        )
    )

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 1, "Should generate one container."
    assert isinstance(
        all_solutions[0], Container
    ), "The generated item should be of type Container."
    assert all_solutions[0].name == "Container1"
    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 1, "Re-eval: Should generate one container."
    assert isinstance(
        all_solutions[0], Container
    ), "Re-eval: The generated item should be of type Container."
    assert (
        all_solutions[0].name == "Container1"
    ), "Re-eval: The generated item should be of type Container."


def test_generate_with_more_than_one_source(handles_and_containers_world):
    world = handles_and_containers_world

    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    drawer_components = (container, handle, fixed_connection, prismatic_connection)

    solutions = a(
        set_of(
            drawer_components,
            container == fixed_connection.parent,
            handle == fixed_connection.child,
            container == prismatic_connection.child,
        )
    )

    all_solutions = list(solutions.evaluate())
    assert (
        len(all_solutions) == 2
    ), "Should generate components for two possible drawer."
    for sol in all_solutions:
        assert sol[container] == sol[fixed_connection].parent
        assert sol[handle] == sol[fixed_connection].child
        assert sol[prismatic_connection].child == sol[fixed_connection].parent


def test_generate_with_more_than_one_source_optimized(handles_and_containers_world):
    world = handles_and_containers_world

    fixed_connection = let(FixedConnection, world.connections)
    prismatic_connection = let(PrismaticConnection, world.connections)
    query = a(
        set_of(
            (fixed_connection, prismatic_connection),
            HasType(fixed_connection.parent, Container),
            HasType(fixed_connection.child, Handle),
            prismatic_connection.child == fixed_connection.parent,
        )
    )

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert (
        len(all_solutions) == 2
    ), "Should generate components for two possible drawer."
    for sol in all_solutions:
        assert isinstance(sol[fixed_connection].parent, Container)
        assert isinstance(sol[fixed_connection].child, Handle)
        assert sol[prismatic_connection].child == sol[fixed_connection].parent


def test_sources(handles_and_containers_world):

    world = let(type_=World, domain=handles_and_containers_world)
    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    drawer_components = (container, handle, fixed_connection, prismatic_connection)
    query = a(
        set_of(
            drawer_components,
            container == fixed_connection.parent,
            handle == fixed_connection.child,
            container == prismatic_connection.child,
        )
    )
    # render_tree(handle._sources_[0]._node_.root, use_dot_exporter=True, view=True)
    sources = list(query._sources_)
    assert len(sources) == 1, "Should have 1 source."
    assert (
        sources[0] is handles_and_containers_world
    ), "The source should be the world."


def test_the(handles_and_containers_world):
    world = handles_and_containers_world

    with pytest.raises(MultipleSolutionFound):
        handle = the(
            entity(
                body := let(type_=Handle, domain=world.bodies),
                body.name.startswith("Handle"),
            )
        ).evaluate()

    handle = the(
        entity(
            body := let(type_=Handle, domain=world.bodies),
            body.name.startswith("Handle1"),
        )
    ).evaluate()


def test_not_domain_mapping(handles_and_containers_world):
    world = handles_and_containers_world

    not_handle = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            not_(body.name.startswith("Handle")),
        )
    ).evaluate()
    all_not_handles = list(not_handle)
    assert len(all_not_handles) == 3, "Should generate 3 not handles"
    assert all(isinstance(b, Container) for b in all_not_handles)


def test_not_comparator(handles_and_containers_world):
    world = handles_and_containers_world

    not_handle = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            not_(contains(body.name, "Handle")),
        )
    ).evaluate()
    all_not_handles = list(not_handle)
    assert len(all_not_handles) == 3, "Should generate 3 not handles"
    assert all(isinstance(b, Container) for b in all_not_handles)


def test_not_and(handles_and_containers_world):
    world = handles_and_containers_world

    query = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            not_(contains(body.name, "Handle") & contains(body.name, "1")),
        )
    )

    all_not_handle1 = list(query.evaluate())
    assert len(all_not_handle1) == 5, "Should generate 5 bodies"
    assert all(
        h.name != "Handle1" for h in all_not_handle1
    ), "All generated items should satisfy query"


def test_not_or(handles_and_containers_world):
    world = handles_and_containers_world

    query = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            not_(contains(body.name, "Handle1") | contains(body.name, "Handle2")),
        )
    )

    all_not_handle1_or2 = list(query.evaluate())
    assert len(all_not_handle1_or2) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Handle2"] for h in all_not_handle1_or2
    ), "All generated items should satisfy query"


def test_not_and_or(handles_and_containers_world):
    world = handles_and_containers_world

    query = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            not_(
                or_(
                    and_(contains(body.name, "Handle"), contains(body.name, "1")),
                    and_(contains(body.name, "Container"), contains(body.name, "1")),
                )
            ),
        )
    )

    all_not_handle1_and_not_container1 = list(query.evaluate())
    assert len(all_not_handle1_and_not_container1) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Container1"]
        for h in all_not_handle1_and_not_container1
    ), "All generated items should satisfy query"
    # print(f"\nCache Search Count = {cache_search_count.values}")
    # print(f"\nCache Match Count = {cache_match_count.values}")
    # query._render_tree_()


def test_empty_list_literal(handles_and_containers_world):
    world = handles_and_containers_world

    query = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            not_(contains([], "Handle") & contains(body.name, "1")),
        )
    )
    results = list(query.evaluate())


def test_not_and_or_with_domain_mapping(handles_and_containers_world):
    world = handles_and_containers_world

    not_handle1_and_not_container1 = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            not_(
                and_(
                    or_(body.name.startswith("Handle"), body.name.endswith("1")),
                    or_(body.name.startswith("Container"), body.name.endswith("1")),
                )
            ),
        )
    )

    all_not_handle1_and_not_container1 = list(not_handle1_and_not_container1.evaluate())
    assert len(all_not_handle1_and_not_container1) == 4, "Should generate 4 bodies"
    assert all(
        h.name not in ["Handle1", "Container1"]
        for h in all_not_handle1_and_not_container1
    ), "All generated items should satisfy query"


def test_generate_with_using_decorated_predicate(handles_and_containers_world):
    """
    Test that symbolic functions can be used inside and outside of queries
    """
    world = handles_and_containers_world

    @symbolic_function
    def is_handle(body_: Body):
        return body_.name.startswith("Handle")

    query_kwargs = an(
        entity(body := let(type_=Body, domain=world.bodies), is_handle(body_=body))
    )
    query_args = an(
        entity(body := let(type_=Body, domain=world.bodies), is_handle(body))
    )

    handles = list(query_kwargs.evaluate())
    assert handles == list(query_args.evaluate()), "Both queries should generate the same items."
    assert len(handles) == 3, "Should generate at least one handle."
    assert all(
        isinstance(h, Handle) for h in handles
    ), "All generated items should be of type Handle."
    assert is_handle(world.bodies[0])


def test_generate_with_using_inherited_predicate(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    @dataclass
    class HaveSameFirstCharacter(Predicate):
        body1: Body
        body2: Body
        body3: Body

        def __call__(self):
            return self.body1.name[0] == self.body2.name[0] == self.body3.name[0]

    query = a(
        set_of(
            (
                body1 := let(Body, world.bodies),
                body2 := let(Body, world.bodies),
                body3 := let(Body, world.bodies),
            ),
            body1 != body2,
            body2 != body3,
            body3 != body1,
            HaveSameFirstCharacter(
                body1,
                body2,
                body3,
            ),
        )
    )

    body_pairs = list(query.evaluate())
    body_pairs = [
        (body_pair[body1], body_pair[body2], body_pair[body3])
        for body_pair in body_pairs
    ]

    expected = factorial(
        len([h for h in world.bodies if isinstance(h, Handle)])
    ) + factorial(len([c for c in world.bodies if isinstance(c, Container)]))
    assert len(body_pairs) == expected, "Should generate at least one handle."
    assert all(
        HaveSameFirstCharacter(b1, b2, b3)() for b1, b2, b3 in body_pairs
    ), "All generated items should satisfy the predicate."
    assert all(
        not HaveSameFirstCharacter(b1, b2, b3)()
        for b1 in world.bodies
        for b2 in world.bodies
        for b3 in world.bodies
        if b1 != b2 and b2 != b3 and b1 != b3 and (b1, b2, b3) not in body_pairs
    ), ("All not generated items " "should not satisfy the " "predicate.")


def test_select_predicate(handles_and_containers_world):
    """
    Test the generation of handles in the HandlesAndContainersWorld.
    """
    world = handles_and_containers_world

    @dataclass
    class HasName(Predicate):
        body: Body
        name: str

        def __call__(self):
            return self.body.name == self.name

    body = let(Body, world.bodies)
    has_name = HasName(body, "Handle1")
    query = the(entity(has_name, has_name))

    handle1 = query.evaluate()
    assert isinstance(handle1, HasName), "Should generate a handle."
    assert handle1.body.name == "Handle1", "The generated handle should have the expected name."


def test_literal_predicate(handles_and_containers_world):
    world = handles_and_containers_world

    @dataclass
    class HasName(Predicate):
        body: Body
        name: str

        def __call__(self):
            return self.body.name == self.name

    has_name = HasName(world.bodies[0], world.bodies[0].name)
    with pytest.raises(LiteralConditionError):
        query = the(entity(let(Body, world.bodies), has_name))



def test_contains_type():
    fb1_fruits = [Apple("apple"), Body("Body1")]
    fb2_fruits = [Body("Body3"), Body("Body2")]
    fb1 = FruitBox("FruitBox1", fb1_fruits)
    fb2 = FruitBox("FruitBox2", fb2_fruits)

    fruit_box_query = an(
        entity(fb := let(FruitBox, domain=None), ContainsType(fb.fruits, Apple))
    )

    query_result = list(fruit_box_query.evaluate())
    assert len(query_result) == 1, "Should generate 1 fruit box."


def test_equivalent_to_contains_type_using_exists():
    fb1_fruits = [Apple("apple"), Body("Body1")]
    fb2_fruits = [Body("Body3"), Body("Body2")]
    fb1 = FruitBox("FruitBox1", fb1_fruits)
    fb2 = FruitBox("FruitBox2", fb2_fruits)
    fruit_box_query = an(
        entity(
            fb := let(FruitBox, domain=None),
            exists(fb, HasType(flatten(fb.fruits), Apple)),
        )
    )

    query_result = list(fruit_box_query.evaluate())
    assert len(query_result) == 1, "Should generate 1 fruit box."


def test_double_not(handles_and_containers_world):
    world = handles_and_containers_world

    query = an(
        entity(
            body := let(type_=Body, domain=world.bodies),
            not_(not_(contains(body.name, "Handle"))),
        )
    )
    results = list(query.evaluate())
    assert all("Handle" in r.name for r in results)


def test_reuse_of_subquery_with_not(handles_and_containers_world):
    world = handles_and_containers_world
    body = let(type_=Body, domain=world.bodies)
    sub_query = contains(body.name, "Handle")
    query = an(
        entity(
            body,
            sub_query,
            body.name.endswith("1"),
        )
    )
    query_with_not = an(
        entity(
            body,
            not_(sub_query),
            body.name.endswith("1"),
        )
    )
    results = list(query.evaluate())
    results_with_not = list(query_with_not.evaluate())
    assert len(results) == 1
    assert isinstance(results[0], Handle)
    assert len(results_with_not) == 1
    assert isinstance(results_with_not[0], Container)


def test_unsupported_negation(handles_and_containers_world):
    world = handles_and_containers_world
    body = let(type_=Body, domain=world.bodies)
    with pytest.raises(UnsupportedNegation):
        query = not_(
            an(
                entity(
                    body,
                    body.name.endswith("1"),
                )
            )
        )

    with pytest.raises(UnsupportedNegation):
        query = an(not_(entity(body, body.name.endswith("1"))))


def test_quantified_query(handles_and_containers_world):
    world = handles_and_containers_world

    def get_quantified_query(quantification: ResultQuantificationConstraint):
        query = an(
            entity(
                body := let(type_=Body, domain=world.bodies),
                contains(body.name, "Handle"),
            ),
            quantification=quantification,
        )
        return query

    results = list(get_quantified_query(AtLeast(3)).evaluate())
    assert len(results) == 3
    results = list(get_quantified_query(Range(AtLeast(2), AtMost(4))).evaluate())
    assert len(results) == 3
    with pytest.raises(LessThanExpectedNumberOfSolutions):
        list(get_quantified_query(AtLeast(4)).evaluate())
    with pytest.raises(GreaterThanExpectedNumberOfSolutions):
        list(get_quantified_query(AtMost(2)).evaluate())
    with pytest.raises(GreaterThanExpectedNumberOfSolutions):
        list(get_quantified_query(Exactly(2)).evaluate())
    with pytest.raises(LessThanExpectedNumberOfSolutions):
        list(get_quantified_query(Exactly(4)).evaluate())


def test_count(handles_and_containers_world):
    world = handles_and_containers_world
    query = count(
        entity(body:=let(type_=Body, domain=world.bodies),
               contains(body.name, "Handle"),
        )
    )
    assert query.evaluate() == len([b for b in world.bodies if "Handle" in b.name])


def test_count_without_entity(handles_and_containers_world):
    world = handles_and_containers_world
    query = count(let(type_=Body, domain=world.bodies))
    assert query.evaluate() == len(world.bodies)


def test_order_by(handles_and_containers_world):
    names = ["Handle1", "Handle1", "Handle2", "Container1", "Container1", "Container3"]
    body_name = let(str, domain=names)
    query = an(entity(body_name).order_by(variable=body_name, descending=False))
    assert list(query.evaluate()) == sorted(names, reverse=False)


def test_sum(handles_and_containers_world):
    heights = [1, 2, 3, 4, 5]
    heights_var = let(int, domain=heights)
    query = eql.sum(entity(heights_var))
    assert query.evaluate() == sum(heights)


def test_average(handles_and_containers_world):
    heights = [1, 2, 3, 4, 5]
    heights_var = let(int, domain=heights)
    query = eql.average(entity(heights_var))
    assert query.evaluate() == sum(heights) / len(heights)


def test_sum_on_empty_list(handles_and_containers_world):
    empty_list = []
    empty_var = let(int, domain=empty_list)
    query = eql.sum(entity(empty_var))
    assert query.evaluate() is None


def test_sum_without_entity():
    heights = [1, 2, 3, 4, 5]
    heights_var = let(int, domain=heights)
    query = eql.sum(heights_var)
    assert query.evaluate() == sum(heights)

def test_limit(handles_and_containers_world):
    world = handles_and_containers_world
    query = an(
        entity(
            body := let(type_=Body, domain=world.bodies), contains(body.name, "Handle")
        )
    )
    assert len(list(query.evaluate(limit=2))) == 2
    assert len(list(query.evaluate(limit=1))) == 1
    assert len(list(query.evaluate(limit=3))) == 3
    with pytest.raises(NonPositiveLimitValue):
        list(query.evaluate(limit=0))
    with pytest.raises(NonPositiveLimitValue):
        list(query.evaluate(limit=-1))
    with pytest.raises(NonPositiveLimitValue):
        list(query.evaluate(limit="0"))


def test_unification_dict(handles_and_containers_world):
    drawer = let(Drawer, domain=None)
    drawer_1 = an(entity(drawer))
    handle = let(Handle, domain=None)
    query = a(set_of((drawer, handle), drawer.handle.name == handle.name))
    results = list(query.evaluate())
    assert results[0][drawer] is results[0][drawer_1]


def test_distinct_entity():
    names = ["Handle1", "Handle1", "Handle2", "Container1", "Container1", "Container3"]
    body_name = let(str, domain=names)
    query = an(
        entity(
            body_name,
            body_name.startswith("Handle"),
        ).distinct()
    )
    results = list(query.evaluate())
    assert len(results) == 2


def test_distinct_set_of():
    handle_names = ["Handle1", "Handle1", "Handle2"]
    container_names = ["Container1", "Container1", "Container3"]
    handle_name = let(str, domain=handle_names)
    container_name = let(str, domain=container_names)
    query = a(set_of((handle_name, container_name)).distinct())
    results = list(query.evaluate())
    assert len(results) == 4
    assert set(tuple(r.values()) for r in results) == {
        (handle_names[0], container_names[0]),
        (handle_names[0], container_names[2]),
        (handle_names[2], container_names[0]),
        (handle_names[2], container_names[2]),
    }


def test_distinct_on():
    handle_names = ["Handle1", "Handle1", "Handle2"]
    container_names = ["Container1", "Container1", "Container3"]
    handle_name = let(str, domain=handle_names)
    container_name = let(str, domain=container_names)
    query = a(set_of((handle_name, container_name)).distinct(handle_name))
    results = list(query.evaluate())
    assert len(results) == 2
    assert set(tuple(r.values()) for r in results) == {
        (handle_names[0], container_names[0]),
        (handle_names[2], container_names[0]),
    }


def test_max_min_no_variable():
    values = [2, 1, 3, 5, 4]
    value = let(int, domain=values)

    max_query = eql.max(entity(value))
    assert max_query.evaluate() == max(values)

    min_query = eql.min(entity(value))
    assert min_query.evaluate() == min(values)
def test_max_min_without_entity():
    values = [2, 1, 3, 5, 4]
    value = let(int, domain=values)

    max_query = eql.max(value)
    assert max_query.evaluate() == max(values)

    min_query = eql.min(value)
    assert min_query.evaluate() == min(values)


def test_max_min_with_empty_list():
    empty_list = []
    value = let(int, domain=empty_list)

    max_query = eql.max(entity(value))
    assert max_query.evaluate() is None

    min_query = eql.min(entity(value))
    assert min_query.evaluate() is None


def test_order_by_key():
    names = ["Handle1", "handle2", "Handle3", "container1", "Container2", "container3"]
    body_name = let(str, domain=names)
    key = lambda x: int(x[-1])
    query = an(
        entity(body_name).order_by(
            variable=body_name,
            key=key,
            descending=True,
        )
    )
    results = list(query.evaluate())
    assert results == sorted(names, key=key, reverse=True)


def test_distinct_with_order_by():
    values = [5, 1, 1, 2, 1, 4, 3, 3, 5]
    values_var = let(int, domain=values)
    query = an(entity(values_var).distinct().order_by(variable=values_var, descending=False))
    results = list(query.evaluate())
    assert results == [1, 2, 3, 4, 5]