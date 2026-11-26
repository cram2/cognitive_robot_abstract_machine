"""
Collections read back from the database in the order, nesting and container classes they
were stored with.
"""

from __future__ import annotations

from sqlalchemy import event, select

from krrood.entity_query_language.factories import an, entity, variable
from krrood.ormatic.collection_shape import LevelColumnNames
from krrood.ormatic.data_access_objects.dao import AssociationDataAccessObject
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.ormatic.result_loading import RelationshipLoading, ResultLoading
from ..dataset.example_classes import Entity
from ..dataset.ordered_collections import (
    NestedAlternativelyMappedEntities,
    OrderedRecipe,
    RecipeStep,
)

# %% helpers


def make_recipe(*, first_number: int = 0) -> OrderedRecipe:
    """
    :param first_number: The number of the recipe's first step.
    :return: A recipe whose collections hold several elements in an order other than
        that of their numbers, and empty inner containers.
    """
    first, second, third = (
        RecipeStep(number=first_number + offset) for offset in (2, 0, 1)
    )
    return OrderedRecipe(
        steps=[first, second, third],
        step_tuple=(third, first),
        step_set={first, second},
        step_lists=[[third, first], [], [second]],
        steps_by_name={"zest": first, "bake": second},
        names_by_step={third: "mix", first: "serve"},
        step_lists_by_name={"prepare": [second, first], "rest": []},
        step_sets_by_step={first: {second, third}, second: set()},
        amounts_by_step={second: 3, first: 1},
        counts_by_name={"eggs": 2, "flour": 500},
        measurements_by_number={3: [1.5, 0.5], 1: []},
        number_sets=[{3, 1}, set()],
    )


def store_and_read_back(session, *, domain_object):
    """
    :param session: The session to store with.
    :param domain_object: The object to store.
    :return: The object, read back from the database.
    """
    dao = to_dao(domain_object)
    session.add(dao)
    session.commit()
    session.expunge_all()
    return session.scalars(select(type(dao))).one().from_dao()


# %% ordering


def test_every_association_relationship_reads_rows_in_order_of_their_positions():
    """
    Every collection stored in an association table is read back sorted by the positions
    of its rows, the outermost level first, since a database returns the rows of an
    unsorted query in any order.
    """
    from ..dataset.ormatic_interface import Base

    association_relationships = [
        relationship
        for mapper in Base.registry.mappers
        for relationship in mapper.relationships
        if relationship.parent is mapper
        and issubclass(relationship.mapper.class_, AssociationDataAccessObject)
    ]

    assert association_relationships
    for relationship in association_relationships:
        ordering_column_names = [column.name for column in relationship.order_by]
        assert ordering_column_names == [
            LevelColumnNames(depth=depth).index
            for depth in range(1, len(ordering_column_names) + 1)
        ]


def test_list_reads_back_in_its_order_whatever_order_its_rows_were_inserted_in(
    session, database
):
    """
    The rows of a list are sorted by the positions they hold, not by the order they were
    inserted in.
    """
    recipe = make_recipe()
    dao = to_dao(recipe)
    dao.steps = list(reversed(dao.steps))
    dao.step_lists = list(reversed(dao.step_lists))
    session.add(dao)
    session.commit()
    session.expunge_all()

    read_back = session.scalars(select(type(dao))).one().from_dao()

    assert read_back.steps == recipe.steps
    assert read_back.step_lists == recipe.step_lists


# %% round trips


def test_every_kind_of_collection_reads_back_as_it_was_stored(session, database):
    """
    Lists, tuples, sets, nested collections and dictionaries read back with their order,
    their empty inner containers, the classes of their containers and of their keys.
    """
    recipe = make_recipe()

    read_back = store_and_read_back(session, domain_object=recipe)

    assert read_back == recipe
    assert list(read_back.steps_by_name) == list(recipe.steps_by_name)
    assert list(read_back.measurements_by_number) == list(recipe.measurements_by_number)
    assert type(read_back.step_tuple) is tuple
    assert type(read_back.step_set) is set


def test_alternatively_mapped_objects_in_a_nested_collection_are_converted(
    session, database
):
    """
    An alternatively mapped object a nested collection holds reads back as its domain
    object, at its position.
    """
    entities = NestedAlternativelyMappedEntities(
        entity_lists=[[Entity(name="a"), Entity(name="b")], [], [Entity(name="c")]]
    )

    read_back = store_and_read_back(session, domain_object=entities)

    assert [
        [type(entity_), entity_.name]
        for entities_ in read_back.entity_lists
        for entity_ in entities_
    ] == [[Entity, "a"], [Entity, "b"], [Entity, "c"]]
    assert [len(entities_) for entities_ in read_back.entity_lists] == [2, 0, 1]


# %% loading for conversion


def count_statements_converting_recipes(session) -> int:
    """
    :param session: The session holding the stored recipes.
    :return: How many statements selecting every recipe and converting it take, loaded
        for conversion.
    """
    statements = []

    def count_statement(*arguments) -> None:
        statements.append(arguments)

    recipe = variable(OrderedRecipe, domain=None)
    event.listen(session.bind, "after_cursor_execute", count_statement)
    try:
        for dao in eql_to_sql(
            an(entity(recipe)),
            session,
            result_loading=ResultLoading(
                relationship_loading=RelationshipLoading.FOR_CONVERSION
            ),
        ).evaluate():
            dao.from_dao()
    finally:
        event.remove(session.bind, "after_cursor_execute", count_statement)
    session.expunge_all()
    return len(statements)


def test_converting_nested_and_keyed_collections_loaded_for_conversion_takes_statements_per_relationship(
    session, database
):
    """
    Selecting and converting objects with nested and keyed collections, loaded for
    conversion, takes as many statements for many objects as for one.
    """
    session.add(to_dao(make_recipe()))
    session.commit()
    session.expunge_all()
    statements_for_one = count_statements_converting_recipes(session)
    for first_number in range(10, 50, 10):
        session.add(to_dao(make_recipe(first_number=first_number)))
    session.commit()
    session.expunge_all()

    assert count_statements_converting_recipes(session) == statements_for_one
