"""
An object of a class with several mapped base classes has one row, under one id, in the
table of every one of them, and comes back through a field typed with any of them.
"""

from __future__ import annotations

import gc

import pytest
from sqlalchemy import event, func, inspect, select, update
from sqlalchemy.orm import ORMExecuteState

from krrood.entity_query_language.factories import an, entity, set_of, variable
from krrood.ormatic.data_access_objects.stored_object_row import StoredObjectRow
from krrood.ormatic.exceptions import (
    PartOfDetachedObjectNotLoadedError,
    UnsupportedResultLoadingError,
)
from krrood.ormatic.result_loading import (
    RelationshipLoading,
    ResultContent,
    ResultLoading,
    SelectedObjectRestriction,
    SubclassLoading,
)
from krrood.ormatic.stored_object import StoredObjectDAO
from krrood.symbol_graph.symbol_graph import GarbageCollectionPause
from krrood.ormatic.data_access_objects.helper import (
    get_mapping_data_access_object_class,
    to_dao,
)
from krrood.ormatic.eql_interface import eql_to_sql
from ..dataset.multiple_inheritance import (
    FramedFurnishing,
    HasHinges,
    Hinge,
    InspectedCloset,
    InventoryItem,
    Inventory,
    LandmarkFrame,
    TaggedFurnishing,
    WalkInCloset,
    Closet,
    Furnishing,
)

# %% generated schema


def test_one_class_maps_the_table_of_every_stored_object():
    """
    However many interfaces are loaded, the rows of all stored objects are created and
    loaded through one class.
    """
    assert StoredObjectRow.mapped_class() is StoredObjectDAO


def test_root_tables_take_their_ids_from_the_table_of_every_stored_object():
    """
    The table of a class without mapped base classes refers with its key to the table
    holding the ids of all stored objects.
    """
    from ..dataset.ormatic_interface import HingeDAO, InventoryItemDAO

    for dao_class in [InventoryItemDAO, HingeDAO]:
        [foreign_key] = dao_class.__table__.c.database_id.foreign_keys
        assert foreign_key.column.table is StoredObjectDAO.__table__


def test_fields_of_a_base_class_live_in_its_own_table():
    """
    A field a secondary base class declares is a column of that class's table only.
    """
    from ..dataset.ormatic_interface import HasHingesDAO, ClosetDAO

    assert "hinge_count" in HasHingesDAO.__table__.columns
    assert "hinge_count" not in ClosetDAO.__table__.columns


def test_an_interface_is_a_table_without_fields():
    """
    An interface a field is typed with gets a table holding only its key and the type
    column of its hierarchy.
    """
    from ..dataset.ormatic_interface import ReferenceFrameDAO, TaggedFrameDAO

    for dao_class in [ReferenceFrameDAO, TaggedFrameDAO]:
        table = dao_class.__table__
        assert set(table.columns) == {
            table.c.database_id,
            inspect(dao_class).polymorphic_on,
        }


# %% one row per base class


def test_an_object_has_one_row_in_the_table_of_every_base_class(session, database):
    """
    The rows of an object in the tables of all its mapped base classes share its id, and
    the ancestor two of its bases share has one row of it.
    """
    from ..dataset.ormatic_interface import (
        HasHingesDAO,
        InspectedDAO,
        InventoryItemDAO,
        ClosetDAO,
    )

    closet = InspectedCloset(label="hall", leg_count=4, shelf_count=2, inspector="Ada")
    dao = to_dao(closet)
    session.add(dao)
    session.commit()

    tables = [InventoryItemDAO, HasHingesDAO, InspectedDAO, ClosetDAO]
    ids = {
        table.__name__: session.scalars(select(table.__table__.c.database_id)).all()
        for table in tables
    }

    assert ids == {table.__name__: [dao.database_id] for table in tables}


# %% round trip


def test_fields_typed_with_any_base_class_round_trip(session, database):
    """
    Fields typed with a secondary base class, a mixin without mapped ancestors, a plain
    interface and a generic interface hold objects of classes that are primarily
    something else, and give them back.
    """
    from ..dataset.ormatic_interface import InventoryDAO

    furnishing_frame = FramedFurnishing(label="desk", leg_count=4)
    inventory = Inventory(
        hinged=Closet(
            label="bedroom",
            leg_count=4,
            shelf_count=3,
            hinge_count=4,
            hinges=[Hinge(stiffness=1.5), Hinge(stiffness=2.0)],
        ),
        inspected=[
            InspectedCloset(label="hall", leg_count=4, shelf_count=2, inspector="Ada")
        ],
        frame=furnishing_frame,
        frames=[LandmarkFrame(landmark_name="door"), furnishing_frame],
        tagged_frame=TaggedFurnishing(label="shelf", leg_count=2),
    )

    session.add(to_dao(inventory))
    session.commit()
    session.expunge_all()

    reconstructed = session.scalars(select(InventoryDAO)).one().from_dao()

    assert reconstructed == inventory
    assert reconstructed.frames[1] is reconstructed.frame


def test_a_subclass_inherits_the_rows_of_its_parents_secondary_bases(session, database):
    """
    A subclass of a class with a secondary base class keeps the fields of that base.
    """
    from ..dataset.ormatic_interface import WalkInClosetDAO

    closet = WalkInCloset(
        label="master",
        leg_count=0,
        shelf_count=6,
        floor_area=4.5,
        hinge_count=2,
        hinges=[Hinge(stiffness=3.0)],
    )

    session.add(to_dao(closet))
    session.commit()
    session.expunge_all()

    reconstructed = session.scalars(select(WalkInClosetDAO)).one().from_dao()

    assert reconstructed == closet


def test_the_row_of_every_stored_object_loads_the_object(session, database):
    """
    Loading an object by its id from the root table gives back the object of its own
    class.
    """
    closet = InspectedCloset(label="hall", leg_count=4, shelf_count=2, inspector="Ada")
    dao = to_dao(closet)
    session.add(dao)
    session.commit()
    database_id = dao.database_id
    session.expunge_all()

    reconstructed = session.get(StoredObjectDAO, database_id).from_dao()

    assert reconstructed == closet


def test_the_rows_of_stored_objects_load_their_daos_in_batches_and_in_order(
    session, database
):
    """
    The DAOs of stored objects' rows come back in the rows' order, also when every
    statement binds a single id.
    """
    items = store_items_of_one_hierarchy(session)
    session.expunge_all()
    rows = session.scalars(
        select(StoredObjectDAO).order_by(StoredObjectDAO.database_id.desc())
    ).all()

    daos = StoredObjectDAO.object_daos(
        session=session, stored_objects=rows, ids_per_statement=1
    )

    assert [dao.database_id for dao in daos] == [row.database_id for row in rows]
    assert sorted(dao.from_dao().label for dao in daos) == sorted(
        item.label for item in items
    )


def test_only_one_row_is_kept_per_object_of_the_root_table(session, database):
    """
    Storing an inventory stores one root row per object, however many fields refer to
    it.
    """
    furnishing_frame = FramedFurnishing(label="desk", leg_count=4)
    inventory = Inventory(frame=furnishing_frame, frames=[furnishing_frame])

    session.add(to_dao(inventory))
    session.commit()

    assert session.scalar(select(func.count()).select_from(StoredObjectDAO)) == 2


# %% detached data access objects


def test_a_detached_dao_refuses_to_convert_without_its_parts(session, database):
    """
    A loaded DAO whose session is gone cannot load the parts holding the fields of its
    secondary bases, and says so instead of converting without them.
    """
    from ..dataset.ormatic_interface import ClosetDAO

    session.add(
        to_dao(
            Closet(
                label="bedroom",
                leg_count=4,
                shelf_count=3,
                hinge_count=4,
                hinges=[Hinge(stiffness=1.5)],
            )
        )
    )
    session.commit()
    session.expunge_all()
    dao = session.scalars(select(ClosetDAO)).one()
    session.expunge(dao)

    with pytest.raises(PartOfDetachedObjectNotLoadedError):
        dao.from_dao()


# %% deletion


def test_deleting_an_object_deletes_its_rows_in_every_table(session, database):
    """
    Deleting an object's DAO removes the object's rows from the table of every stored
    object and from the tables of all its mapped base classes.
    """
    from ..dataset.ormatic_interface import (
        HasHingesDAO,
        InspectedDAO,
        InventoryItemDAO,
        ClosetDAO,
    )

    session.add(
        to_dao(
            InspectedCloset(label="hall", leg_count=4, shelf_count=2, inspector="Ada")
        )
    )
    session.commit()
    session.expunge_all()

    session.delete(session.scalars(select(ClosetDAO)).one())
    session.commit()

    tables = [StoredObjectDAO, InventoryItemDAO, HasHingesDAO, InspectedDAO, ClosetDAO]
    row_counts = {
        table.__name__: session.scalar(select(func.count()).select_from(table))
        for table in tables
    }
    assert row_counts == {table.__name__: 0 for table in tables}


# %% queries


def store_closets_and_plain_hinged_items(session) -> None:
    """
    Store a closet with four hinges in an inventory, and a hinged item that is nothing
    else with one hinge.

    :param session: The session to store them with.
    """
    session.add(
        to_dao(
            Inventory(
                hinged=Closet(
                    label="bedroom", leg_count=4, shelf_count=3, hinge_count=4
                )
            )
        )
    )
    session.add(to_dao(HasHinges(label="gate", hinge_count=1)))
    session.commit()


def test_a_query_for_a_base_class_finds_objects_of_other_primary_classes(
    session, database
):
    """
    Selecting the objects of a secondary base class finds those whose primary class is
    another one, loaded as their own class.
    """
    store_closets_and_plain_hinged_items(session)
    hinged = variable(HasHinges, domain=None)
    query = an(entity(hinged).where(hinged.hinge_count > 1))

    result = eql_to_sql(query, session).evaluate()

    assert [dao.from_dao() for dao in result] == [
        Closet(label="bedroom", leg_count=4, shelf_count=3, hinge_count=4)
    ]


def test_a_query_reads_a_field_a_secondary_base_class_declares(session, database):
    """
    A condition on a field of a secondary base class reads it from that class's table.
    """
    store_closets_and_plain_hinged_items(session)
    closet = variable(Closet, domain=None)
    query = an(entity(closet).where(closet.hinge_count > 1))

    result = eql_to_sql(query, session).evaluate()

    assert [dao.label for dao in result] == ["bedroom"]


def test_a_query_follows_a_field_typed_with_a_secondary_base_class(session, database):
    """
    A condition on a field reached through a field typed with a secondary base class
    reads the declared class's table.
    """
    store_closets_and_plain_hinged_items(session)
    inventory = variable(Inventory, domain=None)
    query = an(entity(inventory).where(inventory.hinged.hinge_count > 1))

    result = eql_to_sql(query, session).evaluate()

    assert [dao.from_dao().hinged.label for dao in result] == ["bedroom"]


def test_conditions_on_one_path_share_its_join(session, database):
    """
    Two conditions on fields reached through the same field typed with a secondary base
    class read the same row of the declared class's table.
    """
    store_closets_and_plain_hinged_items(session)
    inventory = variable(Inventory, domain=None)
    query = an(
        entity(inventory).where(
            inventory.hinged.hinge_count > 1, inventory.hinged.label == "bedroom"
        )
    )

    translator = eql_to_sql(query, session)

    assert str(translator.sql_query).count('"HasHingesDAO" AS') == 1
    assert [dao.from_dao().hinged.label for dao in translator.evaluate()] == ["bedroom"]


# %% result loading

EVERY_RESULT_LOADING = [
    ResultLoading(
        relationship_loading=relationship_loading,
        subclass_loading=subclass_loading,
        selected_object_restriction=selected_object_restriction,
        pauses_garbage_collection=pauses_garbage_collection,
    )
    for relationship_loading in RelationshipLoading
    for subclass_loading in SubclassLoading
    for selected_object_restriction in SelectedObjectRestriction
    for pauses_garbage_collection in [False, True]
]


def store_items_of_one_hierarchy(session) -> list:
    """
    Store items of four classes below one mapped root, one of them holding the part of a
    secondary base.

    :param session: The session to store them with.
    :return: The stored items, ordered by label.
    """
    items = [
        Closet(label="bedroom", leg_count=4, shelf_count=3, hinge_count=4),
        HasHinges(label="gate", hinge_count=1),
        WalkInCloset(
            label="hall", leg_count=4, shelf_count=6, hinge_count=2, floor_area=3.5
        ),
        Furnishing(label="stool", leg_count=3),
    ]
    for item in items:
        session.add(to_dao(item))
    session.commit()
    return items


@pytest.mark.parametrize("result_loading", EVERY_RESULT_LOADING, ids=repr)
def test_every_result_loading_loads_the_objects_of_a_primary_root_as_their_classes(
    session, database, result_loading
):
    """
    Selecting the root of a hierarchy loads each object as its own class, however the
    result is loaded.
    """
    items = store_items_of_one_hierarchy(session)
    item = variable(InventoryItem, domain=None)
    query = an(entity(item).ordered_by(variable=item.label))

    result = eql_to_sql(query, session, result_loading=result_loading).evaluate()

    assert [dao.from_dao() for dao in result] == items


@pytest.mark.parametrize("result_loading", EVERY_RESULT_LOADING, ids=repr)
def test_every_result_loading_loads_the_objects_of_a_secondary_base_as_their_classes(
    session, database, result_loading
):
    """
    Selecting a secondary base class loads each object as its own primary class, however
    the result is loaded.
    """
    items = store_items_of_one_hierarchy(session)
    hinged = variable(HasHinges, domain=None)
    query = an(entity(hinged).where(hinged.hinge_count > 1))

    result = eql_to_sql(query, session, result_loading=result_loading).evaluate()

    assert sorted(
        (dao.from_dao() for dao in result), key=lambda loaded: loaded.label
    ) == [
        item for item in items if isinstance(item, HasHinges) and item.hinge_count > 1
    ]


def test_database_ids_are_the_ids_of_the_selected_objects_in_their_order(
    session, database
):
    """
    Evaluating for the database ids gives the ids of the data access objects the query
    selects, in the same order.
    """
    store_items_of_one_hierarchy(session)
    item = variable(InventoryItem, domain=None)
    query = an(entity(item).ordered_by(variable=item.label))

    ids = eql_to_sql(
        query,
        session,
        result_loading=ResultLoading(result_content=ResultContent.DATABASE_IDS),
    ).evaluate()

    assert ids == [dao.database_id for dao in eql_to_sql(query, session).evaluate()]


@pytest.mark.parametrize(
    "relationship_loading, hinges_unloaded",
    [(RelationshipLoading.AS_MAPPED, False), (RelationshipLoading.ON_ACCESS, True)],
)
def test_relationships_load_on_access_only_when_asked_to(
    session, database, relationship_loading, hinges_unloaded
):
    """
    Loading relationships on access leaves a relationship its mapping loads eagerly
    unloaded.
    """
    session.add(to_dao(HasHinges(label="gate", hinges=[Hinge(stiffness=1.0)])))
    session.commit()
    session.expunge_all()
    hinged = variable(HasHinges, domain=None)
    query = an(entity(hinged))

    [dao] = eql_to_sql(
        query,
        session,
        result_loading=ResultLoading(relationship_loading=relationship_loading),
    ).evaluate()

    assert ("hinges" in inspect(dao).unloaded) == hinges_unloaded


def test_relationships_load_on_access_also_where_a_subclass_declares_them(
    session, database
):
    """
    Loading relationships on access also leaves unloaded the relationships that a
    subclass of the selected class declares.
    """
    session.add(to_dao(HasHinges(label="gate", hinges=[Hinge(stiffness=1.0)])))
    session.commit()
    session.expunge_all()
    item = variable(InventoryItem, domain=None)

    [dao] = eql_to_sql(
        an(entity(item)),
        session,
        result_loading=ResultLoading(
            relationship_loading=RelationshipLoading.ON_ACCESS
        ),
    ).evaluate()

    assert "hinges" in inspect(dao).unloaded


def test_a_distinct_query_for_a_hierarchy_keeps_its_order(session, database):
    """
    A distinct query that orders the objects of a class with subclasses gives them in
    its order.
    """
    items = store_items_of_one_hierarchy(session)
    item = variable(InventoryItem, domain=None)
    query = an(entity(item).ordered_by(variable=item.label).distinct())

    result = eql_to_sql(query, session).evaluate()

    assert [dao.label for dao in result] == [stored.label for stored in items]


def test_a_limited_query_loads_the_objects_it_selected_when_joining_its_selection(
    session, database
):
    """
    Loading by joining the selection still loads the objects a limited query selected,
    though running the selection again would select others.
    """
    store_items_of_one_hierarchy(session)
    item = variable(InventoryItem, domain=None)
    query = an(entity(item).ordered_by(variable=item.label)).limit(n=1)
    item_dao_class = get_mapping_data_access_object_class(InventoryItem)
    statements = []

    def rename_the_first_item_after_the_selection(state: ORMExecuteState) -> None:
        if not state.is_select:
            return
        statements.append(state.statement)
        if len(statements) == 2:
            state.session.execute(
                update(item_dao_class)
                .where(item_dao_class.label == "bedroom")
                .values(label="zzz")
                .execution_options(synchronize_session=False)
            )

    event.listen(session, "do_orm_execute", rename_the_first_item_after_the_selection)
    try:
        [dao] = eql_to_sql(
            query,
            session,
            result_loading=ResultLoading(
                subclass_loading=SubclassLoading.ONE_QUERY_PER_CLASS,
                selected_object_restriction=SelectedObjectRestriction.SELECTION_JOIN,
            ),
        ).evaluate()
    finally:
        event.remove(
            session, "do_orm_execute", rename_the_first_item_after_the_selection
        )

    assert isinstance(dao.from_dao(), Closet)


def test_a_set_of_query_refuses_to_load_its_rows_by_class(session, database):
    """
    A ``set_of`` query loads the rows it selects as they are.
    """
    store_items_of_one_hierarchy(session)
    item = variable(InventoryItem, domain=None)
    translator = eql_to_sql(
        an(set_of(item)),
        session,
        result_loading=ResultLoading(
            subclass_loading=SubclassLoading.ONE_QUERY_PER_CLASS
        ),
    )

    with pytest.raises(UnsupportedResultLoadingError):
        translator.evaluate()


@pytest.mark.parametrize(
    "selected_class, subclass_loading",
    [
        (Hinge, SubclassLoading.AS_MAPPED),
        (InventoryItem, SubclassLoading.ONE_QUERY_PER_CLASS),
        (HasHinges, SubclassLoading.ONE_QUERY_PER_CLASS),
    ],
)
def test_the_result_loading_of_a_query_loads_by_class_where_classes_differ(
    selected_class, subclass_loading
):
    """
    The result loading suited to a query loads relationships on access, and loads by
    class only where the selected class's objects may be of several classes.
    """
    query = an(entity(variable(selected_class, domain=None)))

    assert ResultLoading.from_query(query=query) == ResultLoading(
        relationship_loading=RelationshipLoading.ON_ACCESS,
        subclass_loading=subclass_loading,
    )


def test_the_result_loading_of_a_set_of_query_loads_its_rows_as_selected():
    """
    The result loading suited to a ``set_of`` query keeps the rows it selects.
    """
    query = an(set_of(variable(InventoryItem, domain=None)))

    assert ResultLoading.from_query(query=query) == ResultLoading(
        relationship_loading=RelationshipLoading.ON_ACCESS
    )


def test_a_translation_without_a_result_loading_takes_the_one_suited_to_its_query(
    session, database
):
    """
    Translating a query without a result loading evaluates it with the result loading
    suited to the query.
    """
    query = an(entity(variable(InventoryItem, domain=None)))

    translator = eql_to_sql(query, session)

    assert translator.result_loading == ResultLoading.from_query(query=query)


@pytest.mark.parametrize("selected_object_restriction", SelectedObjectRestriction)
def test_columns_are_the_values_of_each_selected_objects_own_class(
    session, database, selected_object_restriction
):
    """
    Evaluating for columns gives every selected object's values, including those of
    columns only its own class has, in the order of the query.
    """
    items = store_items_of_one_hierarchy(session)
    item = variable(InventoryItem, domain=None)
    query = an(entity(item).ordered_by(variable=item.label))

    rows = eql_to_sql(
        query,
        session,
        result_loading=ResultLoading(
            result_content=ResultContent.COLUMNS,
            selected_object_restriction=selected_object_restriction,
        ),
    ).evaluate()

    assert [row.label for row in rows] == [stored.label for stored in items]
    [walk_in_closet] = [stored for stored in items if isinstance(stored, WalkInCloset)]
    assert rows[items.index(walk_in_closet)].floor_area == walk_in_closet.floor_area


def test_the_garbage_collector_is_paused_only_inside_the_pause():
    """
    The garbage collector does not run inside a pause and runs again after it.
    """
    with GarbageCollectionPause():
        collector_running_inside = gc.isenabled()

    assert (collector_running_inside, gc.isenabled()) == (False, True)
