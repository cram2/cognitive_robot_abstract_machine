"""
An object of a class with several mapped base classes has one row, under one id, in the
table of every one of them, and comes back through a field typed with any of them.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, inspect, select

from krrood.entity_query_language.factories import an, entity, set_of, variable
from krrood.ormatic.data_access_objects.dao import StoredObjectRow
from krrood.ormatic.exceptions import (
    PartOfDetachedObjectNotLoadedError,
    UnsupportedResultLoadingError,
)
from krrood.ormatic.result_loading import (
    RelationshipLoading,
    ResultContent,
    ResultLoading,
    SubclassLoading,
)
from krrood.ormatic.stored_object import StoredObjectDAO
from krrood.ormatic.data_access_objects.helper import to_dao
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
        relationship_loading=relationship_loading, subclass_loading=subclass_loading
    )
    for relationship_loading in RelationshipLoading
    for subclass_loading in SubclassLoading
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
    Selecting a secondary base class loads each object as its own primary class,
    however the result is loaded.
    """
    items = store_items_of_one_hierarchy(session)
    hinged = variable(HasHinges, domain=None)
    query = an(entity(hinged).where(hinged.hinge_count > 1))

    result = eql_to_sql(query, session, result_loading=result_loading).evaluate()

    assert sorted(
        (dao.from_dao() for dao in result), key=lambda loaded: loaded.label
    ) == [item for item in items if item.hinge_count > 1]


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
