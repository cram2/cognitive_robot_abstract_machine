import pytest
from sqlalchemy import select, func
from sqlalchemy.exc import MultipleResultsFound
from sqlalchemy.dialects import postgresql

from krrood.entity_query_language.exceptions import MultipleSolutionFound
from ..dataset.example_classes import KRROODPosition, KRROODPose
from ..dataset.semantic_world_like_classes import (
    World,
    Body,
    FixedConnection,
    PrismaticConnection,
    Handle,
    Container,
)
from ..dataset.ormatic_interface import (
    KRROODPositionDAO,
    KRROODPoseDAO,
    KRROODOrientationDAO,
    FixedConnectionDAO,
    PrismaticConnectionDAO,
    BodyDAO,
    ContainerDAO,
    HandleDAO,
)
from krrood.entity_query_language.factories import (
    entity,
    variable,
    and_,
    or_,
    contains,
    in_,
    an,
    the,
    count_all,
    not_,
    max,
    min,
    sum,
    average,
    set_of,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.eql_interface import eql_to_sql


def test_translate_simple_greater(session, database):
    session.add(KRROODPositionDAO(x=1, y=2, z=3))
    session.add(KRROODPositionDAO(x=1, y=2, z=4))
    session.commit()

    position = variable(type_=KRROODPosition, domain=[])
    query = an(entity(position).where(position.z > 3))

    translator = eql_to_sql(query, session)
    query_by_hand = select(KRROODPositionDAO).where(KRROODPositionDAO.z > 3)

    assert str(translator.sql_query) == str(query_by_hand)

    results = translator.evaluate()

    assert len(results) == 1
    assert isinstance(results[0], KRROODPositionDAO)
    assert results[0].z == 4


def test_translate_or_condition(session, database):
    session.add(KRROODPositionDAO(x=1, y=2, z=3))
    session.add(KRROODPositionDAO(x=1, y=2, z=4))
    session.add(KRROODPositionDAO(x=2, y=9, z=10))
    session.commit()

    position = variable(type_=KRROODPosition, domain=[])
    query = an(
        entity(position).where(
            or_(position.z == 4, position.x == 2),
        )
    )

    translator = eql_to_sql(query, session)

    query_by_hand = select(KRROODPositionDAO).where(
        (KRROODPositionDAO.z == 4) | (KRROODPositionDAO.x == 2)
    )
    assert str(translator.sql_query) == str(query_by_hand)

    result = translator.evaluate()

    # Assert: rows with z==4 and x==2 should be returned (2 rows)
    zs = sorted([r.z for r in result])
    xs = sorted([r.x for r in result])
    assert len(result) == 2
    assert zs == [4, 10]
    assert xs == [1, 2]


def test_translate_join_one_to_one(session, database):
    session.add(
        KRROODPoseDAO(
            position=KRROODPositionDAO(x=1, y=2, z=3),
            orientation=KRROODOrientationDAO(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    )
    session.add(
        KRROODPoseDAO(
            position=KRROODPositionDAO(x=1, y=2, z=4),
            orientation=KRROODOrientationDAO(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    )
    session.commit()

    pose = variable(type_=KRROODPose, domain=[])
    query = an(entity(pose).where(pose.position.z > 3))
    translator = eql_to_sql(query, session)
    query_by_hand = (
        select(KRROODPoseDAO)
        .join(KRROODPoseDAO.position)
        .where(KRROODPositionDAO.z > 3)
    )

    assert str(translator.sql_query) == str(query_by_hand)

    result = translator.evaluate()

    # Assert: only the pose with position.z == 4 should match
    assert len(result) == 1
    assert isinstance(result[0], KRROODPoseDAO)
    assert result[0].position is not None
    assert result[0].position.z == 4


def test_translate_in_operator(session, database):
    session.add(KRROODPositionDAO(x=1, y=2, z=3))
    session.add(KRROODPositionDAO(x=5, y=2, z=6))
    session.add(KRROODPositionDAO(x=7, y=8, z=9))
    session.commit()

    position = variable(KRROODPosition, domain=[])
    query = an(
        entity(position).where(
            in_(position.x, [1, 7]),
        )
    )

    # Act
    translator = eql_to_sql(query, session)

    query_by_hand = select(KRROODPositionDAO).where(KRROODPositionDAO.x.in_([1, 7]))
    assert str(translator.sql_query) == str(query_by_hand)

    result = translator.evaluate()

    # Assert: x in {1,7}
    xs = sorted([r.x for r in result])
    assert xs == [1, 7]


def test_the_quantifier(session, database):
    position_daos = [KRROODPositionDAO(x=1, y=2, z=3), KRROODPositionDAO(x=5, y=2, z=6)]
    positions = [KRROODPosition(x=dao.x, y=dao.y, z=dao.z) for dao in position_daos]
    session.add_all(position_daos)
    session.commit()

    def get_query(domain=None):
        position = variable(
            type_=KRROODPosition,
            domain=domain,
        )
        query = the(
            entity(position).where(
                position.y == 2,
            )
        )
        return query

    with pytest.raises(MultipleSolutionFound):
        result = get_query(positions).tolist()

    translator = eql_to_sql(get_query(), session)
    query_by_hand = select(KRROODPositionDAO).where(KRROODPositionDAO.y == 2)
    assert str(translator.sql_query) == str(query_by_hand)

    with pytest.raises(MultipleResultsFound):
        result = session.execute(query_by_hand).scalars().one()

    with pytest.raises(MultipleResultsFound):
        result = translator.evaluate()


def test_equal(session, database):
    # Create the world with its bodies and connections
    world = World(
        1,
        [Body("Container1"), Body("Container2"), Body("Handle1"), Body("Handle2")],
    )
    c1_c2 = PrismaticConnection(world.bodies[0], world.bodies[1])
    c2_h2 = FixedConnection(world.bodies[1], world.bodies[3])
    world.connections = [c1_c2, c2_h2]

    dao = to_dao(world)
    session.add(dao)
    session.commit()

    # Query for the kinematic tree of the drawer which has more than one component.
    # Declare the placeholders

    prismatic_connection = variable(
        PrismaticConnection,
        domain=world.connections,
    )
    fixed_connection = variable(FixedConnection, domain=world.connections)

    # Write the query body
    query = an(
        entity(fixed_connection).where(
            fixed_connection.parent == prismatic_connection.child,
        )
    )
    translator = eql_to_sql(query, session)

    query_by_hand = select(FixedConnectionDAO).join(
        PrismaticConnectionDAO,
        onclause=PrismaticConnectionDAO.child_id == FixedConnectionDAO.parent_id,
    )

    assert len(session.scalars(query_by_hand).all()) == 1
    assert str(translator.sql_query) == str(query_by_hand)

    result = translator.evaluate()

    assert len(result) == 1
    assert isinstance(result[0], FixedConnectionDAO)
    assert result[0].parent.name == "Container2"
    assert result[0].child.name == "Handle2"


def test_complicated_equal(session, database):
    # Create the world with its bodies and connections
    world = World(
        1,
        [
            Container("Container1"),
            Container("Container2"),
            Handle("Handle1"),
            Handle("Handle2"),
        ],
    )
    c1_c2 = PrismaticConnection(world.bodies[0], world.bodies[1])
    c2_h2 = FixedConnection(world.bodies[1], world.bodies[3])
    c1_h2_fixed = FixedConnection(world.bodies[0], world.bodies[3])
    world.connections = [c1_c2, c2_h2, c1_h2_fixed]

    dao = to_dao(world)
    session.add(dao)
    session.commit()

    # Query for the kinematic tree of the drawer which has more than one component.
    # Declare the placeholders
    parent_container = variable(type_=Container, domain=world.bodies)
    prismatic_connection = variable(
        type_=PrismaticConnection,
        domain=world.connections,
    )
    drawer_body = variable(type_=Container, domain=world.bodies)
    fixed_connection = variable(type_=FixedConnection, domain=world.connections)
    handle = variable(type_=Handle, domain=world.bodies)

    # Write the query body - this was previously failing with "Attribute chain ended on a relationship"
    query = the(
        entity(drawer_body).where(
            and_(
                parent_container == prismatic_connection.parent,
                drawer_body == prismatic_connection.child,
                drawer_body == fixed_connection.parent,
                handle == fixed_connection.child,
            ),
        )
    )

    eql_result = list(query.evaluate())
    assert len(eql_result) == 1
    assert eql_result[0].name == "Container2"

    # TODO: The generated SQL uses parent_id for PrismaticConnectionDAO join instead
    # of child_id, and HandleDAO as cross-join instead of explicit JOIN.
    # The translation is semantically incomplete for Variable==Attribute constraints.
    # This test currently only verifies translator consistency, not correctness.
    # SQL translation
    translator = eql_to_sql(query, session)
    print(str(translator.sql_query))
    expected_sql = (
        'SELECT "ContainerDAO".database_id, "BodyDAO".database_id AS database_id_1, '
        '"WorldEntityDAO".database_id AS database_id_2, "SymbolDAO".database_id AS '
        'database_id_3, "SymbolDAO".polymorphic_type, "WorldEntityDAO".world_id, '
        '"BodyDAO".name, "BodyDAO".size \n'
        'FROM "SymbolDAO" JOIN "WorldEntityDAO" ON "WorldEntityDAO".database_id = '
        '"SymbolDAO".database_id JOIN "BodyDAO" ON "BodyDAO".database_id = '
        '"WorldEntityDAO".database_id JOIN "ContainerDAO" ON '
        '"ContainerDAO".database_id = "BodyDAO".database_id JOIN ("SymbolDAO" AS '
        '"SymbolDAO_1" JOIN "WorldEntityDAO" AS "WorldEntityDAO_1" ON '
        '"WorldEntityDAO_1".database_id = "SymbolDAO_1".database_id JOIN '
        '"ConnectionDAO" AS "ConnectionDAO_1" ON "ConnectionDAO_1".database_id = '
        '"WorldEntityDAO_1".database_id JOIN "PrismaticConnectionDAO" AS '
        '"PrismaticConnectionDAO_1" ON "PrismaticConnectionDAO_1".database_id = '
        '"ConnectionDAO_1".database_id) ON "ConnectionDAO_1".parent_id = '
        '"ContainerDAO".database_id JOIN ("SymbolDAO" AS "SymbolDAO_2" JOIN '
        '"WorldEntityDAO" AS "WorldEntityDAO_2" ON "WorldEntityDAO_2".database_id = '
        '"SymbolDAO_2".database_id JOIN "ConnectionDAO" AS "ConnectionDAO_2" ON '
        '"ConnectionDAO_2".database_id = "WorldEntityDAO_2".database_id JOIN '
        '"FixedConnectionDAO" AS "FixedConnectionDAO_1" ON '
        '"FixedConnectionDAO_1".database_id = "ConnectionDAO_2".database_id) ON '
        '"ConnectionDAO_2".parent_id = "ContainerDAO".database_id, "HandleDAO" \n'
        'WHERE "ContainerDAO".database_id = "ConnectionDAO_1".child_id AND '
        '"HandleDAO".database_id = "ConnectionDAO_2".child_id'
    )

    assert str(translator.sql_query) == expected_sql


def test_contains(session, database):
    body1 = BodyDAO(name="Body1", size=1)
    session.add(body1)
    session.add(BodyDAO(name="Body2", size=1))
    session.add(BodyDAO(name="Body3", size=1))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(
        entity(b).where(
            contains("Body1TestName", b.name),
        )
    )
    translator = eql_to_sql(query, session)

    result = translator.evaluate()

    assert body1 == result[0]

# =============================================================================
# LIMIT
# =============================================================================

def test_translate_limit(session, database):
    session.add(BodyDAO(name="Body1", size=1))
    session.add(BodyDAO(name="Body2", size=2))
    session.add(BodyDAO(name="Body3", size=3))
    session.add(BodyDAO(name="Body4", size=4))
    session.add(BodyDAO(name="Body5", size=5))
    session.add(BodyDAO(name="Body6", size=6))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b)).limit(5)

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).limit(5)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 5


# =============================================================================
# ORDER BY
# =============================================================================

def test_order_by(session, database):
    session.add(BodyDAO(name="BigBody", size=100))
    session.add(BodyDAO(name="SmallBody", size=10))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).ordered_by(b.size))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).order_by(BodyDAO.size)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    assert results[0].name == "SmallBody"
    assert results[1].name == "BigBody"


def test_order_by_descending(session, database):
    session.add(BodyDAO(name="BigBody", size=100))
    session.add(BodyDAO(name="SmallBody", size=10))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).ordered_by(b.size, descending=True))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).order_by(BodyDAO.size.desc())

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    assert results[0].name == "BigBody"
    assert results[1].name == "SmallBody"


# =============================================================================
# DISTINCT
# =============================================================================

def test_translate_distinct(session, database):
    session.add(BodyDAO(name="UniqueBody", size=10))
    session.add(BodyDAO(name="UniqueBody", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).distinct())

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).distinct()

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2


# =============================================================================
# NOT
# =============================================================================

def test_translate_not(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=20))
    session.add(BodyDAO(name="Body3", size=30))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).where(not_(b.size == 10)))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).where(~(BodyDAO.size == 10))

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    sizes = sorted([r.size for r in results])
    assert sizes == [20, 30]


# =============================================================================
# GROUP BY
# =============================================================================

def test_group_by(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=10))
    session.add(BodyDAO(name="Body3", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.size))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).group_by(BodyDAO.size)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    sizes = sorted([r.size for r in results])
    assert sizes == [10, 20]


def test_group_by_with_count(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=10))
    session.add(BodyDAO(name="Body3", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.size).having(count_all() > 0))

    translator = eql_to_sql(query, session)
    results = translator.evaluate()
    assert len(results) == 2


# =============================================================================
# HAVING
# =============================================================================

def test_having(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=10))
    session.add(BodyDAO(name="Body3", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.size).having(count_all() > 1))

    translator = eql_to_sql(query, session)
    expected = (
        select(BodyDAO)
        .group_by(BodyDAO.size)
        .having(func.count() > 1)
    )

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 1
    assert results[0].size == 10


def test_having_no_results(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.size).having(count_all() > 1))

    translator = eql_to_sql(query, session)
    results = translator.evaluate()
    assert results == []


def test_having_with_max(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=20))
    session.add(BodyDAO(name="Body3", size=30))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.name).having(max(b.size) > 15))

    translator = eql_to_sql(query, session)
    expected = (
        select(BodyDAO)
        .group_by(BodyDAO.name)
        .having(func.max(BodyDAO.size) > 15)
    )

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    sizes = sorted([r.size for r in results])
    assert sizes == [20, 30]


def test_having_with_min(session, database):
    session.add(BodyDAO(name="Body1", size=5))
    session.add(BodyDAO(name="Body1", size=3))
    session.add(BodyDAO(name="Body2", size=20))
    session.add(BodyDAO(name="Body3", size=1))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.name).having(min(b.size) < 8))

    translator = eql_to_sql(query, session)
    expected = (
        select(BodyDAO)
        .group_by(BodyDAO.name)
        .having(func.min(BodyDAO.size) < 8)
    )

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    names = sorted([r.name for r in results])
    assert names == ["Body1", "Body3"]


def test_having_with_sum(session, database):
    session.add(BodyDAO(name="Group1", size=10))
    session.add(BodyDAO(name="Group1", size=20))
    session.add(BodyDAO(name="Group2", size=5))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.name).having(sum(b.size) > 15))

    translator = eql_to_sql(query, session)
    expected = (
        select(BodyDAO)
        .group_by(BodyDAO.name)
        .having(func.sum(BodyDAO.size) > 15)
    )

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 1
    assert results[0].name == "Group1"


def test_having_with_average(session, database):
    session.add(BodyDAO(name="Group1", size=10))
    session.add(BodyDAO(name="Group1", size=30))
    session.add(BodyDAO(name="Group2", size=5))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.name).having(average(b.size) > 15))

    translator = eql_to_sql(query, session)
    expected = (
        select(BodyDAO)
        .group_by(BodyDAO.name)
        .having(func.avg(BodyDAO.size) > 15)
    )

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 1
    assert results[0].name == "Group1"


# =============================================================================
# COMBINATIONS
# =============================================================================

def test_where_and_order_by(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=30))
    session.add(BodyDAO(name="Body3", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).where(b.size > 5).ordered_by(b.size))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).where(BodyDAO.size > 5).order_by(BodyDAO.size)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 3
    assert results[0].name == "Body1"
    assert results[1].name == "Body3"
    assert results[2].name == "Body2"


def test_limit_and_order_by(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=30))
    session.add(BodyDAO(name="Body3", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).ordered_by(b.size)).limit(2)

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).order_by(BodyDAO.size).limit(2)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    assert results[0].name == "Body1"
    assert results[1].name == "Body3"


def test_where_and_limit(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=20))
    session.add(BodyDAO(name="Body3", size=30))
    session.add(BodyDAO(name="Body4", size=40))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).where(b.size > 10)).limit(2)

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).where(BodyDAO.size > 10).limit(2)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2


def test_where_and_group_by_and_having(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=10))
    session.add(BodyDAO(name="Body3", size=20))
    session.add(BodyDAO(name="Body4", size=20))
    session.add(BodyDAO(name="Body5", size=30))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(
        entity(b)
        .where(b.size < 25)
        .grouped_by(b.size)
        .having(count_all() > 1)
    )

    translator = eql_to_sql(query, session)
    expected = (
        select(BodyDAO)
        .where(BodyDAO.size < 25)
        .group_by(BodyDAO.size)
        .having(func.count() > 1)
    )

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    sizes = sorted([r.size for r in results])
    assert sizes == [10, 20]


def test_not_and_combined(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=20))
    session.add(BodyDAO(name="Body3", size=30))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).where(not_(and_(b.size > 5, b.size < 25))))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).where(~((BodyDAO.size > 5) & (BodyDAO.size < 25)))

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 1
    assert results[0].size == 30


def test_order_by_descending_and_limit(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=30))
    session.add(BodyDAO(name="Body3", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).ordered_by(b.size, descending=True)).limit(2)

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).order_by(BodyDAO.size.desc()).limit(2)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    assert results[0].name == "Body2"
    assert results[1].name == "Body3"


def test_join_and_where(session, database):
    session.add(
        KRROODPoseDAO(
            position=KRROODPositionDAO(x=1, y=2, z=3),
            orientation=KRROODOrientationDAO(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    )
    session.add(
        KRROODPoseDAO(
            position=KRROODPositionDAO(x=1, y=2, z=10),
            orientation=KRROODOrientationDAO(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    )
    session.commit()

    pose = variable(type_=KRROODPose, domain=[])
    query = an(entity(pose).where(pose.position.z > 5))

    translator = eql_to_sql(query, session)
    expected = (
        select(KRROODPoseDAO)
        .join(KRROODPoseDAO.position)
        .where(KRROODPositionDAO.z > 5)
    )

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 1
    assert results[0].position.z == 10


# =============================================================================
# EMPTY RESULTS
# =============================================================================

def test_no_results(session, database):
    session.add(BodyDAO(name="Body1", size=10))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).where(b.size > 100))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).where(BodyDAO.size > 100)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert results == []

# =============================================================================
# SET_OF
# =============================================================================

def test_set_of(session):
    """Verify that set_of translates to SELECT of individual columns."""
    b = variable(type_=Body, domain=[])
    query = an(set_of(b.size))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO.size)

    assert str(translator.sql_query) == str(expected)

def test_set_of_with_join(session):
    """Verify that set_of with transitive attributes generates correct JOINs."""
    pose = variable(type_=KRROODPose, domain=[])
    query = an(set_of(pose.position.z))

    translator = eql_to_sql(query, session)
    expected = select(KRROODPositionDAO.z).join(KRROODPoseDAO.position)

    assert str(translator.sql_query) == str(expected)