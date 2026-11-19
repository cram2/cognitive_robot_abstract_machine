"""
Tests of translating entity queries over mapped classes into SQL.
"""

from __future__ import annotations

import pytest
from typing_extensions import TYPE_CHECKING

from sqlalchemy import select, func, case
from sqlalchemy.exc import MultipleResultsFound
from sqlalchemy.orm import aliased
from sqlalchemy.dialects import postgresql

from krrood.entity_query_language.exceptions import MultipleSolutionFound
from ..dataset.example_classes import KRROODPosition, KRROODPose, NestedAction
from ..dataset.semantic_world_like_classes import (
    World,
    Body,
    FixedConnection,
    PrismaticConnection,
    Handle,
    Container,
    MoveAction,
    GraspConfig,
)
from krrood.entity_query_language.factories import (
    a,
    entity,
    variable,
    and_,
    or_,
    contains,
    in_,
    an,
    the,
    count,
    count_all,
    not_,
    max,
    min,
    sum,
    average,
    set_of,
    case_when,
    exists,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.eql_interface import (
    eql_to_sql,
    UnsupportedNestedQueryError,
    UnsupportedQueryTypeError,
)
from krrood.entity_query_language.query.query import UnificationDict

if TYPE_CHECKING:
    from ..dataset.ormatic_interface import NestedActionDAO


def test_translate_simple_greater(session, database):
    """
    A comparison with a constant translates to a ``WHERE`` clause selecting the matching
    rows.
    """
    from ..dataset.ormatic_interface import KRROODPositionDAO

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
    """
    A disjunction translates to an ``OR`` of its conditions, selecting the rows matching
    either.
    """
    from ..dataset.ormatic_interface import KRROODPositionDAO

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
    """
    A condition on an attribute of a related object translates to a join with its table.
    """
    from ..dataset.ormatic_interface import (
        KRROODOrientationDAO,
        KRROODPoseDAO,
        KRROODPositionDAO,
    )

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
    """
    Membership in a list translates to an ``IN`` clause.
    """
    from ..dataset.ormatic_interface import KRROODPositionDAO

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
    """
    A query for *the* entity fails when several rows match, in memory as in the
    database.
    """
    from ..dataset.ormatic_interface import KRROODPositionDAO

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
    """
    Equating attributes of two variables translates to a join on the matching foreign
    keys.
    """
    from ..dataset.ormatic_interface import FixedConnectionDAO, PrismaticConnectionDAO

    # Create the world with its bodies and connections
    world = World(
        id=1,
        bodies=[
            Body(name="Container1"),
            Body(name="Container2"),
            Body(name="Handle1"),
            Body(name="Handle2"),
        ],
    )
    c1_c2 = PrismaticConnection(parent=world.bodies[0], child=world.bodies[1])
    c2_h2 = FixedConnection(parent=world.bodies[1], child=world.bodies[3])
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
        onclause=PrismaticConnectionDAO._child_id == FixedConnectionDAO._parent_id,
    )

    assert len(session.scalars(query_by_hand).all()) == 1
    assert str(translator.sql_query) == str(query_by_hand)

    result = translator.evaluate()

    assert len(result) == 1
    assert isinstance(result[0], FixedConnectionDAO)
    assert result[0].parent.name == "Container2"
    assert result[0].child.name == "Handle2"


def test_complicated_equal(session, database):
    """
    Verify that a multi-variable entity query with equality constraints translates
    correctly and returns the same result as EQL evaluation.
    """
    from ..dataset.ormatic_interface import (
        ContainerDAO,
        FixedConnectionDAO,
        HandleDAO,
        PrismaticConnectionDAO,
    )

    world = World(
        id=1,
        bodies=[
            Container(name="Container1"),
            Container(name="Container2"),
            Handle(name="Handle1"),
            Handle(name="Handle2"),
        ],
    )
    c1_c2 = PrismaticConnection(parent=world.bodies[0], child=world.bodies[1])
    c2_h2 = FixedConnection(parent=world.bodies[1], child=world.bodies[3])
    c1_h2_fixed = FixedConnection(parent=world.bodies[0], child=world.bodies[3])
    world.connections = [c1_c2, c2_h2, c1_h2_fixed]

    dao = to_dao(world)
    session.add(dao)
    session.commit()

    parent_container = variable(type_=Container, domain=world.bodies)
    prismatic_connection = variable(type_=PrismaticConnection, domain=world.connections)
    drawer_body = variable(type_=Container, domain=world.bodies)
    fixed_connection = variable(type_=FixedConnection, domain=world.connections)
    handle = variable(type_=Handle, domain=world.bodies)

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
    translator = eql_to_sql(query, session)

    prismatic_alias = aliased(PrismaticConnectionDAO, flat=True)
    drawer_alias = aliased(ContainerDAO, flat=True)
    fixed_alias = aliased(FixedConnectionDAO, flat=True)
    handle_alias = aliased(HandleDAO, flat=True)

    expected = (
        select(ContainerDAO)
        .join(
            prismatic_alias,
            onclause=prismatic_alias._parent_id == ContainerDAO.database_id,
        )
        .join(
            drawer_alias, onclause=prismatic_alias._child_id == drawer_alias.database_id
        )
        .join(fixed_alias, onclause=fixed_alias._parent_id == drawer_alias.database_id)
        .join(handle_alias, onclause=fixed_alias._child_id == handle_alias.database_id)
        .with_only_columns(drawer_alias)
    )

    eql_result = list(query.evaluate())
    assert len(eql_result) == 1
    assert eql_result[0].name == "Container2"

    assert str(translator.sql_query) == str(expected)

    # SQL result matches EQL result
    [sql_result] = translator.evaluate()
    assert sql_result.name == eql_result[0].name


def test_contains(session, database):
    """
    Containment of an attribute in a constant finds the rows whose value is contained in
    it.
    """
    from ..dataset.ormatic_interface import BodyDAO

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


def test_translate_limit(session, database):
    """
    A limit translates to ``LIMIT`` and caps the number of rows.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=1))
    session.add(BodyDAO(name="Body2", size=2))
    session.add(BodyDAO(name="Body3", size=3))
    session.add(BodyDAO(name="Body4", size=4))
    session.add(BodyDAO(name="Body5", size=5))
    session.add(BodyDAO(name="Body6", size=6))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b)).limit(n=5)

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).limit(5)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 5


def test_order_by(session, database):
    """
    Ordering by an attribute translates to ``ORDER BY``, ascending by default.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="BigBody", size=100))
    session.add(BodyDAO(name="SmallBody", size=10))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).ordered_by(variable=b.size))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).order_by(BodyDAO.size)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    assert results[0].name == "SmallBody"
    assert results[1].name == "BigBody"


def test_order_by_descending(session, database):
    """
    Ordering in descending order translates to ``ORDER BY ...

    DESC``.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="BigBody", size=100))
    session.add(BodyDAO(name="SmallBody", size=10))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).ordered_by(variable=b.size, descending=True))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).order_by(BodyDAO.size.desc())

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    assert results[0].name == "BigBody"
    assert results[1].name == "SmallBody"


def test_translate_distinct(session, database):
    """
    A distinct query translates to ``SELECT DISTINCT``; rows differing in any column
    stay apart.
    """
    from ..dataset.ormatic_interface import BodyDAO

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


def test_translate_not(session, database):
    """
    A negated condition translates to ``NOT``, selecting the rows that fail it.
    """
    from ..dataset.ormatic_interface import BodyDAO

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


def test_group_by(session, database):
    """
    Grouping by an attribute translates to ``GROUP BY``, one row per group.
    """
    from ..dataset.ormatic_interface import BodyDAO

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
    """
    A ``HAVING`` on the count of a group that every group passes keeps every group.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=10))
    session.add(BodyDAO(name="Body3", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.size).having(count_all() > 0))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).group_by(BodyDAO.size).having(func.count() > 0)

    assert str(translator.sql_query) == str(expected)
    results = translator.evaluate()
    assert len(results) == 2


def test_having(session, database):
    """
    A ``HAVING`` on the count of a group keeps only the groups passing it.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=10))
    session.add(BodyDAO(name="Body3", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.size).having(count_all() > 1))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).group_by(BodyDAO.size).having(func.count() > 1)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 1
    assert results[0].size == 10


def test_having_no_results(session, database):
    """
    A ``HAVING`` no group passes yields no rows.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.size).having(count_all() > 1))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).group_by(BodyDAO.size).having(func.count() > 1)
    assert str(translator.sql_query) == str(expected)
    results = translator.evaluate()
    assert results == []


def test_having_with_max(session, database):
    """
    A ``HAVING`` on the maximum of an attribute keeps the groups whose maximum passes
    it.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=20))
    session.add(BodyDAO(name="Body3", size=30))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.name).having(max(b.size) > 15))

    translator = eql_to_sql(query, session)
    expected = (
        select(BodyDAO).group_by(BodyDAO.name).having(func.max(BodyDAO.size) > 15)
    )

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    sizes = sorted([r.size for r in results])
    assert sizes == [20, 30]


def test_having_with_min(session, database):
    """
    A ``HAVING`` on the minimum of an attribute keeps the groups whose minimum passes
    it.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=5))
    session.add(BodyDAO(name="Body1", size=3))
    session.add(BodyDAO(name="Body2", size=20))
    session.add(BodyDAO(name="Body3", size=1))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.name).having(min(b.size) < 8))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).group_by(BodyDAO.name).having(func.min(BodyDAO.size) < 8)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    names = sorted([r.name for r in results])
    assert names == ["Body1", "Body3"]


def test_having_with_sum(session, database):
    """
    A ``HAVING`` on the sum of an attribute keeps the groups whose sum passes it.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Group1", size=10))
    session.add(BodyDAO(name="Group1", size=20))
    session.add(BodyDAO(name="Group2", size=5))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.name).having(sum(b.size) > 15))

    translator = eql_to_sql(query, session)
    expected = (
        select(BodyDAO).group_by(BodyDAO.name).having(func.sum(BodyDAO.size) > 15)
    )

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 1
    assert results[0].name == "Group1"


def test_having_with_average(session, database):
    """
    A ``HAVING`` on the average of an attribute keeps the groups whose average passes
    it.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Group1", size=10))
    session.add(BodyDAO(name="Group1", size=30))
    session.add(BodyDAO(name="Group2", size=5))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).grouped_by(b.name).having(average(b.size) > 15))

    translator = eql_to_sql(query, session)
    expected = (
        select(BodyDAO).group_by(BodyDAO.name).having(func.avg(BodyDAO.size) > 15)
    )

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 1
    assert results[0].name == "Group1"


def test_where_and_order_by(session, database):
    """
    A condition and an ordering combine into ``WHERE`` followed by ``ORDER BY``.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=30))
    session.add(BodyDAO(name="Body3", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).where(b.size > 5).ordered_by(variable=b.size))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).where(BodyDAO.size > 5).order_by(BodyDAO.size)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 3
    assert results[0].name == "Body1"
    assert results[1].name == "Body3"
    assert results[2].name == "Body2"


def test_limit_and_order_by(session, database):
    """
    An ordering and a limit combine into ``ORDER BY`` followed by ``LIMIT``, keeping the
    first rows of the order.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=30))
    session.add(BodyDAO(name="Body3", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).ordered_by(variable=b.size)).limit(n=2)

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).order_by(BodyDAO.size).limit(2)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    assert results[0].name == "Body1"
    assert results[1].name == "Body3"


def test_where_and_limit(session, database):
    """
    A condition and a limit combine into ``WHERE`` followed by ``LIMIT``.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=20))
    session.add(BodyDAO(name="Body3", size=30))
    session.add(BodyDAO(name="Body4", size=40))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).where(b.size > 10)).limit(n=2)

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).where(BodyDAO.size > 10).limit(2)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2


def test_where_and_group_by_and_having(session, database):
    """
    A condition, a grouping and a ``HAVING`` combine in that order, the condition
    filtering rows before they are grouped.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=10))
    session.add(BodyDAO(name="Body3", size=20))
    session.add(BodyDAO(name="Body4", size=20))
    session.add(BodyDAO(name="Body5", size=30))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).where(b.size < 25).grouped_by(b.size).having(count_all() > 1))

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
    """
    A negated conjunction translates to ``NOT`` around an ``AND`` of its conditions.
    """
    from ..dataset.ormatic_interface import BodyDAO

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
    """
    A descending ordering and a limit keep the rows with the largest values.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=30))
    session.add(BodyDAO(name="Body3", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).ordered_by(variable=b.size, descending=True)).limit(n=2)

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).order_by(BodyDAO.size.desc()).limit(2)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert len(results) == 2
    assert results[0].name == "Body2"
    assert results[1].name == "Body3"


def test_join_and_where(session, database):
    """
    A condition on a related object's attribute joins its table and filters on the
    attribute.
    """
    from ..dataset.ormatic_interface import (
        KRROODOrientationDAO,
        KRROODPoseDAO,
        KRROODPositionDAO,
    )

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


def test_no_results(session, database):
    """
    A condition no row passes yields no rows.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=10))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(entity(b).where(b.size > 100))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO).where(BodyDAO.size > 100)

    assert str(translator.sql_query) == str(expected)

    results = translator.evaluate()
    assert results == []


def test_set_of(session):
    """
    Verify that set_of translates to SELECT of individual columns.
    """
    from ..dataset.ormatic_interface import BodyDAO

    b = variable(type_=Body, domain=[])
    query = an(set_of(b.size))

    translator = eql_to_sql(query, session)
    expected = select(BodyDAO.size)

    assert str(translator.sql_query) == str(expected)


def test_set_of_with_join(session):
    """
    Verify that set_of with transitive attributes generates correct JOINs.
    """
    from ..dataset.ormatic_interface import KRROODPoseDAO, KRROODPositionDAO

    pose = variable(type_=KRROODPose, domain=[])
    query = an(set_of(pose.position.z))

    translator = eql_to_sql(query, session)
    expected = select(KRROODPositionDAO.z).join(KRROODPoseDAO.position)

    assert str(translator.sql_query) == str(expected)


def test_set_of_multi_variable(session, database):
    """
    A set of several related variables translates to a select of their tables joined on
    the conditions, with as many rows as the query has solutions in memory.
    """
    from ..dataset.ormatic_interface import (
        ContainerDAO,
        FixedConnectionDAO,
        HandleDAO,
        PrismaticConnectionDAO,
    )

    world = World(id=1, bodies=[Container(name="Container1"), Handle(name="Handle1")])
    fc = FixedConnection(parent=world.bodies[0], child=world.bodies[1])
    pc = PrismaticConnection(parent=world.bodies[0], child=world.bodies[1])
    world.connections = [fc, pc]

    dao = to_dao(world)
    session.add(dao)
    session.commit()

    C = variable(Container, domain=world.bodies)
    H = variable(Handle, domain=world.bodies)
    FC = variable(FixedConnection, domain=world.connections)
    PC = variable(PrismaticConnection, domain=world.connections)

    query = an(
        set_of(C, H, FC, PC).where(
            C == FC.parent,
            H == FC.child,
            C == PC.child,
        )
    )

    translator = eql_to_sql(query, session)

    expected = (
        select(ContainerDAO, HandleDAO, FixedConnectionDAO, PrismaticConnectionDAO)
        .join(
            FixedConnectionDAO,
            onclause=FixedConnectionDAO._parent_id == ContainerDAO.database_id,
        )
        .join(HandleDAO, onclause=FixedConnectionDAO._child_id == HandleDAO.database_id)
        .join(
            PrismaticConnectionDAO,
            onclause=PrismaticConnectionDAO._child_id == ContainerDAO.database_id,
        )
    )
    assert str(translator.sql_query) == str(expected)

    sql_results = translator.evaluate()
    eql_results = list(query.evaluate())
    assert len(sql_results) == len(eql_results)


def test_set_of_move_action_transitive(session):
    """
    Verify that set_of with both direct and transitive attributes generates correct
    JOINs.
    """
    from ..dataset.ormatic_interface import GraspConfigDAO, MoveActionDAO

    from sqlalchemy.orm import aliased

    move = variable(type_=MoveAction, domain=[])
    query = an(
        set_of(
            move.robot_x,
            move.robot_y,
            move.hip_rotation,
            move.grasp_config.rotate_gripper,
            move.grasp_config.approach_direction,
            move.grasp_config.manipulation_offset,
        )
    )

    translator = eql_to_sql(query, session)

    grasp_alias = aliased(GraspConfigDAO, flat=True)

    expected = select(
        MoveActionDAO.robot_x,
        MoveActionDAO.robot_y,
        MoveActionDAO.hip_rotation,
        grasp_alias.rotate_gripper,
        grasp_alias.approach_direction,
        grasp_alias.manipulation_offset,
    ).join(
        grasp_alias, onclause=grasp_alias.database_id == MoveActionDAO._grasp_config_id
    )

    assert str(translator.sql_query) == str(expected)


def test_set_of_with_where(session):
    """
    Verify set_of with transitive attributes and WHERE condition.

    Simulates: SELECT x, y, z FROM PoseDAO JOIN PositionDAO WHERE z < 0.9
    """
    from ..dataset.ormatic_interface import KRROODPoseDAO, KRROODPositionDAO

    pose = variable(type_=KRROODPose, domain=[])
    query = an(
        set_of(
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ).where(pose.position.z < 0.9)
    )

    translator = eql_to_sql(query, session)

    position_alias = aliased(KRROODPositionDAO, flat=True)

    expected = (
        select(KRROODPoseDAO)
        .join(
            position_alias,
            onclause=position_alias.database_id == KRROODPoseDAO._position_id,
        )
        .with_only_columns(
            position_alias.x,
            position_alias.y,
            position_alias.z,
        )
        .where(position_alias.z < 0.9)
    )

    assert str(translator.sql_query) == str(expected)


def test_set_of_same_table_twice(session):
    """
    Verify that two variables of the same type produce separate JOINs with aliases.

    Simulates: JOIN NavigateActionDAO np ON ... JOIN NavigateActionDAO np2 ON ...
    This uses two KRROODPose variables to test the same pattern.
    """
    from ..dataset.ormatic_interface import ContainerDAO

    world = World(
        id=1,
        bodies=[
            Container(name="Container1"),
            Container(name="Container2"),
        ],
    )
    fc1 = FixedConnection(parent=world.bodies[0], child=world.bodies[1])
    fc2 = FixedConnection(parent=world.bodies[1], child=world.bodies[0])
    world.connections = [fc1, fc2]

    fc_pick = variable(FixedConnection, domain=world.connections)
    fc_place = variable(FixedConnection, domain=world.connections)
    C = variable(Container, domain=world.bodies)

    query = an(
        set_of(fc_pick, fc_place, C).where(
            C == fc_pick.parent,
            C == fc_place.child,
        )
    )

    translator = eql_to_sql(query, session)
    sql = str(translator.sql_query)
    assert "FixedConnectionDAO" in sql
    assert sql.count("JOIN") >= 2
    assert "_parent_id" in sql
    assert "_child_id" in sql
    assert str(ContainerDAO.__tablename__) in sql


def test_plan_like_query(session):
    """
    Simulate the big plan query pattern:
    SELECT pick.arm, place_pos.x, place_pos.y, nav_pos.x, nav_pos.y
    FROM ... JOIN ... JOIN ...
    WHERE nav_pos.z < 0.9

    Uses MoveAction/GraspConfig to simulate PickUpAction/NavigateAction pattern.
    """
    from ..dataset.ormatic_interface import (
        FixedConnectionDAO,
        GraspConfigDAO,
        MoveActionDAO,
    )

    world = World(
        id=1,
        bodies=[
            Container(name="StartPos"),
            Container(name="EndPos"),
        ],
    )
    fc1 = FixedConnection(parent=world.bodies[0], child=world.bodies[1])
    fc2 = FixedConnection(parent=world.bodies[1], child=world.bodies[0])
    world.connections = [fc1, fc2]

    move_pick = variable(type_=MoveAction, domain=[])
    move_place = variable(type_=MoveAction, domain=[])
    fc_connection = variable(FixedConnection, domain=world.connections)

    query = an(
        set_of(
            move_pick.robot_x,
            move_pick.robot_y,
            move_place.robot_x,
            move_place.robot_y,
            move_pick.grasp_config.rotate_gripper,
        ).where(
            fc_connection.parent == move_pick.grasp_config,
            move_place.robot_x > 0.0,
        )
    )

    translator = eql_to_sql(query, session)
    expected = (
        select(
            MoveActionDAO.robot_x,
            MoveActionDAO.robot_y,
            MoveActionDAO.robot_x,
            MoveActionDAO.robot_y,
            GraspConfigDAO.rotate_gripper,
        )
        .join(
            GraspConfigDAO,
            onclause=GraspConfigDAO.database_id == MoveActionDAO._grasp_config_id,
        )
        .join(
            FixedConnectionDAO,
            onclause=FixedConnectionDAO._parent_id == MoveActionDAO._grasp_config_id,
        )
        .where(MoveActionDAO.robot_x > 0.0)
    )
    assert str(translator.sql_query) == str(expected)


def test_set_of_multi_variable_evaluate(session, database):
    """
    Verify that evaluate() for set_of with multiple variables returns dicts.
    """
    world = World(
        id=1,
        bodies=[
            Container(name="Container1"),
            Handle(name="Handle1"),
        ],
    )
    fc = FixedConnection(parent=world.bodies[0], child=world.bodies[1])
    world.connections = [fc]

    dao = to_dao(world)
    session.add(dao)
    session.commit()

    C = variable(Container, domain=world.bodies)
    H = variable(Handle, domain=world.bodies)
    FC = variable(FixedConnection, domain=world.connections)

    query = an(
        set_of(C, H, FC).where(
            C == FC.parent,
            H == FC.child,
        )
    )

    translator = eql_to_sql(query, session)
    results = translator.evaluate()

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, UnificationDict)
    assert result[C].name == "Container1"
    assert result[H].name == "Handle1"


def test_set_of_attribute_evaluate(session, database):
    """
    Verify that evaluate() for set_of with Attribute variables returns dicts.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(set_of(b.name, b.size))

    translator = eql_to_sql(query, session)
    results = translator.evaluate()

    assert len(results) == 2
    assert isinstance(results[0], UnificationDict)
    keys = list(results[0].keys())
    assert len(keys) == 2
    names = sorted([r[keys[0]] for r in results])
    assert names == ["Body1", "Body2"]


def test_set_of_transitive_evaluate(session, database):
    """
    Verify evaluate() for set_of with transitive attributes returns dicts.
    """
    from ..dataset.ormatic_interface import (
        KRROODOrientationDAO,
        KRROODPoseDAO,
        KRROODPositionDAO,
    )

    session.add(
        KRROODPoseDAO(
            position=KRROODPositionDAO(x=1.0, y=2.0, z=3.0),
            orientation=KRROODOrientationDAO(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    )
    session.commit()

    pose = variable(type_=KRROODPose, domain=[])
    query = an(
        set_of(
            pose.position.x,
            pose.position.y,
            pose.position.z,
        )
    )

    translator = eql_to_sql(query, session)
    results = translator.evaluate()

    assert len(results) == 1
    assert isinstance(results[0], UnificationDict)
    keys = list(results[0].keys())
    assert len(keys) == 3
    values = list(results[0].values())
    assert 1.0 in values
    assert 2.0 in values
    assert 3.0 in values


def test_big_query_select_part(session):
    """
    Simulate the SELECT part of the big plan query using existing test classes.

    Simulates: SELECT robot_x, robot_y, rotate_gripper FROM ...
               WHERE rotate_gripper < 0.9 ORDER BY robot_x
    """
    from ..dataset.ormatic_interface import GraspConfigDAO, MoveActionDAO

    move_pick = variable(type_=MoveAction, domain=[])
    move_place = variable(type_=MoveAction, domain=[])

    query = an(
        set_of(
            move_pick.robot_x,
            move_pick.robot_y,
            move_place.robot_x,
            move_place.robot_y,
            move_pick.grasp_config.rotate_gripper,
            move_pick.grasp_config.approach_direction,
        )
        .where(move_pick.grasp_config.rotate_gripper < 0.9)
        .ordered_by(variable=move_pick.robot_x)
    )

    translator = eql_to_sql(query, session)

    grasp_alias = aliased(GraspConfigDAO, flat=True)

    expected = (
        select(
            MoveActionDAO.robot_x,
            MoveActionDAO.robot_y,
            MoveActionDAO.robot_x,
            MoveActionDAO.robot_y,
            grasp_alias.rotate_gripper,
            grasp_alias.approach_direction,
        )
        .join(
            grasp_alias,
            onclause=grasp_alias.database_id == MoveActionDAO._grasp_config_id,
        )
        .where(grasp_alias.rotate_gripper < 0.9)
        .order_by(MoveActionDAO.robot_x)
    )

    assert str(translator.sql_query) == str(expected)


def test_cte_from_eql(session, database):
    """
    Verify that an EQL query can be translated to a CTE and used in an outer query.

    Simulates the WITH clause pattern:
    WITH large_bodies AS (SELECT * FROM BodyDAO WHERE size > 5)
    SELECT * FROM ContainerDAO JOIN large_bodies ON large_bodies.database_id = ContainerDAO.database_id
    """
    from ..dataset.ormatic_interface import BodyDAO, ContainerDAO

    session.add(BodyDAO(name="SmallBody", size=1))
    session.add(ContainerDAO(name="LargeContainer", size=10))
    session.commit()

    b = variable(type_=Body, domain=[])
    inner_query = an(entity(b).where(b.size > 5))
    large_bodies = eql_to_sql(inner_query, session).sql_query.cte("large_bodies")

    c = variable(type_=Container, domain=[])
    outer_translator = eql_to_sql(an(entity(c)), session)
    outer_translator.sql_query = outer_translator.sql_query.join(
        large_bodies, large_bodies.c.database_id == ContainerDAO.database_id
    )

    # Build expected using SQLAlchemy objects — same starting point as the translator
    inner_expected = select(BodyDAO).where(BodyDAO.size > 5).cte("large_bodies")

    expected = select(ContainerDAO).join(
        inner_expected,
        onclause=inner_expected.c.database_id == ContainerDAO.database_id,
    )

    assert str(outer_translator.sql_query) == str(expected)

    # Verify that the CTE filters correctly — only LargeContainer has size > 5
    results = session.execute(outer_translator.sql_query).all()
    assert len(results) == 1


def test_case_when_with_min(session):
    """
    Verify that min(case_when(...)) translates to SQL MIN(CASE WHEN ...

    THEN ... END). Simulates: MIN(CASE WHEN d.polymorphic_type='PickUpActionDAO' THEN id
    END) from the big plan query.
    """
    from ..dataset.ormatic_interface import MoveActionDAO, SymbolDAO

    action = variable(MoveAction, domain=None)

    query = an(
        set_of(
            min(
                case_when(
                    action.polymorphic_type == "PickUpActionDAO", action.database_id
                )
            )
        )
    )

    translator = eql_to_sql(query, session)

    expected = select(MoveActionDAO).with_only_columns(
        func.min(
            case(
                (
                    SymbolDAO.polymorphic_type == "PickUpActionDAO",
                    MoveActionDAO.database_id,
                )
            )
        )
    )

    assert str(translator.sql_query) == str(expected)


def test_case_when_direct_in_set_of(session, database):
    """
    Verify case_when directly in set_of without aggregator using rich setup.
    """
    action_obj = MoveAction(robot_x=15.5, robot_y=0.0, hip_rotation=0.0)

    dao = to_dao(action_obj)
    session.add(dao)
    session.commit()

    action = variable(MoveAction, domain=None)
    query = an(
        set_of(case_when(action.polymorphic_type == "PickUpActionDAO", action.robot_x))
    )
    translator = eql_to_sql(query, session)

    expected_sql = (
        'SELECT CASE WHEN ("SymbolDAO".polymorphic_type = :polymorphic_type_1) '
        'THEN "MoveActionDAO".robot_x END AS anon_1 \n'
        'FROM "SymbolDAO" JOIN "WorldEntityDAO" ON "WorldEntityDAO".database_id = '
        '"SymbolDAO".database_id JOIN "MoveActionDAO" ON '
        '"MoveActionDAO".database_id = "WorldEntityDAO".database_id'
    )
    assert str(translator.sql_query) == expected_sql

    sql_results = translator.evaluate()

    flat_results = [list(row.values())[0] for row in sql_results]

    assert flat_results == [None]


def test_case_when_with_max(session):
    """
    Verify max(case_when(...)) translates correctly.
    """
    from ..dataset.ormatic_interface import MoveActionDAO, SymbolDAO

    action = variable(MoveAction, domain=None)
    query = an(
        set_of(
            max(
                case_when(
                    action.polymorphic_type == "PlaceActionDAO", action.database_id
                )
            )
        )
    )
    translator = eql_to_sql(query, session)

    expected = select(MoveActionDAO).with_only_columns(
        func.max(
            case(
                (
                    SymbolDAO.polymorphic_type == "PlaceActionDAO",
                    MoveActionDAO.database_id,
                )
            )
        )
    )

    assert str(translator.sql_query) == str(expected)


def test_entity_from_multi_hop_attribute(session, database):
    """
    Prove bug: entity(a.pose.position) where the chain is two hops deep (NestedAction →
    pose → KRROODPose → position → KRROODPosition) raises AttributeResolutionError
    because _translate_entity_from_attribute looks up the outermost attribute name
    ('position') directly on the leaf DAO (NestedActionDAO) instead of walking the full
    chain.

    After the fix the translator must join through KRROODPoseDAO and return
    KRROODPositionDAO instances for all linked actions.
    """
    from ..dataset.ormatic_interface import (
        BodyDAO,
        KRROODOrientationDAO,
        KRROODPoseDAO,
        KRROODPositionDAO,
        NestedActionDAO,
    )

    orientation = KRROODOrientationDAO(x=0.0, y=0.0, z=0.0, w=1.0)
    position_low = KRROODPositionDAO(x=0.0, y=0.0, z=3.0)
    position_high = KRROODPositionDAO(x=0.0, y=0.0, z=30.0)
    pose_low = KRROODPoseDAO(position=position_low, orientation=orientation)
    pose_high = KRROODPoseDAO(position=position_high, orientation=orientation)
    body = BodyDAO(name="TestBody", size=1)
    session.add_all(
        [
            NestedActionDAO(obj=body, pose=pose_low),
            NestedActionDAO(obj=body, pose=pose_high),
        ]
    )
    session.commit()

    a = variable(NestedAction, domain=[])
    query = an(entity(a.pose.position))

    translator = eql_to_sql(query, session)
    results = translator.evaluate()

    assert len(results) == 2
    assert all(isinstance(r, KRROODPositionDAO) for r in results)
    z_values = sorted(r.z for r in results)
    assert z_values == pytest.approx([3.0, 30.0])


def test_entity_with_relationship_selected_variable(session, database):
    """
    Prove bug: entity(m.grasp_config).where(m.robot_x > 0.5) produces a cross-join
    instead of an INNER JOIN, returning too many results.

    After the fix, GraspConfigDAO must be joined with MoveActionDAO via the FK so only
    the grasp_config linked to a high-robot_x move is returned.
    """
    from ..dataset.ormatic_interface import GraspConfigDAO, MoveActionDAO

    grasp_matching = GraspConfigDAO(
        rotate_gripper=0.3, approach_direction=0.0, manipulation_offset=0.0
    )
    grasp_not_matching = GraspConfigDAO(
        rotate_gripper=0.9, approach_direction=0.0, manipulation_offset=0.0
    )
    session.add(grasp_matching)
    session.add(grasp_not_matching)

    move_high = MoveActionDAO(robot_x=1.0, robot_y=0.0, hip_rotation=0.0)
    move_high.grasp_config = grasp_matching
    session.add(move_high)

    move_low = MoveActionDAO(robot_x=0.1, robot_y=0.0, hip_rotation=0.0)
    move_low.grasp_config = grasp_not_matching
    session.add(move_low)

    session.commit()

    m = variable(MoveAction, domain=[])
    query = an(entity(m.grasp_config).where(m.robot_x > 0.5))

    translator = eql_to_sql(query, session)
    results = translator.evaluate()

    assert len(results) == 1
    assert isinstance(results[0], GraspConfigDAO)
    assert results[0].rotate_gripper == pytest.approx(0.3)


def test_count_all_derives_dao_from_where_clause(session, database):
    """
    Prove bug: set_of(count_all()).where(m.robot_x > 0.5) raises NoDAOFoundError because
    CountAll carries no DAO information.

    After the fix, the translator must fall back to the WHERE clause to find the base
    DAO (MoveActionDAO via m.robot_x) and emit SELECT count(*) FROM MoveActionDAO.
    """
    from ..dataset.ormatic_interface import MoveActionDAO

    session.add(MoveActionDAO(robot_x=1.0, robot_y=0.0, hip_rotation=0.0))
    session.add(MoveActionDAO(robot_x=2.0, robot_y=0.0, hip_rotation=0.0))
    session.add(MoveActionDAO(robot_x=0.1, robot_y=0.0, hip_rotation=0.0))
    session.commit()

    m = variable(MoveAction, domain=[])
    c = count_all()
    query = an(set_of(c).where(m.robot_x > 0.5))

    translator = eql_to_sql(query, session)
    results = translator.evaluate()

    assert len(results) == 1
    assert results[0][c] == 2


def test_order_by_aggregate(session, database):
    """
    Prove bug: set_of(b.size, c).grouped_by(b.size).ordered_by(variable=c,
    descending=True) raises an error because _apply_clauses calls translate_attribute()
    on a Count aggregator instead of _translate_comparator_operand().

    After the fix, the translator must emit: SELECT size, COUNT(*) FROM BodyDAO GROUP BY
    size ORDER BY COUNT(*) DESC and return rows ordered with the most-frequent size
    first.
    """
    from ..dataset.ormatic_interface import BodyDAO

    session.add_all(
        [
            BodyDAO(name="Body1", size=10),
            BodyDAO(name="Body2", size=10),
            BodyDAO(name="Body3", size=10),
            BodyDAO(name="Body4", size=20),
            BodyDAO(name="Body5", size=20),
        ]
    )
    session.commit()

    b = variable(type_=Body, domain=[])
    c = count(b)
    query = an(
        set_of(b.size, c).grouped_by(b.size).ordered_by(variable=c, descending=True)
    )

    translator = eql_to_sql(query, session)
    results = translator.evaluate()

    assert len(results) == 2
    sizes = [list(row.values())[0] for row in results]
    assert sizes[0] == 10  # appears 3 times → largest count, comes first
    assert sizes[1] == 20  # appears 2 times


def test_exists_in_where_clause(session, database):
    """
    Prove bug: exists(g, m.grasp_config == g) in a WHERE clause raises
    UnsupportedQueryTypeError because Exists is not handled in translate_query.

    After the fix, the translator must emit a correlated EXISTS subquery:

    SELECT MoveActionDAO.*
    FROM MoveActionDAO
    WHERE EXISTS (
        SELECT 1 FROM GraspConfigDAO
        WHERE MoveActionDAO._grasp_config_id = GraspConfigDAO.database_id
    )

    so only MoveActions that have an associated GraspConfig are returned.
    """
    from ..dataset.ormatic_interface import GraspConfigDAO, MoveActionDAO

    grasp_a = GraspConfigDAO(
        rotate_gripper=0.1, approach_direction=0.0, manipulation_offset=0.0
    )
    grasp_b = GraspConfigDAO(
        rotate_gripper=0.5, approach_direction=0.0, manipulation_offset=0.0
    )
    session.add_all([grasp_a, grasp_b])

    move_with_grasp_a = MoveActionDAO(robot_x=1.0, robot_y=0.0, hip_rotation=0.0)
    move_with_grasp_a.grasp_config = grasp_a
    move_with_grasp_b = MoveActionDAO(robot_x=2.0, robot_y=0.0, hip_rotation=0.0)
    move_with_grasp_b.grasp_config = grasp_b
    move_without_grasp = MoveActionDAO(robot_x=3.0, robot_y=0.0, hip_rotation=0.0)

    session.add_all([move_with_grasp_a, move_with_grasp_b, move_without_grasp])
    session.commit()

    m = variable(MoveAction, domain=[])
    g = variable(GraspConfig, domain=[])
    query = an(entity(m).where(exists(g, m.grasp_config == g)))

    translator = eql_to_sql(query, session)
    results = translator.evaluate()

    assert len(results) == 2
    result_robot_xs = {r.robot_x for r in results}
    assert result_robot_xs == {1.0, 2.0}


# %% match patterns


@pytest.fixture
def nested_actions_at_two_heights(session, database) -> dict[float, NestedActionDAO]:
    """
    Store two nested actions on one body whose poses differ only in the height of their
    position.

    :return: The stored actions, keyed by the height of their position.
    """
    from ..dataset.ormatic_interface import (
        BodyDAO,
        KRROODOrientationDAO,
        KRROODPoseDAO,
        KRROODPositionDAO,
        NestedActionDAO,
    )

    body = BodyDAO(name="TestBody", size=1)
    actions = {
        height: NestedActionDAO(
            obj=body,
            pose=KRROODPoseDAO(
                position=KRROODPositionDAO(x=0.0, y=0.0, z=height),
                orientation=KRROODOrientationDAO(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )
        for height in (3.0, 30.0)
    }
    session.add_all(actions.values())
    session.commit()
    return actions


def test_match_selects_only_objects_with_stated_values(
    session, nested_actions_at_two_heights
):
    """
    A match selects only the objects whose nested attributes have the values it states.
    """
    query = a(NestedAction)(pose=a(KRROODPose)(position=a(KRROODPosition)(z=30.0)))

    results = eql_to_sql(query, session).evaluate()

    assert results == [nested_actions_at_two_heights[30.0]]


def test_match_selects_nothing_for_unstored_values(
    session, nested_actions_at_two_heights
):
    """
    A match whose stated values no stored object has selects nothing.
    """
    query = a(NestedAction)(
        obj=a(Body)(name="TestBody"),
        pose=a(KRROODPose)(position=a(KRROODPosition)(z=4.0)),
    )

    assert eql_to_sql(query, session).evaluate() == []


def test_match_keeps_its_quantifier(session, nested_actions_at_two_heights):
    """
    A match built with :func:`the` evaluates to its single object.
    """
    query = the(NestedAction)(pose=a(KRROODPose)(position=a(KRROODPosition)(z=3.0)))

    results = eql_to_sql(query, session).evaluate()

    assert results == [nested_actions_at_two_heights[3.0]]


def test_match_keeps_its_limit(session, nested_actions_at_two_heights):
    """
    A match keeps the limit stated on it.
    """
    query = a(NestedAction)(obj=a(Body)(name="TestBody")).limit(n=1)

    assert len(eql_to_sql(query, session).evaluate()) == 1


def test_match_wrapped_in_entity_keeps_its_conditions(
    session, nested_actions_at_two_heights
):
    """
    A match wrapped in :func:`entity` selects what the match on its own selects.
    """
    query = an(
        entity(a(NestedAction)(pose=a(KRROODPose)(position=a(KRROODPosition)(z=30.0))))
    )

    results = eql_to_sql(query, session).evaluate()

    assert results == [nested_actions_at_two_heights[30.0]]


def test_match_with_limit_wrapped_in_entity_raises(
    session, nested_actions_at_two_heights
):
    """
    A match wrapped in :func:`entity` whose own modifiers the translation cannot keep
    raises instead of selecting more than the match does.
    """
    query = an(entity(a(KRROODPosition)(x=0.0).limit(n=1)))

    with pytest.raises(UnsupportedNestedQueryError):
        eql_to_sql(query, session)


def test_expression_that_is_not_a_query_raises(session):
    """
    An expression that is not a query, such as a variable on its own, has no
    translation.
    """
    with pytest.raises(UnsupportedQueryTypeError):
        eql_to_sql(variable(NestedAction, domain=[]), session)
