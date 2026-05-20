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
    MoveAction,
    GraspConfig,
)
from ..dataset.ormatic_interface import (
    KRROODPositionDAO,
    KRROODPoseDAO,
    KRROODOrientationDAO,
    FixedConnectionDAO,
    PrismaticConnectionDAO,
    BodyDAO,
    MoveActionDAO,
    GraspConfigDAO,
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
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.orm.ormatic_interface import PickUpActionDAO, GraspDescriptionDAO


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

    translator = eql_to_sql(query, session)
    expected_sql = str(translator.sql_query)
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
    expected = (
        select(BodyDAO)
        .group_by(BodyDAO.size)
        .having(func.count() > 0)
    )

    assert str(translator.sql_query) == str(expected)
    results = translator.evaluate()
    assert len(results) == 2


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
    expected = (
        select(BodyDAO)
        .group_by(BodyDAO.size)
        .having(func.count() > 1)
    )
    assert str(translator.sql_query) == str(expected)
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

def test_set_of_multi_variable(session):
    """Verify that set_of with multiple variables generates correct JOINs."""
    world = World(1, [
        Container("Container1"),
        Handle("Handle1"),
    ])
    fc = FixedConnection(world.bodies[0], world.bodies[1])
    pc = PrismaticConnection(world.bodies[0], world.bodies[1])
    world.connections = [fc, pc]

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
    expected_sql = str(translator.sql_query)

    assert str(translator.sql_query) == expected_sql
    assert ", \"HandleDAO\"" not in str(translator.sql_query)
    assert "JOIN" in str(translator.sql_query)



def test_set_of_transitive_attributes(session):
    """Verify that set_of with transitive attributes generates a JOIN to GraspDescriptionDAO."""
    pu = variable(type_=PickUpAction, domain=[])
    query = an(set_of(
        pu.arm,
        pu.grasp_description.rotate_gripper,
        pu.grasp_description.approach_direction,
        pu.grasp_description.manipulation_offset,
    ))

    translator = eql_to_sql(query, session)
    expected_sql = (
        'SELECT "PickUpActionDAO".arm, "GraspDescriptionDAO_1".rotate_gripper, '
        '"GraspDescriptionDAO_1".approach_direction, '
        '"GraspDescriptionDAO_1".manipulation_offset \n'
        'FROM "DesignatorDAO" JOIN "ActionDescriptionDAO" ON '
        '"ActionDescriptionDAO".database_id = "DesignatorDAO".database_id '
        'JOIN "PickUpActionDAO" ON "PickUpActionDAO".database_id = '
        '"ActionDescriptionDAO".database_id JOIN "GraspDescriptionDAO" AS '
        '"GraspDescriptionDAO_1" ON "GraspDescriptionDAO_1".database_id = '
        '"PickUpActionDAO".grasp_description_id'
    )
    assert str(translator.sql_query) == expected_sql

def test_set_of_move_action_transitive(session):
    """
    Verify that set_of with both direct and transitive attributes generates correct JOINs.
    This simulates the pattern of MoveToReachDAO.robot_x and
    MoveToReachDAO.grasp_description.rotate_gripper from pycram.
    """
    move = variable(type_=MoveAction, domain=[])
    query = an(set_of(
        move.robot_x,
        move.robot_y,
        move.hip_rotation,
        move.grasp_config.rotate_gripper,
        move.grasp_config.approach_direction,
        move.grasp_config.manipulation_offset,
    ))

    translator = eql_to_sql(query, session)
    expected_sql = (
        'SELECT "MoveActionDAO".robot_x, "MoveActionDAO".robot_y, '
        '"MoveActionDAO".hip_rotation, "GraspConfigDAO_1".rotate_gripper, '
        '"GraspConfigDAO_1".approach_direction, '
        '"GraspConfigDAO_1".manipulation_offset \n'
        'FROM "SymbolDAO" JOIN "WorldEntityDAO" ON "WorldEntityDAO".database_id = '
        '"SymbolDAO".database_id JOIN "MoveActionDAO" ON '
        '"MoveActionDAO".database_id = "WorldEntityDAO".database_id JOIN '
        '("SymbolDAO" AS "SymbolDAO_1" JOIN "WorldEntityDAO" AS '
        '"WorldEntityDAO_1" ON "WorldEntityDAO_1".database_id = '
        '"SymbolDAO_1".database_id JOIN "GraspConfigDAO" AS "GraspConfigDAO_1" '
        'ON "GraspConfigDAO_1".database_id = "WorldEntityDAO_1".database_id) ON '
        '"GraspConfigDAO_1".database_id = "MoveActionDAO".grasp_config_id'
    )
    assert str(translator.sql_query) == expected_sql


def test_set_of_with_where(session):
    """
    Verify set_of with transitive attributes and WHERE condition.
    Simulates: SELECT x, y, z FROM PoseDAO JOIN PositionDAO WHERE z < 0.9
    """
    pose = variable(type_=KRROODPose, domain=[])
    query = an(
        set_of(
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ).where(pose.position.z < 0.9)
    )

    translator = eql_to_sql(query, session)
    expected_sql = (
        'SELECT "KRROODPositionDAO_1".x, "KRROODPositionDAO_1".y, '
        '"KRROODPositionDAO_1".z \n'
        'FROM "SymbolDAO" JOIN "KRROODPoseDAO" ON "KRROODPoseDAO".database_id = '
        '"SymbolDAO".database_id JOIN ("SymbolDAO" AS "SymbolDAO_1" JOIN '
        '"KRROODPositionDAO" AS "KRROODPositionDAO_1" ON '
        '"KRROODPositionDAO_1".database_id = "SymbolDAO_1".database_id) ON '
        '"KRROODPositionDAO_1".database_id = "KRROODPoseDAO".position_id \n'
        'WHERE "KRROODPositionDAO_1".z < :z_1'
    )
    assert str(translator.sql_query) == expected_sql


def test_set_of_same_table_twice(session):
    """
    Verify that two variables of the same type produce separate JOINs with aliases.
    Simulates: JOIN NavigateActionDAO np ON ... JOIN NavigateActionDAO np2 ON ...
    This uses two KRROODPose variables to test the same pattern.
    """
    world = World(1, [
        Container("Container1"),
        Container("Container2"),
    ])
    fc1 = FixedConnection(world.bodies[0], world.bodies[1])
    fc2 = FixedConnection(world.bodies[1], world.bodies[0])
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
    expected_sql = (
        'SELECT "FixedConnectionDAO_1".database_id, "ConnectionDAO_1".database_id AS database_id_1, '
        '"WorldEntityDAO_1".database_id AS database_id_2, "SymbolDAO_1".database_id AS database_id_3, '
        '"SymbolDAO_1".polymorphic_type, "WorldEntityDAO_1".world_id, '
        '"ConnectionDAO_1".parent_id, "ConnectionDAO_1".child_id, '
        '"ContainerDAO_1".database_id AS database_id_4, '
        '"BodyDAO_1".database_id AS database_id_5, '
        '"WorldEntityDAO_2".database_id AS database_id_6, '
        '"SymbolDAO_2".database_id AS database_id_7, '
        '"SymbolDAO_2".polymorphic_type AS polymorphic_type_1, '
        '"WorldEntityDAO_2".world_id AS world_id_1, '
        '"BodyDAO_1".name, "BodyDAO_1".size \n'
        'FROM "SymbolDAO" JOIN "WorldEntityDAO" ON "WorldEntityDAO".database_id = '
        '"SymbolDAO".database_id JOIN "BodyDAO" ON "BodyDAO".database_id = '
        '"WorldEntityDAO".database_id JOIN "ContainerDAO" ON '
        '"ContainerDAO".database_id = "BodyDAO".database_id JOIN ("SymbolDAO" AS '
        '"SymbolDAO_1" JOIN "WorldEntityDAO" AS "WorldEntityDAO_1" ON '
        '"WorldEntityDAO_1".database_id = "SymbolDAO_1".database_id JOIN '
        '"ConnectionDAO" AS "ConnectionDAO_1" ON "ConnectionDAO_1".database_id = '
        '"WorldEntityDAO_1".database_id JOIN "FixedConnectionDAO" AS '
        '"FixedConnectionDAO_1" ON "FixedConnectionDAO_1".database_id = '
        '"ConnectionDAO_1".database_id) ON "ConnectionDAO_1".parent_id = '
        '"ContainerDAO".database_id JOIN ("SymbolDAO" AS "SymbolDAO_2" JOIN '
        '"WorldEntityDAO" AS "WorldEntityDAO_2" ON "WorldEntityDAO_2".database_id = '
        '"SymbolDAO_2".database_id JOIN "BodyDAO" AS "BodyDAO_1" ON '
        '"BodyDAO_1".database_id = "WorldEntityDAO_2".database_id JOIN "ContainerDAO" '
        'AS "ContainerDAO_1" ON "ContainerDAO_1".database_id = '
        '"BodyDAO_1".database_id) ON "ConnectionDAO_1".child_id = '
        '"ContainerDAO_1".database_id'
    )
    assert str(translator.sql_query) == expected_sql



def test_plan_like_query(session):
    """
    Simulate the big plan query pattern:
    SELECT pick.arm, place_pos.x, place_pos.y, nav_pos.x, nav_pos.y
    FROM ... JOIN ... JOIN ...
    WHERE nav_pos.z < 0.9

    Uses MoveAction/GraspConfig to simulate PickUpAction/NavigateAction pattern.
    """
    world = World(1, [
        Container("StartPos"),
        Container("EndPos"),
    ])
    fc1 = FixedConnection(world.bodies[0], world.bodies[1])
    fc2 = FixedConnection(world.bodies[1], world.bodies[0])
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
    expected_sql = (
        'SELECT "MoveActionDAO".robot_x, "MoveActionDAO".robot_y, '
        '"MoveActionDAO".robot_x AS robot_x__1, '
        '"MoveActionDAO".robot_y AS robot_y__1, '
        '"GraspConfigDAO_1".rotate_gripper \n'
        'FROM "SymbolDAO" JOIN "WorldEntityDAO" ON "WorldEntityDAO".database_id = '
        '"SymbolDAO".database_id JOIN "MoveActionDAO" ON '
        '"MoveActionDAO".database_id = "WorldEntityDAO".database_id JOIN '
        '("SymbolDAO" AS "SymbolDAO_1" JOIN "WorldEntityDAO" AS '
        '"WorldEntityDAO_1" ON "WorldEntityDAO_1".database_id = '
        '"SymbolDAO_1".database_id JOIN "GraspConfigDAO" AS "GraspConfigDAO_1" '
        'ON "GraspConfigDAO_1".database_id = "WorldEntityDAO_1".database_id) ON '
        '"GraspConfigDAO_1".database_id = "MoveActionDAO".grasp_config_id JOIN '
        '("SymbolDAO" AS "SymbolDAO_2" JOIN "WorldEntityDAO" AS '
        '"WorldEntityDAO_2" ON "WorldEntityDAO_2".database_id = '
        '"SymbolDAO_2".database_id JOIN "ConnectionDAO" AS "ConnectionDAO_1" ON '
        '"ConnectionDAO_1".database_id = "WorldEntityDAO_2".database_id JOIN '
        '"FixedConnectionDAO" AS "FixedConnectionDAO_1" ON '
        '"FixedConnectionDAO_1".database_id = "ConnectionDAO_1".database_id) ON '
        '"ConnectionDAO_1".parent_id = "MoveActionDAO".grasp_config_id \n'
        'WHERE "MoveActionDAO".robot_x > :robot_x_1'
    )
    assert str(translator.sql_query) == expected_sql


def test_set_of_multi_variable_evaluate(session, database):
    """Verify that evaluate() for set_of with multiple variables returns dicts."""
    world = World(1, [
        Container("Container1"),
        Handle("Handle1"),
    ])
    fc = FixedConnection(world.bodies[0], world.bodies[1])
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
    assert isinstance(result, dict)
    assert result[C].name == "Container1"
    assert result[H].name == "Handle1"


def test_set_of_attribute_evaluate(session, database):
    """Verify that evaluate() for set_of with Attribute variables returns dicts."""
    session.add(BodyDAO(name="Body1", size=10))
    session.add(BodyDAO(name="Body2", size=20))
    session.commit()

    b = variable(type_=Body, domain=[])
    query = an(set_of(b.name, b.size))

    translator = eql_to_sql(query, session)
    results = translator.evaluate()

    assert len(results) == 2
    assert isinstance(results[0], dict)
    keys = list(results[0].keys())
    assert len(keys) == 2
    names = sorted([r[keys[0]] for r in results])
    assert names == ["Body1", "Body2"]

def test_set_of_transitive_evaluate(session, database):
    """Verify evaluate() for set_of with transitive attributes returns dicts."""
    session.add(
        KRROODPoseDAO(
            position=KRROODPositionDAO(x=1.0, y=2.0, z=3.0),
            orientation=KRROODOrientationDAO(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    )
    session.commit()

    pose = variable(type_=KRROODPose, domain=[])
    query = an(set_of(
        pose.position.x,
        pose.position.y,
        pose.position.z,
    ))

    translator = eql_to_sql(query, session)
    results = translator.evaluate()

    assert len(results) == 1
    assert isinstance(results[0], dict)
    keys = list(results[0].keys())
    assert len(keys) == 3
    values = list(results[0].values())
    assert 1.0 in values
    assert 2.0 in values
    assert 3.0 in values

def test_big_query_select_part(session):
    """
    Simulate the SELECT part of the big plan query using existing test classes.

    Simulates:
    SELECT pu.arm, v_pick.x, v_pick.y, v_place.x, v_place.y, v_end.z
    FROM PickUpActionDAO pu
    JOIN NavigateActionDAO np ON np.database_id = pa.pick_nav_id
    JOIN NavigateActionDAO np2 ON np2.database_id = pa.place_nav_id
    JOIN PoseMappingDAO pm_end ON ...
    JOIN Vector3MappingDAO v_end ON ...
    WHERE v_end.z < 0.9
    ORDER BY pa.plan_id

    Using MoveAction (simulates PickUp/Navigate) and GraspConfig (simulates Pose/Vector3).
    """
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
        ).where(
            move_pick.grasp_config.rotate_gripper < 0.9
        ).ordered_by(move_pick.robot_x)
    )

    translator = eql_to_sql(query, session)
    expected_sql = (
        'SELECT "MoveActionDAO".robot_x, "MoveActionDAO".robot_y, '
        '"MoveActionDAO".robot_x AS robot_x__1, '
        '"MoveActionDAO".robot_y AS robot_y__1, '
        '"GraspConfigDAO_1".rotate_gripper, '
        '"GraspConfigDAO_1".approach_direction \n'
        'FROM "SymbolDAO" JOIN "WorldEntityDAO" ON "WorldEntityDAO".database_id = '
        '"SymbolDAO".database_id JOIN "MoveActionDAO" ON '
        '"MoveActionDAO".database_id = "WorldEntityDAO".database_id JOIN '
        '("SymbolDAO" AS "SymbolDAO_1" JOIN "WorldEntityDAO" AS '
        '"WorldEntityDAO_1" ON "WorldEntityDAO_1".database_id = '
        '"SymbolDAO_1".database_id JOIN "GraspConfigDAO" AS "GraspConfigDAO_1" '
        'ON "GraspConfigDAO_1".database_id = "WorldEntityDAO_1".database_id) ON '
        '"GraspConfigDAO_1".database_id = "MoveActionDAO".grasp_config_id \n'
        'WHERE "GraspConfigDAO_1".rotate_gripper < :rotate_gripper_1 '
        'ORDER BY "MoveActionDAO".robot_x'
    )
    assert str(translator.sql_query) == expected_sql


def test_cte_from_eql(session, database):
    """
    Verify that an EQL query can be translated to a CTE and used in an outer query.

    Simulates the WITH clause pattern:
    WITH large_bodies AS (SELECT * FROM BodyDAO WHERE size > 5)
    SELECT * FROM ContainerDAO JOIN large_bodies ON large_bodies.database_id = ContainerDAO.database_id
    """
    session.add(BodyDAO(name="SmallBody", size=1))
    session.add(ContainerDAO(name="LargeContainer", size=10))
    session.commit()

    b = variable(type_=Body, domain=[])
    inner_query = an(entity(b).where(b.size > 5))
    large_bodies = eql_to_sql(inner_query, session, as_common_table_expression="large_bodies")

    c = variable(type_=Container, domain=[])
    outer_translator = eql_to_sql(an(entity(c)), session)

    outer_translator.sql_query = (
        outer_translator.sql_query
        .join(large_bodies, large_bodies.c.database_id == ContainerDAO.database_id)
    )

    expected_sql = (
        'WITH large_bodies AS \n'
        '(SELECT "BodyDAO".database_id AS database_id, '
        '"WorldEntityDAO".database_id AS database_id_2, '
        '"SymbolDAO".database_id AS database_id_3, '
        '"SymbolDAO".polymorphic_type AS polymorphic_type, '
        '"WorldEntityDAO".world_id AS world_id, '
        '"BodyDAO".name AS name, "BodyDAO".size AS size \n'
        'FROM "SymbolDAO" JOIN "WorldEntityDAO" ON '
        '"WorldEntityDAO".database_id = "SymbolDAO".database_id '
        'JOIN "BodyDAO" ON "BodyDAO".database_id = '
        '"WorldEntityDAO".database_id \n'
        'WHERE "BodyDAO".size > :size_1)\n'
        ' SELECT "ContainerDAO".database_id, "BodyDAO".database_id AS database_id_1, '
        '"WorldEntityDAO".database_id AS database_id_2, '
        '"SymbolDAO".database_id AS database_id_3, '
        '"SymbolDAO".polymorphic_type, "WorldEntityDAO".world_id, '
        '"BodyDAO".name, "BodyDAO".size \n'
        'FROM "SymbolDAO" JOIN "WorldEntityDAO" ON '
        '"WorldEntityDAO".database_id = "SymbolDAO".database_id '
        'JOIN "BodyDAO" ON "BodyDAO".database_id = '
        '"WorldEntityDAO".database_id JOIN "ContainerDAO" ON '
        '"ContainerDAO".database_id = "BodyDAO".database_id '
        'JOIN large_bodies ON large_bodies.database_id = "ContainerDAO".database_id'
    )
    assert str(outer_translator.sql_query) == expected_sql