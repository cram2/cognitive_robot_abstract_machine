import pytest
from sqlalchemy import select
from sqlalchemy.exc import MultipleResultsFound

from krrood.entity_query_language.failures import MultipleSolutionFound
from ..dataset.example_classes import Position, Pose
from ..dataset.semantic_world_like_classes import (
    World,
    Body,
    FixedConnection,
    PrismaticConnection,
    Handle,
    Container,
)
from ..dataset.ormatic_interface import (
    PositionDAO,
    PoseDAO,
    OrientationDAO,
    FixedConnectionDAO,
    PrismaticConnectionDAO,
    BodyDAO,
)
from krrood.entity_query_language.entity import (
    variable,
    entity,
    contains,
    and_,
    or_,
    in_,
)
from krrood.entity_query_language.entity_result_processors import an, the
from krrood.ormatic.dao import to_dao
from krrood.ormatic.eql_interface import eql_to_sql


def test_translate_simple_greater(session, database):
    session.add(PositionDAO(x=1, y=2, z=3))
    session.add(PositionDAO(x=1, y=2, z=4))
    session.commit()

    position = variable(type_=Position, domain=[])
    query = an(entity(position).where(position.z > 3))

    translator = eql_to_sql(query, session)
    query_by_hand = select(PositionDAO).where(PositionDAO.z > 3)

    assert str(translator.sql_query) == str(query_by_hand)

    results = translator.evaluate()

    assert len(results) == 1
    assert isinstance(results[0], PositionDAO)
    assert results[0].z == 4


def test_translate_or_condition(session, database):
    session.add(PositionDAO(x=1, y=2, z=3))
    session.add(PositionDAO(x=1, y=2, z=4))
    session.add(PositionDAO(x=2, y=9, z=10))
    session.commit()

    position = variable(type_=Position, domain=[])
    query = an(
        entity(position).where(
            or_(position.z == 4, position.x == 2),
        )
    )

    translator = eql_to_sql(query, session)

    query_by_hand = select(PositionDAO).where(
        (PositionDAO.z == 4) | (PositionDAO.x == 2)
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
        PoseDAO(
            position=PositionDAO(x=1, y=2, z=3),
            orientation=OrientationDAO(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    )
    session.add(
        PoseDAO(
            position=PositionDAO(x=1, y=2, z=4),
            orientation=OrientationDAO(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    )
    session.commit()

    pose = variable(type_=Pose, domain=[])
    query = an(entity(pose).where(pose.position.z > 3))
    translator = eql_to_sql(query, session)
    query_by_hand = select(PoseDAO).join(PoseDAO.position).where(PositionDAO.z > 3)

    assert str(translator.sql_query) == str(query_by_hand)

    result = translator.evaluate()

    # Assert: only the pose with position.z == 4 should match
    assert len(result) == 1
    assert isinstance(result[0], PoseDAO)
    assert result[0].position is not None
    assert result[0].position.z == 4


def test_translate_in_operator(session, database):
    session.add(PositionDAO(x=1, y=2, z=3))
    session.add(PositionDAO(x=5, y=2, z=6))
    session.add(PositionDAO(x=7, y=8, z=9))
    session.commit()

    position = variable(Position, domain=[])
    query = an(
        entity(position).where(
            in_(position.x, [1, 7]),
        )
    )

    # Act
    translator = eql_to_sql(query, session)

    query_by_hand = select(PositionDAO).where(PositionDAO.x.in_([1, 7]))
    assert str(translator.sql_query) == str(query_by_hand)

    result = translator.evaluate()

    # Assert: x in {1,7}
    xs = sorted([r.x for r in result])
    assert xs == [1, 7]


def test_the_quantifier(session, database):
    position_daos = [PositionDAO(x=1, y=2, z=3), PositionDAO(x=5, y=2, z=6)]
    positions = [Position(x=dao.x, y=dao.y, z=dao.z) for dao in position_daos]
    session.add_all(position_daos)
    session.commit()

    def get_query(domain=None):
        position = variable(
            type_=Position,
            domain=domain,
        )
        query = the(
            entity(position).where(
                position.y == 2,
            )
        )
        return query

    with pytest.raises(MultipleSolutionFound):
        result = get_query(positions).evaluate()

    translator = eql_to_sql(get_query(), session)
    query_by_hand = select(PositionDAO).where(PositionDAO.y == 2)
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
        name="prismatic_connection",
    )
    fixed_connection = variable(
        FixedConnection, domain=world.connections, name="fixed_connection"
    )

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
    assert str(translator.sql_query) == str(query_by_hand)

    result = translator.evaluate()

    assert len(result) == 1
    assert isinstance(result[0], FixedConnectionDAO)
    assert result[0].parent.name == "Container2"
    assert result[0].child.name == "Handle2"


@pytest.mark.skip(reason="Not finished yet-")
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
    parent_container = variable(
        type_=Container, domain=world.bodies, name="parent_connection"
    )
    prismatic_connection = variable(
        type_=PrismaticConnection,
        domain=world.connections,
        name="prismatic_connection",
    )
    drawer_body = variable(type_=Container, domain=world.bodies, name="drawer_body")
    fixed_connection = variable(
        type_=FixedConnection, domain=world.connections, name="fixed_connection"
    )
    handle = variable(type_=Handle, domain=world.bodies, name="handle")

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

    print(query.evaluate())

    # translator = eql_to_sql(query, session)


def test_contains(session, database):
    body1 = BodyDAO(name="Body1", size=1)
    session.add(body1)
    session.add(BodyDAO(name="Body2", size=1))
    session.add(BodyDAO(name="Body3", size=1))
    session.commit()

    b = variable(type_=Body, domain=[], name="b")
    query = an(
        entity(b).where(
            contains("Body1TestName", b.name),
        )
    )
    translator = eql_to_sql(query, session)

    result = translator.evaluate()

    assert body1 == result[0]
