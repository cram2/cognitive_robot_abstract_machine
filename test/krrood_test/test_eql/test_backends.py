from sqlalchemy.orm import sessionmaker

from krrood.entity_query_language.backends import (
    SQLAlchemyBackend,
    PythonBackend,
    ProbabilisticBackend,
)
from krrood.entity_query_language.factories import (
    variable,
    entity,
    an,
    probable_variable,
    probable,
    variable_from,
)
from krrood.ormatic.dao import to_dao
from krrood.probabilistic_knowledge.model_registries import DictRegistry
from krrood.probabilistic_knowledge.parameterizer import MatchParameterizer
from krrood.probabilistic_knowledge.probable_variable import MatchToInstanceTranslator
from ..dataset.example_classes import Pose, Position, Orientation


def test_same_query_multiple_domains(session, database):

    p1 = Pose(position=Position(1, 0, 0), orientation=Orientation(0, 0, 0, 1))
    p2 = Pose(position=Position(0, 1, 0), orientation=Orientation(0, 0, 0, 1))

    python_domain = [p1, p2]

    daos = [to_dao(p1), to_dao(p2)]
    session.add_all(daos)
    session.commit()
    session_maker = sessionmaker(session.bind)

    pose_variable = variable(Pose, python_domain)

    q = an(
        entity(pose_variable).where(
            pose_variable.position.x > 0.5,
        )
    )

    python_backend = PythonBackend()
    result = list(python_backend.evaluate(q))
    assert len(result) == 1

    database_backend = SQLAlchemyBackend(session_maker)
    result = list(database_backend.evaluate(q))
    assert len(result) == 1

    probable_pose = probable_variable(Pose)

    prob_q = probable_pose(
        position=probable(Position)(x=..., y=..., z=...),
        orientation=probable(Orientation)(x=0.0, y=0.0, z=0.0, w=1.0),
    ).where(probable_pose.variable.position.x > 0.5)

    parameters = MatchParameterizer(
        MatchToInstanceTranslator(prob_q).translate()
    ).parameterize()
    model = parameters.create_fully_factorized_distribution()

    registry = DictRegistry({Pose: model})

    pm_backend = ProbabilisticBackend(registry, 10)
    values = list(pm_backend.evaluate(prob_q))
    for value in values:
        assert value.position.x > 0.5

    assert pm_backend.number_of_samples == len({v.position for v in values})
