from sqlalchemy.orm import sessionmaker

from krrood.entity_query_language.backends import (
    SQLAlchemyBackend,
    ProbabilisticBackend,
    EntityQueryLanguageBackend,
)
from krrood.entity_query_language.factories import (
    variable,
    entity,
    an,
    underspecified,
    variable_from,
)
from krrood.ormatic.dao import to_dao
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from random_events.set import Set
from random_events.variable import Symbolic
from ..dataset.example_classes import KRROODPose, KRROODPosition, KRROODOrientation


def test_same_query_multiple_backends(session, database):

    p1 = KRROODPose(
        position=KRROODPosition(1, 0, 0), orientation=KRROODOrientation(0, 0, 0, 1)
    )
    p2 = KRROODPose(
        position=KRROODPosition(0, 1, 0), orientation=KRROODOrientation(0, 0, 0, 1)
    )

    python_domain = [p1, p2]

    daos = [to_dao(p1), to_dao(p2)]
    session.add_all(daos)
    session.commit()
    session_maker = sessionmaker(session.bind)

    pose_variable = variable(KRROODPose, python_domain)

    q = an(
        entity(pose_variable).where(
            pose_variable.position.x > 0.5,
        )
    )

    python_backend = EntityQueryLanguageBackend()
    result = list(python_backend.evaluate(q))
    assert len(result) == 1

    database_backend = SQLAlchemyBackend(session_maker)
    result = list(database_backend.evaluate(q))
    assert len(result) == 1

    prob_q = underspecified(KRROODPose)(
        position=underspecified(KRROODPosition)(x=..., y=..., z=...),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    prob_q.expression
    prob_q.where(prob_q.variable.position.x > 0.5)

    parameters = UnderspecifiedParameters(prob_q)
    model = fully_factorized(parameters.variables.values())

    registry = DictRegistry({KRROODPose: model})

    pm_backend = ProbabilisticBackend(model_registry=registry, number_of_samples=10)
    values = list(pm_backend.evaluate(prob_q))
    for value in values:
        assert value.position.x > 0.5

    assert pm_backend.number_of_samples == len({v.position for v in values})


def test_probabilistic_backend_with_symbolic_expression():
    prob_q = underspecified(KRROODPosition)(
        x=..., y=..., z=variable(int, domain=[1, 2, 3])
    )
    parameters = UnderspecifiedParameters(prob_q)
    assert parameters.variables["KRROODPosition.z"] == Symbolic(
        "KRROODPosition.z", Set.from_iterable([1, 2, 3])
    )
