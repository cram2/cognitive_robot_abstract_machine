"""
Tests that a causal EQL query over a relational circuit fitted as layered circuits is
answered in that circuit type and gives the same answers as the same query over a
relational circuit fitted as graphs of units.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
from typing_extensions import Callable

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a, cause
from krrood.entity_query_language.operators.causal import ScoredIntervention
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import LayeredLearning
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.exchangeable_grounding import (
    ExchangeablePartGrounder,
)
from probabilistic_model.probabilistic_circuit.relational.part_learning_methods import (
    PartLearningMethods,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from ...dataset import ormatic_interface  # noqa: F401, registers the DAOs
from ...dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneObjectType,
    SceneRoom,
)
from ...test_feature_extraction.test_relational_probabilistic_circuit import (
    _correlated_rooms,
)

# %% the same data, fitted in either circuit type


@pytest.fixture
def units_model() -> RelationalProbabilisticCircuit:
    """
    The relational circuit fitted as graphs of units.
    """
    return RelationalProbabilisticCircuit(SceneRoom).fit(_correlated_rooms())


@pytest.fixture
def layers_model() -> RelationalProbabilisticCircuit:
    """
    The same relational circuit, fitted as layered circuits throughout.
    """
    return RelationalProbabilisticCircuit(
        SceneRoom,
        learning_method=LayeredLearning(JointProbabilityTree()),
        part_learning_methods=PartLearningMethods(
            LayeredLearning(JointProbabilityTree())
        ),
    ).fit(_correlated_rooms())


def cause_query() -> Match[SceneRoom]:
    """
    :return: A match that marks two coordinates of the position as cause candidates and
        declares the type of the first object as the effect.
    """
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=cause, y=cause, z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(3)],
    )
    query.causes_effect(query.objects[0].type == SceneObjectType.CHAIR)
    return query


def backend_over(model: RelationalProbabilisticCircuit) -> ProbabilisticBackend:
    """
    :param model: The relational circuit to answer queries with.
    :return: A backend that grounds it for every query.
    """
    return ProbabilisticBackend(
        model_registry=RelationalCircuitRegistry(
            relational_probabilistic_circuit=model
        ),
        number_of_samples=3,
    )


def causal_circuit_over(model: RelationalProbabilisticCircuit) -> CausalCircuit:
    """
    :param model: The relational circuit to ground.
    :return: The causal circuit the registry wraps its grounding of :func:`cause_query`
        in.
    """
    np.random.seed(0)
    return RelationalCircuitRegistry(relational_probabilistic_circuit=model).get_model(
        UnderspecifiedParameters(cause_query())
    )


@dataclass
class RankingsOfBothCircuitTypes:
    """
    The cause candidates of one query, ranked over the same data fitted in either
    circuit type.
    """

    over_units: list[ScoredIntervention]
    """
    The ranking over the relational circuit fitted as graphs of units.
    """

    over_layers: list[ScoredIntervention]
    """
    The ranking over the relational circuit fitted as layered circuits.
    """


def rankings_with_the_same_samples(
    units_backend: ProbabilisticBackend,
    layers_backend: ProbabilisticBackend,
    make_query: Callable[[], Match[SceneRoom]],
) -> RankingsOfBothCircuitTypes:
    """
    Rank the cause candidates of one query with both backends, the layered one replaying
    the samples of the undetermined statistics the other drew, since the two circuit
    types sample differently.

    :param units_backend: The backend over the circuit of units.
    :param layers_backend: The backend over the layered circuit.
    :param make_query: Builds a fresh match, since a backend consumes the one it is
        given.
    :return: Both rankings.
    """
    drawn = []
    sample = ExchangeablePartGrounder._sample_undetermined_latents

    def recording_sample(grounder, node=None):
        drawn.append(sample(grounder, node))
        return drawn[-1]

    np.random.seed(0)
    with patch.object(
        ExchangeablePartGrounder, "_sample_undetermined_latents", recording_sample
    ):
        over_units = units_backend.rank_causes(make_query())
    replayed = iter(drawn)
    with patch.object(
        ExchangeablePartGrounder,
        "_sample_undetermined_latents",
        lambda grounder, node=None: next(replayed),
    ):
        over_layers = layers_backend.rank_causes(make_query())
    assert next(replayed, None) is None
    return RankingsOfBothCircuitTypes(over_units=over_units, over_layers=over_layers)


# %% the circuit type a causal query is answered in


def test_a_layered_fit_is_wrapped_as_a_layered_causal_circuit(layers_model):
    causal_circuit = causal_circuit_over(layers_model)
    assert isinstance(causal_circuit, CausalCircuit)
    assert isinstance(causal_circuit.probabilistic_circuit, LayeredProbabilisticCircuit)


def test_a_fit_of_units_is_wrapped_as_a_causal_circuit_of_units(units_model):
    assert isinstance(
        causal_circuit_over(units_model).probabilistic_circuit, ProbabilisticCircuit
    )


def test_the_interventional_circuit_of_a_layered_fit_is_layered(layers_model):
    causal_circuit = causal_circuit_over(layers_model)
    assert isinstance(
        causal_circuit.backdoor_adjustment(
            causal_circuit.causal_variables[0], causal_circuit.effect_variables[0]
        ),
        LayeredProbabilisticCircuit,
    )


def test_a_layered_fit_narrows_a_ranked_candidate_to_a_layered_circuit(layers_model):
    [first, *_] = backend_over(layers_model).rank_causes(cause_query())
    assert isinstance(first.narrowed_circuit, LayeredProbabilisticCircuit)


# %% the answers both circuit types give


def test_both_circuit_types_rank_the_same_candidates(units_model, layers_model):
    rankings = rankings_with_the_same_samples(
        backend_over(units_model), backend_over(layers_model), cause_query
    )
    # both coordinates the query marks are candidates, so neither ranking is empty
    assert len(rankings.over_units) == 2
    assert [scored.cause_variable.name for scored in rankings.over_layers] == [
        scored.cause_variable.name for scored in rankings.over_units
    ]


def test_both_circuit_types_score_the_candidates_the_same(units_model, layers_model):
    rankings = rankings_with_the_same_samples(
        backend_over(units_model), backend_over(layers_model), cause_query
    )
    assert [
        scored.effect_probability_given_region for scored in rankings.over_layers
    ] == pytest.approx(
        [scored.effect_probability_given_region for scored in rankings.over_units]
    )


def test_a_layered_fit_answers_a_causal_query_with_rooms(layers_model):
    generated = list(backend_over(layers_model).evaluate(cause_query()))
    assert len(generated) == 3
    assert all(isinstance(room, SceneRoom) for room in generated)
