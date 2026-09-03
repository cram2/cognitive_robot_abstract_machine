import numpy as np
import pytest
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Symbolic

from krrood.entity_query_language.factories import a
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.distributions.helper import make_dirac
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.relational.markov_chain import (
    MarkovChainDistributionTemplate,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    leaf,
)
from probabilistic_model.utils import MissingDict
from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneObjectType,
    SceneRoom,
)

# %% fixtures


@pytest.fixture
def type_variable() -> Symbolic:
    return Symbolic(name="type", domain=Set.from_iterable(SceneObjectType))


@pytest.fixture
def dirac_hidden_state_template(type_variable):
    """
    A ``RelationalProbabilisticCircuit(SceneObject)`` whose class circuit is a
    2-state hidden Markov emission model built by hand: one state always emits
    ``CHAIR``, the other always emits ``TABLE``. The dirac emissions make the
    "type" observed at each grounded position reveal the hidden state exactly,
    so the joint probability of a sequence of observations can be computed
    independently and compared against the grounded circuit's own
    ``probability``.
    """
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(
        leaf(make_dirac(type_variable, SceneObjectType.CHAIR), circuit), np.log(0.5)
    )
    root.add_subcircuit(
        leaf(make_dirac(type_variable, SceneObjectType.TABLE), circuit), np.log(0.5)
    )
    template_distribution = RelationalProbabilisticCircuit(SceneObject)
    template_distribution.class_probabilistic_circuit = circuit
    return template_distribution


@pytest.fixture
def type_by_state(dirac_hidden_state_template) -> list:
    """
    The type each hidden state deterministically emits, indexed the same way
    ``SumUnit.latent_variable`` indexes ``root.subcircuits`` -- i.e.
    ``type_by_state[i]`` is the type emitted by hidden state ``i``.

    ``root.subcircuits`` is not guaranteed to preserve the order ``add_subcircuit`` was
    called in, so this has to be discovered from the fitted leaves themselves rather
    than assumed from construction order.
    """
    root = dirac_hidden_state_template.class_probabilistic_circuit.root
    return [
        next(
            candidate
            for candidate in SceneObjectType
            if subcircuit.distribution.probabilities[hash(candidate)] == 1.0
        )
        for subcircuit in root.subcircuits
    ]


@pytest.fixture
def state_domain() -> Set:
    return Set.from_iterable(range(2))


@pytest.fixture
def starting_probabilities() -> np.ndarray:
    return np.array([0.3, 0.7])


@pytest.fixture
def transition_probabilities() -> np.ndarray:
    return np.array([[0.9, 0.1], [0.2, 0.8]])


@pytest.fixture
def starting_distribution(state_domain, starting_probabilities) -> SymbolicDistribution:
    return SymbolicDistribution(
        variable=Symbolic(name="state", domain=state_domain),
        probabilities=MissingDict(
            float, {hash(state): p for state, p in enumerate(starting_probabilities)}
        ),
    )


@pytest.fixture
def transition_model(state_domain, transition_probabilities) -> MultinomialDistribution:
    return MultinomialDistribution(
        distribution_variables=(
            Symbolic(name="state", domain=state_domain),
            Symbolic(name="next_state", domain=state_domain),
        ),
        probabilities=transition_probabilities,
    )


@pytest.fixture
def markov_chain_template(
    dirac_hidden_state_template, starting_distribution, transition_model
):
    return MarkovChainDistributionTemplate(
        template_distribution=dirac_hidden_state_template,
        starting_distribution=starting_distribution,
        transition_model=transition_model,
    )


def _scene_object_parts(count: int) -> list:
    """
    Resolve ``count`` independent ``SceneObject`` query parts, one per sequence
    position, the same way :class:`ExchangeableDistributionTemplate` receives one query
    part per exchangeable child.
    """
    room_query = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(count)],
    )
    room_query.resolve()
    return room_query.kwargs["objects"]


# %% __post_init__ validation


def test_mismatched_transition_shape_raises(dirac_hidden_state_template):
    mismatched_domain = Set.from_iterable(range(1))
    with pytest.raises(ShapeMismatchError):
        MarkovChainDistributionTemplate(
            template_distribution=dirac_hidden_state_template,
            starting_distribution=SymbolicDistribution(
                variable=Symbolic(name="state", domain=Set.from_iterable(range(2))),
                probabilities=MissingDict(float, {0: 0.5, 1: 0.5}),
            ),
            transition_model=MultinomialDistribution(
                distribution_variables=(
                    Symbolic(name="state", domain=mismatched_domain),
                    Symbolic(name="next_state", domain=mismatched_domain),
                ),
                probabilities=np.array([[1.0]]),
            ),
        )


# %% ground()


def test_ground_with_no_parts_returns_empty_circuit(markov_chain_template):
    grounded = markov_chain_template.ground([])
    assert len(grounded.nodes()) == 0


def test_ground_single_part_uses_starting_distribution(
    markov_chain_template, starting_probabilities, type_by_state
):
    [part] = _scene_object_parts(1)
    grounded = markov_chain_template.ground([part])
    assert grounded.is_valid()
    [type_variable_at_position] = [
        v for v in grounded.variables if v.name == f"{part.variable}.type"
    ]
    for state, type_value in enumerate(type_by_state):
        probability = grounded.probability(
            SimpleEvent.from_data(
                {type_variable_at_position: type_value}
            ).as_composite_set()
        )
        assert probability == pytest.approx(starting_probabilities[state])


def test_ground_three_parts_is_valid(markov_chain_template):
    parts = _scene_object_parts(3)
    assert markov_chain_template.ground(parts).is_valid()


def test_ground_three_parts_matches_forward_algorithm(
    markov_chain_template, starting_probabilities, transition_probabilities, type_by_state
):
    """
    With dirac emissions, the observed "type" sequence reveals the hidden state sequence
    exactly, so the joint probability of every state path is.

    independently computable as
    ``starting_probabilities[i] * transition_probabilities[i, j] * transition_probabilities[j, k]``
    -- the textbook forward-algorithm formula for a length-3 chain -- and must
    match what the grounded circuit itself reports.
    """
    parts = _scene_object_parts(3)
    grounded = markov_chain_template.ground(parts)
    type_variables_by_position = [
        next(v for v in grounded.variables if v.name == f"{part.variable}.type")
        for part in parts
    ]

    state_count = len(type_by_state)
    for i in range(state_count):
        for j in range(state_count):
            for k in range(state_count):
                event = SimpleEvent.from_data(
                    {
                        type_variables_by_position[0]: type_by_state[i],
                        type_variables_by_position[1]: type_by_state[j],
                        type_variables_by_position[2]: type_by_state[k],
                    }
                ).as_composite_set()
                expected = (
                    starting_probabilities[i]
                    * transition_probabilities[i, j]
                    * transition_probabilities[j, k]
                )
                assert grounded.probability(event) == pytest.approx(expected)


def test_ground_position_probability_matches_forward_marginal(
    markov_chain_template, starting_probabilities, transition_probabilities, type_by_state
):
    """
    Each position's "type" probability, with the other positions left.

    unconstrained, must recover that position's forward marginal
    (``starting_probabilities @ transition_probabilities ** t``).
    """
    parts = _scene_object_parts(3)
    grounded = markov_chain_template.ground(parts)

    forward_marginal = starting_probabilities
    for part in parts:
        type_variable_at_position = next(
            v for v in grounded.variables if v.name == f"{part.variable}.type"
        )
        for state, type_value in enumerate(type_by_state):
            probability = grounded.probability(
                SimpleEvent.from_data(
                    {type_variable_at_position: type_value}
                ).as_composite_set()
            )
            assert probability == pytest.approx(forward_marginal[state])
        forward_marginal = forward_marginal @ transition_probabilities
