from __future__ import annotations

import copy
import itertools
from enum import IntEnum

import numpy as np
import pytest
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Symbolic

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.distributions.multinomial import MultinomialDistribution
from probabilistic_model.exceptions import ShapeMismatchError


class TwoCategories(IntEnum):
    A = 0
    B = 1


class ThreeCategories(IntEnum):
    A = 0
    B = 1
    C = 2


class FourCategories(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3


# %% fixtures


@pytest.fixture
def symbolic_x() -> Symbolic:
    """
    A variable over :class:`TwoCategories`.
    """
    return Symbolic(name="X", domain=Set.from_iterable(TwoCategories))


@pytest.fixture
def symbolic_y() -> Symbolic:
    """
    A variable over :class:`ThreeCategories`.
    """
    return Symbolic(name="Y", domain=Set.from_iterable(ThreeCategories))


@pytest.fixture
def symbolic_z() -> Symbolic:
    """
    A variable over :class:`FourCategories`.
    """
    return Symbolic(name="Z", domain=Set.from_iterable(FourCategories))


@pytest.fixture
def random_distribution(symbolic_x, symbolic_y, symbolic_z) -> MultinomialDistribution:
    """
    An unnormalized distribution over all three variables with random masses.
    """
    return MultinomialDistribution(
        (symbolic_x, symbolic_y, symbolic_z), np.random.rand(2, 3, 4)
    )


@pytest.fixture
def crafted_probabilities() -> np.ndarray:
    """
    The unnormalized masses of :func:`crafted_distribution`.
    """
    return np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]])


@pytest.fixture
def crafted_distribution(
    symbolic_x, symbolic_y, crafted_probabilities
) -> MultinomialDistribution:
    """
    An unnormalized distribution over *X* and *Y*.
    """
    return MultinomialDistribution((symbolic_x, symbolic_y), crafted_probabilities)


# %% construction


def test_creation_without_probabilities_is_uniform(symbolic_x):
    distribution = MultinomialDistribution([symbolic_x])
    assert np.allclose(distribution.probabilities, 1.0 / 2.0)


def test_creation_with_probabilities_of_the_wrong_shape_raises(symbolic_x, symbolic_y):
    with pytest.raises(ShapeMismatchError):
        MultinomialDistribution(
            [symbolic_x, symbolic_y], np.array([[0.1, 0.1], [0.2, 0.2]])
        )


def test_deep_copy_is_independent(crafted_distribution):
    probabilities_before = crafted_distribution.probabilities.copy()
    copied = copy.deepcopy(crafted_distribution)
    assert copied == crafted_distribution
    copied.probabilities[:] = 0.0
    np.testing.assert_array_equal(
        crafted_distribution.probabilities, probabilities_before
    )


def test_table_has_a_header_and_a_row_per_assignment(crafted_distribution):
    table = crafted_distribution.to_tabulate()
    assert table[0] == ["X", "Y", "P"]
    assert len(table) == 1 + crafted_distribution.probabilities.size


def test_serialization_round_trip(random_distribution):
    assert from_json(to_json(random_distribution)) == random_distribution


# %% inference


def test_normalize(random_distribution):
    random_distribution.normalize()
    assert random_distribution.probabilities.sum() == pytest.approx(1.0)


def test_marginal_is_normalized(random_distribution, symbolic_x, symbolic_y):
    marginal = random_distribution.marginal([symbolic_x, symbolic_y])
    assert marginal.probabilities.sum() == pytest.approx(1)
    assert marginal.variables == (symbolic_x, symbolic_y)


def test_marginal(crafted_distribution, crafted_probabilities, symbolic_x):
    marginal = crafted_distribution.marginal([symbolic_x])
    expected = crafted_probabilities.sum(axis=1) / crafted_probabilities.sum()
    assert marginal.probabilities == pytest.approx(expected)


def test_mode_is_the_most_likely_assignment(
    random_distribution, symbolic_x, symbolic_y, symbolic_z
):
    mode = random_distribution.mode()
    assert mode.likelihood == pytest.approx(random_distribution.probabilities.max())
    x_index, y_index, z_index = np.unravel_index(
        random_distribution.probabilities.argmax(),
        random_distribution.probabilities.shape,
    )
    assert (
        mode.event
        == SimpleEvent.from_data(
            {
                symbolic_x: TwoCategories(x_index),
                symbolic_y: ThreeCategories(y_index),
                symbolic_z: FourCategories(z_index),
            }
        ).as_composite_set()
    )


def test_mode_holds_every_most_likely_assignment(symbolic_x, symbolic_y):
    distribution = MultinomialDistribution(
        (symbolic_x, symbolic_y), np.array([[0.1, 0.7, 0.3], [0.7, 0.4, 0.1]])
    )
    mode = distribution.mode()
    assert mode.likelihood == 0.7
    assert mode.event == (
        SimpleEvent.from_data(
            {symbolic_x: TwoCategories.A, symbolic_y: ThreeCategories.B}
        ).as_composite_set()
        | SimpleEvent.from_data(
            {symbolic_x: TwoCategories.B, symbolic_y: ThreeCategories.A}
        ).as_composite_set()
    )


def test_likelihood(crafted_distribution, crafted_probabilities):
    crafted_distribution.normalize()
    data = np.array(
        [
            [TwoCategories.A, ThreeCategories.A],
            [TwoCategories.B, ThreeCategories.B],
        ]
    )
    likelihood = crafted_distribution.likelihood(data)
    mass = crafted_probabilities.sum()
    assert likelihood == pytest.approx([0.1 / mass, 0.4 / mass])


def test_probability(
    crafted_distribution, crafted_probabilities, symbolic_x, symbolic_y
):
    crafted_distribution.normalize()
    mass = crafted_probabilities.sum()
    event = SimpleEvent.from_data()
    assert crafted_distribution.probability(event.as_composite_set()) == pytest.approx(
        1
    )

    event[symbolic_x] = TwoCategories.A
    event._update_cpp_object()
    assert crafted_distribution.probability(event.as_composite_set()) == pytest.approx(
        crafted_probabilities[0].sum() / mass
    )

    event[symbolic_y] = (ThreeCategories.A, ThreeCategories.B)
    assert crafted_distribution.probability(event.as_composite_set()) == pytest.approx(
        crafted_probabilities[0, :2].sum() / mass
    )


def test_truncation(crafted_distribution, symbolic_y):
    event = SimpleEvent.from_data({symbolic_y: (ThreeCategories.A, ThreeCategories.B)})
    conditional = crafted_distribution.truncated(event.as_composite_set()).distribution
    assert conditional.probability(event.as_composite_set()) == pytest.approx(1)

    excluded = SimpleEvent.from_data({symbolic_y: ThreeCategories.C})
    excluded.fill_missing_variables(crafted_distribution.variables)
    assert conditional.probability(excluded.as_composite_set()) == 0.0


def test_as_probabilistic_circuit_keeps_every_probability(
    random_distribution, symbolic_x, symbolic_y, symbolic_z
):
    random_distribution.normalize()
    circuit = random_distribution.as_probabilistic_circuit().probabilistic_circuit
    variables = [symbolic_x, symbolic_y, symbolic_z]
    for assignment in itertools.product(
        *(variable.domain.simple_sets for variable in variables)
    ):
        event = SimpleEvent.from_data(zip(variables, assignment)).as_composite_set()
        assert circuit.probability(event) == pytest.approx(
            random_distribution.probability(event)
        )
