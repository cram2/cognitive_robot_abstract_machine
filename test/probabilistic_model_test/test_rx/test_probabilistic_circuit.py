from __future__ import annotations

import copy
from enum import IntEnum

import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib import pyplot as plt
from random_events.interval import closed, open, singleton
from random_events.product_algebra import Event, SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic

from probabilistic_model.distributions.distributions import (
    DiracDeltaDistribution,
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.helper import make_dirac
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    UnivariateContinuousLeaf,
    UnivariateDiscreteLeaf,
    leaf,
)
from probabilistic_model.utils import MissingDict

from ..circuit_building import mixture_of_products_of_mixtures, uniform_leaf


class Letter(IntEnum):
    """
    The categories of a symbolic variable.
    """

    A = 0
    B = 1
    C = 2


# %% a mixture of products of mixtures


@pytest.fixture
def mixture_of_products(x, y) -> ProbabilisticCircuit:
    """
    A mixture of two products, each of a mixture over *x* and a mixture over *y*, that
    share their uniform leaves.
    """
    circuit = ProbabilisticCircuit()
    return mixture_of_products_of_mixtures(
        circuit,
        (uniform_leaf(x, 0, 1, circuit), uniform_leaf(x, 2, 3, circuit)),
        (uniform_leaf(y, 0, 1, circuit), uniform_leaf(y, 3, 4, circuit)),
        [((0.8, 0.2), (0.5, 0.5)), ((0.7, 0.3), (0.1, 0.9))],
    )


def test_samples_are_distinct(mixture_of_products):
    samples = mixture_of_products.sample(100)
    assert len(np.unique(samples, axis=0)) > 95


def test_truncation(mixture_of_products, x):
    event = SimpleEvent.from_data(
        {x: closed(0, 0.25) | closed(0.5, 0.75)}
    ).as_composite_set()
    probability = mixture_of_products.truncated(event).probability
    assert probability == pytest.approx(0.375)


def test_translation_moves_the_probability_along(mixture_of_products, x, y):
    mixture_of_products.apply_translation({x: 5, y: 10})
    event = SimpleEvent.from_data(
        {x: closed(5, 5.25) | closed(5.5, 5.75)}
    ).as_composite_set()
    assert mixture_of_products.probability(event) == pytest.approx(0.375)


def test_deep_copy_is_a_valid_independent_circuit(mixture_of_products):
    copied = copy.deepcopy(mixture_of_products)
    assert copied is not mixture_of_products
    assert copied.graph is not mixture_of_products.graph
    assert copied.is_valid()
    assert len(copied.nodes()) == len(mixture_of_products.nodes())
    assert len(copied.graph.edges()) == len(mixture_of_products.graph.edges())


def test_plot_structure_with_inference_results(mixture_of_products):
    mixture_of_products.log_likelihood(np.array([[0.5, 0.5]]))
    color_map = {
        mixture_of_products.root: "red",
        mixture_of_products.root.subcircuits[0]: "blue",
        mixture_of_products.leaves[0]: "green",
    }
    mixture_of_products.plot_structure(
        color_map,
        plot_inference=True,
        inference_representation=lambda node: round(
            node.result_of_current_query[0].item(), 2
        ),
    )
    assert len(plt.gcf().axes) > 0
    plt.close("all")


def test_plot_of_a_symbolic_leaf_has_traces():
    variable = Symbolic(name="x", domain=Set.from_iterable(Letter))
    probabilities = MissingDict(float, {int(Letter.A): 7 / 20, int(Letter.B): 13 / 20})
    circuit = ProbabilisticCircuit()
    circuit.add_node(
        UnivariateDiscreteLeaf(
            SymbolicDistribution(variable=variable, probabilities=probabilities)
        )
    )
    assert len(go.Figure(circuit.plot(), circuit.plotly_layout()).data) > 0


# %% conditioning


def test_truncation_to_an_event_outside_one_factor_is_impossible(x, y):
    product = ProductUnit(probabilistic_circuit=ProbabilisticCircuit())
    product.add_subcircuit(
        UnivariateContinuousLeaf(GaussianDistribution(variable=x, location=1, scale=1))
    )
    product.add_subcircuit(
        UnivariateContinuousLeaf(GaussianDistribution(variable=y, location=1, scale=1))
    )
    event = SimpleEvent.from_data(
        {
            x: closed(1.191472053527832, 1.2024999856948853),
            y: open(1500.0009765625, np.inf) | open(-np.inf, -1500.0009765625),
        }
    ).as_composite_set()
    truncated = product.probabilistic_circuit.truncated(event).distribution
    assert truncated is None


@pytest.fixture
def uniform_and_dirac_delta(x) -> ProbabilisticCircuit:
    """
    An equal mixture of a uniform distribution over [0, 1] and a Dirac delta at 0.5 with
    a density cap of two.
    """
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(uniform_leaf(x, 0, 1, circuit), np.log(0.5))
    root.add_subcircuit(
        leaf(
            DiracDeltaDistribution(variable=x, location=0.5, density_cap=2.0), circuit
        ),
        np.log(0.5),
    )
    return circuit


def test_truncation_to_the_whole_support_keeps_all_mass(uniform_and_dirac_delta, x):
    event = SimpleEvent.from_data({x: closed(0.0, 1.0)}).as_composite_set()
    probability = uniform_and_dirac_delta.truncated(event).probability
    assert probability == pytest.approx(1.0)


def test_truncation_beside_the_dirac_delta_drops_it(uniform_and_dirac_delta, x):
    event = SimpleEvent.from_data(
        {x: closed(0.0, 0.25) | closed(0.75, 1.0)}
    ).as_composite_set()
    truncation = uniform_and_dirac_delta.truncated(event)
    assert truncation.probability == pytest.approx(0.25)
    assert len(list(truncation.distribution.nodes())) == 3
    assert all(
        isinstance(node.distribution, UniformDistribution)
        for node in truncation.distribution.leaves
    )


def test_truncation_and_conditioning_on_the_dirac_delta_keep_only_it(
    uniform_and_dirac_delta, x
):
    event = SimpleEvent.from_data({x: singleton(0.5)}).as_composite_set()
    truncated = uniform_and_dirac_delta.truncated(event).distribution
    assert len(list(truncated.nodes())) == 1
    assert isinstance(truncated.root.distribution, DiracDeltaDistribution)

    conditioning = uniform_and_dirac_delta.conditional({x: 0.5})
    assert conditioning.likelihood == pytest.approx(1.5)
    assert len(list(conditioning.distribution.nodes())) == 1
    assert isinstance(
        conditioning.distribution.root.distribution, DiracDeltaDistribution
    )


@pytest.fixture
def mixture_of_nested_squares(x, y) -> ProbabilisticCircuit:
    """
    An equal mixture of the uniform distributions over [0, 1]² and over [0, 2]².
    """
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for upper in [1, 2]:
        product = ProductUnit(probabilistic_circuit=circuit)
        product.add_subcircuit(uniform_leaf(x, 0, upper, circuit))
        product.add_subcircuit(uniform_leaf(y, 0, upper, circuit))
        root.add_subcircuit(product, np.log(0.5))
    return circuit


def test_conditioning_on_a_point_moves_mass_towards_the_smaller_square(
    mixture_of_nested_squares, x, y
):
    event = SimpleEvent.from_data({y: closed(0.0, 1.0)}).as_composite_set()
    prior = mixture_of_nested_squares.marginal([y]).probability(event)
    conditional = mixture_of_nested_squares.conditional({x: 0.5}).distribution
    assert conditional.marginal([y]).probability(event) > prior


def test_conditioning_on_a_symbolic_point(mixture_of_nested_squares):
    circuit = copy.deepcopy(mixture_of_nested_squares)
    letter = Symbolic(name="s", domain=Set.from_iterable(Letter))
    probabilities = MissingDict(
        float, {hash(Letter.A): 7 / 20, hash(Letter.B): 13 / 20}
    )
    old_root = circuit.root
    new_root = ProductUnit(probabilistic_circuit=circuit)
    new_root.add_subcircuit(old_root)
    new_root.add_subcircuit(
        leaf(
            SymbolicDistribution(variable=letter, probabilities=probabilities), circuit
        )
    )
    probability = circuit.conditional({letter: Letter.A}).likelihood
    assert probability == pytest.approx(7 / 20)


def test_probability_of_a_point_that_only_one_component_supports(
    mixture_of_nested_squares, x
):
    """
    Only the second component has density at 1.5, so conditioning removes the root sum
    unit; its weight still is part of the probability.
    """
    log_probability = mixture_of_nested_squares.log_conditional({x: 1.5}).log_likelihood
    assert log_probability == pytest.approx(np.log(0.5 * 0.5))
    assert log_probability == pytest.approx(
        mixture_of_nested_squares.marginal([x]).log_likelihood(np.array([[1.5]]))[0]
    )


# %% expectation of a subset of variables


@pytest.mark.parametrize("variable_type", [Continuous, Integer])
def test_expectation_of_a_subset_of_the_variables(variable_type):
    first, second = variable_type("x"), variable_type("y")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    product = ProductUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(product, 1.0)
    product.add_subcircuit(leaf(make_dirac(first, 0), circuit))
    product.add_subcircuit(leaf(make_dirac(second, 0), circuit))
    assert circuit.expectation([first])[first] == 0


# %% truncation of mixed continuous and discrete leaves


@pytest.fixture
def mixed_circuit(x, count) -> ProbabilisticCircuit:
    """
    A mixture of two products, each of a uniform distribution over *x* and an integer
    distribution over *count*.
    """
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, (lower, upper), probabilities in [
        (0.6, (0.0, 1.0), {0: 0.5, 1: 0.3, 2: 0.2}),
        (0.4, (0.5, 2.0), {1: 0.4, 2: 0.4, 3: 0.2}),
    ]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(uniform_leaf(x, lower, upper, circuit))
        component.add_subcircuit(
            leaf(
                IntegerDistribution(
                    variable=count, probabilities=MissingDict(float, probabilities)
                ),
                circuit,
            )
        )
        root.add_subcircuit(component, np.log(weight))
    return circuit


@pytest.fixture
def mixed_event(x, count) -> Event:
    """
    An event of two boxes over *x* and *count*.
    """
    return (
        SimpleEvent.from_data(
            {x: closed(0.0, 0.7), count: closed(0, 1)}
        ).as_composite_set()
        | SimpleEvent.from_data(
            {x: closed(1.0, 1.8), count: closed(2, 3)}
        ).as_composite_set()
    )


def test_truncation_of_mixed_leaves_reports_the_probability_of_the_event(
    mixed_circuit, mixed_event
):
    log_probability = mixed_circuit.log_truncated(mixed_event).log_probability
    assert np.exp(log_probability) == pytest.approx(
        mixed_circuit.probability(mixed_event), abs=1e-9
    )


def test_truncation_of_mixed_leaves_places_all_mass_inside_the_event(
    mixed_circuit, mixed_event
):
    truncated = mixed_circuit.log_truncated(mixed_event).distribution
    assert truncated.probability(mixed_event) == pytest.approx(1.0, abs=1e-9)


# %% a circuit of a single leaf


@pytest.fixture
def single_leaf_circuit(x) -> ProbabilisticCircuit:
    """
    A circuit whose root is itself a leaf, with no wrapping product or sum unit.
    """
    circuit = ProbabilisticCircuit()
    uniform_leaf(x, 0, 1, circuit)
    return circuit


def test_a_leaf_is_its_own_only_leaf(single_leaf_circuit):
    assert single_leaf_circuit.root.leaves == [single_leaf_circuit.root]


def test_a_circuit_rooted_at_a_leaf_lists_that_leaf(single_leaf_circuit):
    assert single_leaf_circuit.leaves == [single_leaf_circuit.root]


def test_update_variables_renames_a_leaf_root(single_leaf_circuit, x):
    renamed = Continuous("renamed_x")
    single_leaf_circuit.update_variables({x: renamed})
    assert single_leaf_circuit.root.distribution.variable == renamed


# %% simplification of nested units


def test_simplify_flattens_a_nested_product_unit(x, y, z):
    circuit = ProbabilisticCircuit()
    outer = ProductUnit(probabilistic_circuit=circuit)
    inner = ProductUnit(probabilistic_circuit=circuit)
    leaf_x, leaf_y, leaf_z = (
        uniform_leaf(variable, 0, 1, circuit) for variable in (x, y, z)
    )
    inner.add_subcircuit(leaf_x)
    inner.add_subcircuit(leaf_y)
    outer.add_subcircuit(inner)
    outer.add_subcircuit(leaf_z)

    circuit.simplify()

    assert set(outer.subcircuits) == {leaf_x, leaf_y, leaf_z}
    assert inner.probabilistic_circuit is None
    assert len(circuit.nodes()) == 4


@pytest.mark.parametrize("unit_type", [ProductUnit, SumUnit])
def test_simplifying_one_parent_keeps_a_shared_child_for_the_other(unit_type, x, y, z):
    """
    A unit mounted below two parents of its own type, as relational grounding shares one
    grounded instance, still serves the second parent after the first absorbed it.
    """
    w = Continuous("w")
    circuit = ProbabilisticCircuit()
    shared = unit_type(probabilistic_circuit=circuit)
    leaf_x, leaf_y, leaf_z, leaf_w = (
        uniform_leaf(variable, 0, 1, circuit) for variable in (x, y, z, w)
    )
    shared.add_subcircuit(leaf_x, log_weight=0.0)
    shared.add_subcircuit(leaf_y, log_weight=0.0)
    first_parent = unit_type(probabilistic_circuit=circuit)
    first_parent.add_subcircuit(shared, log_weight=0.0)
    first_parent.add_subcircuit(leaf_z, log_weight=0.0)
    second_parent = unit_type(probabilistic_circuit=circuit)
    second_parent.add_subcircuit(shared, log_weight=0.0)
    second_parent.add_subcircuit(leaf_w, log_weight=0.0)

    first_parent.simplify()
    assert set(first_parent.subcircuits) == {leaf_x, leaf_y, leaf_z}
    second_parent.simplify()
    assert set(second_parent.subcircuits) == {leaf_x, leaf_y, leaf_w}
