"""
Tests for the individual layer classes, the helpers and the integration with a learned
circuit.
"""

from __future__ import annotations

import numpy as np
import pytest
from random_events.interval import (
    Bound,
    SimpleInterval,
    closed,
    open,
    reals,
    singleton,
)
from random_events.product_algebra import SimpleEvent
from scipy.sparse import coo_array
from sortedcontainers import SortedSet

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.tensorized_to_rustworkx import (
    LayeredCircuitToRustworkxCircuitConverter,
)
from probabilistic_model.distributions.distributions import IntegerDistribution
from probabilistic_model.distributions.gaussian import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.exceptions import (
    NumberOfWeightsMismatchError,
    ShapeMismatchError,
)
from probabilistic_model.probabilistic_circuit.rx.helper import (
    fully_factorized,
    uniform_measure_of_event,
    uniform_measure_of_simple_event,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    NonPositiveScaleError,
)
from probabilistic_model.probabilistic_circuit.tensorized.forward_sample_assignment import (
    ForwardSampleAssignment,
    SampleRowsOfNode,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.inner_layer_edge import (
    InnerLayerEdge,
    InnerLayerEdges,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.dirac_delta_layer import (
    DiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    IntegerLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
    TruncatedGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.base import (
    AbstractContinuousLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
    SparseProbabilityTable,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layer_with_depth import (
    LayerWithDepth,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.moment_query import (
    MomentQuery,
)
from probabilistic_model.probabilistic_circuit.tensorized.row_grouped_sparse_array import (
    RowGroupedSparseArray,
    SparseEntries,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
    LogProbabilitiesOfLayers,
)
from probabilistic_model.probabilistic_circuit.tensorized.utils import (
    embedded_logsumexp,
)
from probabilistic_model.utils import MissingDict

from ..circuit_building import (
    correlated_gaussian_samples,
    fit_joint_probability_tree,
    uniform_layer_of,
)


@pytest.fixture(
    params=[
        (Bound.CLOSED, Bound.CLOSED),
        (Bound.CLOSED, Bound.OPEN),
        (Bound.OPEN, Bound.CLOSED),
        (Bound.OPEN, Bound.OPEN),
    ],
    ids=lambda bounds: f"{bounds[0].name.lower()}-{bounds[1].name.lower()}",
)
def event_bounds(request) -> tuple[Bound, Bound]:
    """
    Every combination of bounds of the interval an input layer is truncated to.
    """
    return request.param


# %% log-sum-exp


def test_embedded_logsumexp():
    np.testing.assert_allclose(
        embedded_logsumexp(
            np.array([[np.log(0.2), np.log(0.3)], [np.log(0.5), -np.inf]]), axis=1
        ),
        np.array([np.log(0.5), np.log(0.5)]),
    )


def test_logsumexp_of_only_minus_infinity_is_minus_infinity():
    # the padding of the edge gather is -inf, so a node without edges reduces to a
    # whole row of -inf and does not turn into a nan
    np.testing.assert_array_equal(
        embedded_logsumexp(np.array([[-np.inf, -np.inf]]), axis=1),
        np.array([-np.inf]),
    )


# %% decomposability


def test_decomposability_is_decided_per_node():
    leaf_x = uniform_layer_of(0, [(0, 1)])
    other_leaf_x = uniform_layer_of(0, [(1, 2)])
    leaf_y = uniform_layer_of(1, [(0, 1)])
    # node 0 multiplies x and y, node 1 multiplies x twice
    edges = coo_array(
        (np.zeros(4, dtype=np.int64), ([0, 1, 0, 2], [0, 0, 1, 1])), shape=(3, 2)
    )
    layer = ProductLayer([leaf_x, leaf_y, other_leaf_x], edges)
    np.testing.assert_array_equal(
        layer.is_decomposable_of_nodes(), np.array([True, False])
    )
    assert not layer.is_decomposable()


# %% truncation of input layers in a circuit


def test_truncating_to_a_composite_interval_introduces_a_selecting_sum_layer(x):
    circuit = LayeredProbabilisticCircuit(
        SortedSet([x]), SumLayer.mixture_of([uniform_layer_of(0, [(0, 4)])], [0.0])
    )
    event = SimpleEvent.from_data(
        {x: closed(0.0, 1.0) | closed(3.0, 4.0)}
    ).as_composite_set()
    truncation = circuit.truncated(event)

    assert truncation.probability == pytest.approx(0.5)
    # the single uniform node became two, mixed back together by a sum layer
    [uniform_layer] = [
        layer
        for layer in truncation.distribution.layers
        if isinstance(layer, UniformLayer)
    ]
    assert uniform_layer.number_of_nodes == 2
    np.testing.assert_allclose(
        truncation.distribution.likelihood(np.array([[0.5], [2.0], [3.5]])),
        np.array([0.5, 0.0, 0.5]),
    )


def test_truncating_to_a_singleton(x):
    circuit = LayeredProbabilisticCircuit(
        SortedSet([x]), SumLayer.mixture_of([uniform_layer_of(0, [(0, 2)])], [0.0])
    )
    event = SimpleEvent.from_data({x: singleton(1.0)}).as_composite_set()
    truncation = circuit.truncated(event, singleton_allowed=True)
    assert truncation.probability == pytest.approx(0.5)
    assert any(
        isinstance(layer, DiracDeltaLayer) for layer in truncation.distribution.layers
    )


def test_an_impossible_node_is_removed_but_its_siblings_survive(x):
    root = SumLayer(
        [uniform_layer_of(0, [(0, 1), (2, 3)])],
        RowGroupedSparseArray.from_entries(
            SparseEntries(np.log([0.25, 0.75]), [0, 0], [0, 1]), (1, 2)
        ),
    )
    circuit = LayeredProbabilisticCircuit(SortedSet([x]), root)
    event = SimpleEvent.from_data({x: closed(2.0, 3.0)}).as_composite_set()
    truncation = circuit.truncated(event)

    assert truncation.probability == pytest.approx(0.75)
    [remaining] = [
        layer
        for layer in truncation.distribution.layers
        if isinstance(layer, UniformLayer)
    ]
    assert remaining.number_of_nodes == 1
    np.testing.assert_allclose(
        truncation.distribution.likelihood(np.array([[2.5], [0.5]])),
        np.array([1.0, 0.0]),
    )


def test_a_structural_pass_does_not_write_into_the_circuit_it_reads(
    shared_children_circuit, x, y
):
    """
    Truncating a composite event reuses one circuit for every simple set instead of
    copying it, which is only sound because the pass builds new layers.
    """
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(shared_children_circuit)
    layers_before = list(layered.layers)
    parameters_before = [
        (type(layer).__name__, layer.number_of_nodes, to_json(layer))
        for layer in layers_before
    ]
    points = np.array([[0.5, 0.5], [1.5, 1.5]])
    likelihood_before = layered.log_likelihood(points)

    event = (
        SimpleEvent.from_data(
            {x: closed(0.0, 0.5), y: closed(0.0, 0.5)}
        ).as_composite_set()
        | SimpleEvent.from_data(
            {x: closed(1.5, 2.0), y: closed(1.5, 2.0)}
        ).as_composite_set()
    )
    layered.truncated(event.__deepcopy__())

    assert [id(layer) for layer in layered.layers] == [
        id(layer) for layer in layers_before
    ]
    assert [
        (type(layer).__name__, layer.number_of_nodes, to_json(layer))
        for layer in layered.layers
    ] == parameters_before
    np.testing.assert_array_equal(layered.log_likelihood(points), likelihood_before)


# %% truncation of input layers agrees with the distributions


@pytest.mark.parametrize(
    "node_bounds",
    [
        (Bound.CLOSED, Bound.CLOSED),
        (Bound.CLOSED, Bound.OPEN),
        (Bound.OPEN, Bound.CLOSED),
        (Bound.OPEN, Bound.OPEN),
    ],
    ids=lambda bounds: f"nodes {bounds[0].name.lower()}-{bounds[1].name.lower()}",
)
@pytest.mark.parametrize(
    "lower, upper", [(0.5, 2.5), (0.0, 4.0), (2.0, 2.0), (10.0, 11.0)]
)
def test_uniform_layer_agrees_with_the_scalar_truncation(
    x, node_bounds, event_bounds, lower, upper
):
    # supports that overlap the event fully, partially, at a point and not at all
    node_ranges = [(0.0, 1.0), (0.5, 2.5), (2.0, 3.0), (3.0, 4.0), (-1.0, 5.0)]
    distributions = [
        UniformDistribution(
            variable=x,
            interval=SimpleInterval.from_data(node_lower, node_upper, *node_bounds),
        )
        for node_lower, node_upper in node_ranges
    ]
    layer = UniformLayer.from_distributions(0, distributions)
    event_interval = SimpleInterval.from_data(lower, upper, *event_bounds)
    truncated = layer.log_truncated_of_assignment(
        event_interval.as_composite_set(), False
    )

    for node, distribution in enumerate(distributions):
        expected_truncation = distribution.log_conditional_from_simple_interval(
            event_interval, False
        )
        if expected_truncation.distribution is None:
            assert truncated.log_probabilities[node] == -np.inf
            continue
        assert float(truncated.log_probabilities[node]) == pytest.approx(
            float(expected_truncation.log_probability)
        )
        assert (
            truncated.layer.simple_interval_of(node)
            == expected_truncation.distribution.interval
        )


def test_a_singleton_turns_every_uniform_node_into_a_dirac_delta(x):
    layer = uniform_layer_of(0, [(0, 2), (1, 5)])
    assignment = singleton(1.0)
    truncated = layer.log_truncated_of_assignment(assignment, True)
    assert isinstance(truncated.layer, DiracDeltaLayer)
    for node in range(layer.number_of_nodes):
        expected_truncation = layer.node_distribution(
            node, x
        ).log_conditional_from_simple_interval(assignment.simple_sets[0], True)
        assert float(truncated.log_probabilities[node]) == pytest.approx(
            float(expected_truncation.log_probability)
        )
        assert (
            truncated.layer.node_distribution(node, x).location
            == expected_truncation.distribution.location
        )


def test_a_composite_assignment_mixes_the_truncations_to_its_simple_intervals(x):
    layer = uniform_layer_of(0, [(0, 4), (0, 8)])
    assignment = closed(0, 1) | closed(3, 4)
    truncated = layer.log_truncated_of_assignment(assignment, False)
    assert isinstance(truncated.layer, SumLayer)
    assert truncated.layer.number_of_nodes == layer.number_of_nodes
    for node in range(layer.number_of_nodes):
        expected = layer.node_distribution(node, x).probability(
            SimpleEvent.from_data({x: assignment}).as_composite_set()
        )
        assert float(np.exp(truncated.log_probabilities[node])) == pytest.approx(
            expected
        )


@pytest.fixture
def three_dirac_deltas() -> DiracDeltaLayer:
    """
    Dirac deltas at 0, 1 and 2.
    """
    return DiracDeltaLayer(0, np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    "assignment",
    [closed(1.0, 2.0), open(1.0, 2.0), closed(0.5, 1.5), closed(5.0, 6.0)],
    ids=str,
)
def test_dirac_delta_layer_probability_agrees_with_the_scalar_truncation(
    three_dirac_deltas, x, assignment
):
    event = SimpleEvent.from_data({x: assignment})
    probabilities = three_dirac_deltas.probability_of_simple_event_of_nodes(
        event, SortedSet([x])
    )
    expected_log_probabilities = [
        three_dirac_deltas.node_distribution(node, x)
        .log_truncated(event.as_composite_set())
        .log_probability
        for node in range(three_dirac_deltas.number_of_nodes)
    ]
    np.testing.assert_array_equal(probabilities, np.exp(expected_log_probabilities))


@pytest.mark.parametrize(
    "assignment",
    [
        closed(0.5, 1.5),
        closed(1.0, 1.0),
        open(1.0, 2.0),
        closed(0.0, 2.0),
        closed(5.0, 6.0),
    ],
    ids=str,
)
def test_dirac_delta_layer_agrees_with_the_scalar_truncation(
    three_dirac_deltas, x, assignment
):
    log_probabilities = three_dirac_deltas.log_truncated_of_assignment(
        assignment, False
    ).log_probabilities
    for node in range(three_dirac_deltas.number_of_nodes):
        expected = (
            three_dirac_deltas.node_distribution(node, x)
            .log_truncated(SimpleEvent.from_data({x: assignment}).as_composite_set())
            .log_probability
        )
        assert float(log_probabilities[node]) == float(expected)


def test_dirac_delta_layer_contains_a_location_an_interval_cannot_hold_exactly(x):
    """
    An interval holds its bounds with less precision than a location, so a singleton
    built from a location does not hold that location exactly.

    The layer still recognises it, or a circuit gives its own support probability zero.
    """
    location = 3.427709925174713
    layer = DiracDeltaLayer(0, np.array([location]), np.array([1.0]))
    assert singleton(location).simple_sets[0].lower != location
    assert layer.probability_of_simple_event_of_nodes(
        SimpleEvent.from_data({x: singleton(location)}), SortedSet([x])
    ) == np.array([1.0])


def test_dirac_delta_layer_excludes_a_location_outside_its_tolerance(x):
    layer = DiracDeltaLayer(0, np.array([1.0]), np.array([1.0]))
    assert layer.probability_of_simple_event_of_nodes(
        SimpleEvent.from_data({x: closed(1.0 + 1e-3, 2.0)}), SortedSet([x])
    ) == np.array([0.0])


@pytest.mark.parametrize(
    "assignment", [closed(0, 1), closed(2, 2), closed(5, 6)], ids=str
)
def test_discrete_layer_agrees_with_the_scalar_truncation(count, assignment):
    distributions = [
        IntegerDistribution(
            variable=count, probabilities=MissingDict(float, {0: 0.2, 1: 0.3, 2: 0.5})
        ),
        IntegerDistribution(variable=count, probabilities=MissingDict(float, {0: 1.0})),
    ]
    layer = IntegerLayer.from_distributions(0, distributions)
    truncated = layer.log_truncated_of_assignment(assignment, False)
    for node, distribution in enumerate(distributions):
        expected_truncation = distribution.log_truncated(
            SimpleEvent.from_data({count: assignment}).as_composite_set()
        )
        if expected_truncation.distribution is None:
            assert truncated.log_probabilities[node] == -np.inf
            continue
        assert float(truncated.log_probabilities[node]) == pytest.approx(
            float(expected_truncation.log_probability)
        )
        assert (
            truncated.layer.probabilities_of_node(node)
            == expected_truncation.distribution.probabilities
        )


@pytest.mark.parametrize("lower, upper", [(-1.0, 1.0), (-10.0, 10.0), (5.0, 6.0)])
def test_gaussian_layer_agrees_with_the_scalar_truncation(
    x, event_bounds, lower, upper
):
    node_parameters = [(0.0, 1.0), (2.0, 0.5), (-3.0, 2.0), (100.0, 0.01)]
    distributions = [
        GaussianDistribution(variable=x, location=location, scale=scale)
        for location, scale in node_parameters
    ]
    layer = GaussianLayer.from_distributions(0, distributions)
    event_interval = SimpleInterval.from_data(lower, upper, *event_bounds)
    truncated = layer.log_truncated_of_assignment(
        event_interval.as_composite_set(), False
    )

    for node, distribution in enumerate(distributions):
        expected_truncation = distribution.log_conditional_from_simple_interval(
            event_interval, False
        )
        if expected_truncation.distribution is None:
            assert truncated.log_probabilities[node] == -np.inf
            continue
        assert float(truncated.log_probabilities[node]) == pytest.approx(
            float(expected_truncation.log_probability)
        )
        assert isinstance(
            expected_truncation.distribution, TruncatedGaussianDistribution
        )
        assert (
            truncated.layer.simple_interval_of(node)
            == expected_truncation.distribution.interval
        )


def test_gaussian_layer_truncated_to_the_real_line_stays_gaussian(x):
    layer = GaussianLayer.from_distributions(
        0,
        [
            GaussianDistribution(variable=x, location=0.0, scale=1.0),
            GaussianDistribution(variable=x, location=5.0, scale=2.0),
        ],
    )
    truncated = layer.log_truncated_of_assignment(reals(), False)
    assert type(truncated.layer) is GaussianLayer
    np.testing.assert_allclose(truncated.log_probabilities, 0.0, atol=1e-9)
    np.testing.assert_array_equal(truncated.layer.location, layer.location)
    np.testing.assert_array_equal(truncated.layer.scale, layer.scale)


def test_gaussian_layer_marks_a_node_with_no_probability_left_as_impossible(x):
    layer = GaussianLayer.from_distributions(
        0,
        [
            GaussianDistribution(variable=x, location=0.0, scale=0.001),
            GaussianDistribution(variable=x, location=100.0, scale=1.0),
        ],
    )
    log_probabilities = layer.log_truncated_of_assignment(
        closed(99.0, 101.0), False
    ).log_probabilities
    assert log_probabilities[0] == -np.inf
    assert log_probabilities[1] > -np.inf


def test_a_singleton_turns_every_gaussian_node_into_a_dirac_delta(x):
    layer = GaussianLayer.from_distributions(
        0, [GaussianDistribution(variable=x, location=0.0, scale=1.0)]
    )
    truncated = layer.log_truncated_of_assignment(singleton(1.0), True)
    assert isinstance(truncated.layer, DiracDeltaLayer)
    assert float(truncated.log_probabilities[0]) == pytest.approx(
        float(layer.log_likelihood_of_nodes_from_column(np.array([1.0]))[0, 0])
    )


def test_a_composite_assignment_mixes_the_truncated_gaussians(x):
    distribution = GaussianDistribution(variable=x, location=0.0, scale=1.0)
    layer = GaussianLayer.from_distributions(0, [distribution])
    assignment = closed(-1, 0) | closed(1, 2)
    truncated = layer.log_truncated_of_assignment(assignment, False)
    assert isinstance(truncated.layer, SumLayer)
    assert isinstance(truncated.layer.child_layers[0], TruncatedGaussianLayer)
    assert float(np.exp(truncated.log_probabilities[0])) == pytest.approx(
        distribution.probability(
            SimpleEvent.from_data({x: assignment}).as_composite_set()
        )
    )


@pytest.mark.parametrize(
    "assignment", [closed(0.0, 1.0), open(-5.0, 0.5), closed(2.0, 10.0)], ids=str
)
def test_truncated_gaussian_layer_agrees_with_the_scalar_truncation(x, assignment):
    distributions = [
        TruncatedGaussianDistribution(
            variable=x,
            interval=closed(-1.0, 3.0).simple_sets[0],
            location=location,
            scale=scale,
        )
        for location, scale in ((0.0, 1.0), (2.5, 0.5))
    ]
    layer = TruncatedGaussianLayer.from_distributions(0, distributions)
    points = np.linspace(-1.5, 3.5, 11).reshape(-1, 1)
    truncated = layer.log_truncated_of_assignment(assignment, False)
    assert isinstance(truncated.layer, TruncatedGaussianLayer)
    for node, distribution in enumerate(distributions):
        expected_truncation = distribution.log_truncated(
            SimpleEvent.from_data({x: assignment}).as_composite_set()
        )
        assert truncated.log_probabilities[node] == pytest.approx(
            expected_truncation.log_probability
        )
        assert (
            truncated.layer.simple_interval_of(node)
            == expected_truncation.distribution.interval
        )
        np.testing.assert_allclose(
            np.exp(truncated.layer.log_likelihood_of_nodes(points)[:, node]),
            expected_truncation.distribution.likelihood(points),
            atol=1e-9,
        )


def test_conditioning_a_discrete_layer_on_a_value(count):
    layer = IntegerLayer.from_distributions(
        0,
        [
            IntegerDistribution(
                variable=count, probabilities=MissingDict(float, {0: 0.2, 2: 0.8})
            ),
            IntegerDistribution(
                variable=count, probabilities=MissingDict(float, {0: 1.0})
            ),
        ],
    )
    conditioned = layer.log_conditional_of_value(2)
    np.testing.assert_allclose(conditioned.log_probabilities, np.log([0.8, 0.0]))
    assert conditioned.layer.probabilities_of_node(0) == MissingDict(float, {2: 1.0})
    np.testing.assert_array_equal(
        layer.log_conditional_of_value(7).log_probabilities, [-np.inf, -np.inf]
    )


# %% traversal of the layer graph


@pytest.fixture
def leaf_x() -> UniformLayer:
    """
    A uniform distribution over the first variable in [0, 1].
    """
    return uniform_layer_of(0, [(0, 1)])


@pytest.fixture
def leaf_y() -> UniformLayer:
    """
    A uniform distribution over the second variable in [0, 2].
    """
    return uniform_layer_of(1, [(0, 2)])


@pytest.fixture
def product_of_leaves(leaf_x, leaf_y) -> ProductLayer:
    """
    The product of :func:`leaf_x` and :func:`leaf_y`.
    """
    return ProductLayer.product_of([leaf_x, leaf_y])


@pytest.fixture
def wrapper(product_of_leaves) -> SumLayer:
    """
    A mixture of only :func:`product_of_leaves`.
    """
    return SumLayer.mixture_of([product_of_leaves], [np.log(1.0)])


@pytest.fixture
def root_with_a_shared_layer(product_of_leaves, wrapper) -> SumLayer:
    """
    A mixture whose product layer is both its child and, through the wrapper, its
    grandchild, so it is reachable at two different depths.
    """
    return SumLayer.mixture_of([product_of_leaves, wrapper], np.log([0.5, 0.5]))


def test_an_input_layer_has_no_child_layers(leaf_x):
    assert leaf_x.child_layers == []


def test_all_layers_reports_a_shared_layer_once(
    root_with_a_shared_layer, product_of_leaves, wrapper, leaf_x, leaf_y
):
    layers = root_with_a_shared_layer.all_layers()
    assert {id(layer) for layer in layers} == {
        id(root_with_a_shared_layer),
        id(product_of_leaves),
        id(wrapper),
        id(leaf_x),
        id(leaf_y),
    }
    assert len({id(layer) for layer in layers}) == len(layers)


def test_all_layers_with_depth_pairs_every_layer_with_its_depth(
    root_with_a_shared_layer,
):
    assert root_with_a_shared_layer.all_layers_with_depth()[0] == LayerWithDepth(
        0, root_with_a_shared_layer
    )


def test_all_layers_with_depth_reports_a_shared_layer_once_per_path(
    root_with_a_shared_layer, product_of_leaves
):
    assert sorted(
        entry.depth
        for entry in root_with_a_shared_layer.all_layers_with_depth()
        if entry.layer is product_of_leaves
    ) == [1, 2]


def test_all_layers_visits_every_parent_before_the_layer(root_with_a_shared_layer):
    order = root_with_a_shared_layer.all_layers()
    positions = {id(layer): index for index, layer in enumerate(order)}
    for layer in order:
        for child_layer in layer.child_layers:
            assert positions[id(layer)] < positions[id(child_layer)]


# %% helpers that build circuits


def test_uniform_measure_of_a_simple_event(x, y):
    event = SimpleEvent.from_data({x: closed(0.0, 2.0), y: closed(0.0, 4.0)})
    circuit = RustworkxCircuitToLayeredCircuitConverter.convert(
        uniform_measure_of_simple_event(event)
    )
    assert list(circuit.variables) == [x, y]
    np.testing.assert_allclose(
        circuit.likelihood(np.array([[1.0, 2.0]])), np.array([1 / 8])
    )
    assert circuit.probability_of_simple_event(event) == pytest.approx(1.0)


def test_fully_factorized(x, y):
    circuit = RustworkxCircuitToLayeredCircuitConverter.convert(
        fully_factorized([x, y], means={x: 1.0}, variances={y: 2.0})
    )
    assert list(circuit.variables) == [x, y]
    assert circuit.expectation()[x] == pytest.approx(1.0)
    assert circuit.expectation()[y] == pytest.approx(0.0)


def test_uniform_measure_of_a_composite_event(x, y):
    event = (
        SimpleEvent.from_data(
            {x: closed(0.0, 1.0), y: closed(0.0, 1.0)}
        ).as_composite_set()
        | SimpleEvent.from_data(
            {x: closed(2.0, 3.0), y: closed(2.0, 3.0)}
        ).as_composite_set()
    )
    circuit = RustworkxCircuitToLayeredCircuitConverter.convert(
        uniform_measure_of_event(event.__deepcopy__())
    )
    assert circuit.probability(event.__deepcopy__()) == pytest.approx(1.0)


# %% layers of a single node


def test_product_of_and_mixture_of(product_of_leaves, x, y):
    point = np.array([[0.5, 1.0]])
    product_circuit = LayeredProbabilisticCircuit(SortedSet([x, y]), product_of_leaves)
    np.testing.assert_allclose(product_circuit.likelihood(point), np.array([0.5]))

    mixture = SumLayer.mixture_of([product_of_leaves], [np.log(1.0)])
    mixture_circuit = LayeredProbabilisticCircuit(SortedSet([x, y]), mixture)
    np.testing.assert_allclose(mixture_circuit.likelihood(point), np.array([0.5]))


def test_mixture_of_rejects_a_wrong_number_of_weights(leaf_x):
    with pytest.raises(NumberOfWeightsMismatchError):
        SumLayer.mixture_of([leaf_x], [0.0, 0.0])


# %% datastructures of the queries


def test_edges_into_one_child_layer():
    edges = InnerLayerEdges(
        np.array([0, 0, 1, 1]), np.array([0, 1, 0, 0]), np.array([2, 0, 1, 3])
    )
    into_first = edges.into_child_layer(0)
    assert list(into_first) == [
        InnerLayerEdge(0, 0, 2),
        InnerLayerEdge(1, 0, 1),
        InnerLayerEdge(1, 0, 3),
    ]
    assert not into_first.every_node_at_most_once
    assert edges.into_child_layer(1).every_node_at_most_once


def test_product_layer_edges_per_child_layer():
    layer = ProductLayer.product_of(
        [uniform_layer_of(0, [(0, 1)]), uniform_layer_of(1, [(0, 1)])]
    )
    assert [list(edges) for edges in layer.edges_per_child_layer] == [
        [InnerLayerEdge(0, 0, 0)],
        [InnerLayerEdge(0, 1, 0)],
    ]


def test_sparse_entries_concatenate_into_one_array():
    entries = SparseEntries.concatenate(
        [SparseEntries([1.0], [0], [1]), SparseEntries([2.0], [1], [0])]
    )
    np.testing.assert_array_equal(
        entries.to_coo_array((2, 2)).toarray(), np.array([[0.0, 1.0], [2.0, 0.0]])
    )


def test_sample_rows_of_a_node_join_the_chunks_of_all_parents():
    rows = SampleRowsOfNode()
    assert rows.is_empty
    rows.add(np.array([0, 2]))
    rows.add(np.array([5]))
    assert not rows.is_empty
    np.testing.assert_array_equal(rows.rows, np.array([0, 2, 5]))


def test_moment_query_from_maps(x, y):
    query = MomentQuery.from_maps({y: 2}, {x: 1.0, y: 3.0}, SortedSet([x, y]))
    np.testing.assert_array_equal(query.order, np.array([0, 2]))
    np.testing.assert_array_equal(query.center, np.array([1.0, 3.0]))
    np.testing.assert_array_equal(query.requested, np.array([False, True]))
    assert query.number_of_variables == 2


def test_alive_nodes_of_recorded_and_unrecorded_layers():
    recorded = uniform_layer_of(0, [(0, 1), (1, 2)])
    unrecorded = uniform_layer_of(0, [(0, 1)])
    log_probabilities = LogProbabilitiesOfLayers()
    log_probabilities.record(
        LayerWithLogProbabilities(recorded, np.array([-np.inf, 0.0]))
    )
    np.testing.assert_array_equal(
        log_probabilities.alive_nodes_of(recorded), np.array([False, True])
    )
    np.testing.assert_array_equal(
        log_probabilities.alive_nodes_of(unrecorded), np.array([True])
    )


def test_uniform_moment_is_the_difference_of_the_antiderivative(x):
    layer = uniform_layer_of(0, [(0, 2), (1, 5)])
    np.testing.assert_allclose(
        layer.moment_of_nodes_own(1, 0.0, x), np.array([1.0, 3.0])
    )
    np.testing.assert_allclose(
        layer.antiderivative_of_moment_at(np.array([2.0, 5.0]), 1, 0.0),
        np.array([1.0, 25 / 8]),
    )


# %% sampling of continuous input layers


@pytest.fixture
def gaussian_layer() -> GaussianLayer:
    """
    Gaussians at -1 and 2.
    """
    return GaussianLayer(0, np.array([-1.0, 2.0]), np.array([0.5, 1.5]))


@pytest.fixture
def truncated_gaussian_layer() -> TruncatedGaussianLayer:
    """
    Three Gaussians, truncated to closed intervals around, beside and far from their
    location.
    """
    return TruncatedGaussianLayer(
        0,
        np.array([[-1.0, 0.5], [0.0, 3.0], [2.0, 10.0]]),
        np.full((3, 2), int(Bound.CLOSED), dtype=np.int64),
        np.array([0.0, 2.5, 0.0]),
        np.array([1.0, 0.5, 2.0]),
    )


@pytest.fixture
def overlapping_uniform_layer() -> UniformLayer:
    """
    Uniform distributions over [0, 2] and [1, 5].
    """
    return uniform_layer_of(0, [(0, 2), (1, 5)])


@pytest.fixture
def dirac_delta_layer() -> DiracDeltaLayer:
    """
    Dirac deltas at 0.5 and 3.
    """
    return DiracDeltaLayer(0, np.array([0.5, 3.0]), np.ones(2))


@pytest.fixture(
    params=[
        "gaussian_layer",
        "truncated_gaussian_layer",
        "overlapping_uniform_layer",
        "dirac_delta_layer",
    ]
)
def continuous_input_layer(request) -> AbstractContinuousLayer:
    """
    An input layer of every continuous type, with several nodes.
    """
    return request.getfixturevalue(request.param)


def test_samples_of_every_node_have_the_mean_of_the_node(continuous_input_layer, x):
    layer = continuous_input_layer
    nodes = np.repeat(np.arange(layer.number_of_nodes), 20000)
    samples = layer.sample_of_nodes(nodes, SortedSet([x]))
    means = layer.moment_of_nodes_own(1, 0.0, x)
    for node in range(layer.number_of_nodes):
        of_node = samples[nodes == node]
        assert of_node.mean() == pytest.approx(means[node], abs=0.05)
        assert np.all(
            np.isfinite(layer.log_likelihood_of_nodes_from_column(of_node)[:, node])
        )


def test_sampling_forward_fills_every_row_from_its_node(continuous_input_layer, x):
    layer = continuous_input_layer
    assignment = ForwardSampleAssignment.for_layers([layer])
    last = layer.number_of_nodes - 1
    assignment.assign(layer, last, np.array([0, 2]))
    assignment.assign(layer, 0, np.array([1]))
    assignment.assign(layer, last, np.array([3]))
    samples = np.full((4, 1), np.nan)
    layer.sample_forward(assignment, samples, SortedSet([x]))

    log_likelihoods = layer.log_likelihood_of_nodes_from_column(samples[:, 0])
    node_of_row = np.array([last, 0, last, last])
    assert np.all(np.isfinite(log_likelihoods[np.arange(4), node_of_row]))


# %% validation of locations and scales


def gaussian_layer_with_scale(
    scale: np.ndarray, truncated: bool, number_of_intervals: int = 2
) -> GaussianLayer:
    """
    :param scale: The scale of every node, each at location zero.
    :param truncated: Whether the Gaussians are truncated to [-1, 1].
    :param number_of_intervals: The number of nodes a truncated layer has intervals for.
    :return: The layer.
    """
    location = np.zeros(2)
    if not truncated:
        return GaussianLayer(0, location, scale)
    return TruncatedGaussianLayer(
        0,
        np.tile([[-1.0, 1.0]], (number_of_intervals, 1)),
        np.full((number_of_intervals, 2), int(Bound.CLOSED), dtype=np.int64),
        location,
        scale,
    )


@pytest.mark.parametrize("truncated", [False, True], ids=["gaussian", "truncated"])
def test_layers_with_positive_scales_are_valid(truncated):
    gaussian_layer_with_scale(np.array([0.5, 2.0]), truncated).validate()


@pytest.mark.parametrize("truncated", [False, True], ids=["gaussian", "truncated"])
@pytest.mark.parametrize(
    "scale",
    [np.array([1.0, 0.0]), np.array([-1.0, 1.0]), np.array([np.nan, 1.0])],
    ids=["zero", "negative", "nan"],
)
def test_non_positive_scale_is_invalid(truncated, scale):
    with pytest.raises(NonPositiveScaleError):
        gaussian_layer_with_scale(scale, truncated).validate()


def test_truncated_gaussian_layer_with_one_scale_per_location_is_required():
    with pytest.raises(ShapeMismatchError):
        gaussian_layer_with_scale(np.ones(3), truncated=True).validate()


def test_truncated_gaussian_layer_still_needs_one_bound_per_interval_end():
    layer = gaussian_layer_with_scale(np.ones(2), truncated=True)
    layer.bounds = layer.bounds[:1]
    with pytest.raises(ShapeMismatchError):
        layer.validate()


# %% sampling of discrete input layers


@pytest.fixture
def integer_distributions(count) -> list[IntegerDistribution]:
    """
    A distribution over 2, 5 and 9 and one that always takes 5.
    """
    return [
        IntegerDistribution(
            variable=count, probabilities=MissingDict(float, {2: 0.2, 5: 0.3, 9: 0.5})
        ),
        IntegerDistribution(variable=count, probabilities=MissingDict(float, {5: 1.0})),
    ]


@pytest.fixture(
    params=[DenseProbabilityTable, SparseProbabilityTable],
    ids=lambda table_type: table_type.__name__,
)
def integer_layer(request, integer_distributions) -> IntegerLayer:
    """
    :func:`integer_distributions` as a layer, with every way to store its table.
    """
    return IntegerLayer.from_distributions(0, integer_distributions, request.param)


def test_samples_of_every_discrete_node_follow_its_distribution(
    integer_layer, integer_distributions, count
):
    amount = 20000
    nodes = np.repeat(np.arange(integer_layer.number_of_nodes), amount)
    samples = integer_layer.sample_of_nodes(nodes, SortedSet([count]))
    for node, distribution in enumerate(integer_distributions):
        of_node = samples[nodes == node]
        for state, probability in distribution.probabilities.items():
            assert float(np.mean(of_node == state)) == pytest.approx(
                probability, abs=0.015
            )


def test_a_discrete_node_without_mass_samples_nothing(integer_layer, count):
    truncated = integer_layer.log_truncated_of_assignment(closed(2, 2), False).layer
    samples = truncated.sample_of_nodes(np.array([0, 1, 1]), SortedSet([count]))
    assert samples[0] == 2.0
    assert np.isnan(samples[1:]).all()


# %% support and mode without copies


def test_the_support_of_a_sum_does_not_alter_the_support_of_its_children(x):
    child = uniform_layer_of(0, [(0, 1), (2, 3)])
    root = SumLayer(
        [child],
        RowGroupedSparseArray.from_entries(
            SparseEntries(np.log([0.5, 0.5]), [0, 0], [0, 1]), (1, 2)
        ),
    )
    variables = SortedSet([x])
    [support] = root.support_of_nodes(variables)
    assert (
        support
        == SimpleEvent.from_data({x: closed(0, 1) | closed(2, 3)}).as_composite_set()
    )
    assert (
        child.support_of_nodes(variables)[0]
        == SimpleEvent.from_data({x: closed(0, 1)}).as_composite_set()
    )


def test_the_mode_of_a_product_fills_in_the_variables_of_the_other_factors(
    product_of_leaves, x, y
):
    [mode], values = product_of_leaves.log_mode_of_nodes(SortedSet([x, y]))
    assert (
        mode
        == SimpleEvent.from_data({x: closed(0, 1), y: closed(0, 2)}).as_composite_set()
    )
    assert float(values[0]) == pytest.approx(np.log(0.5))


# %% a learned circuit


@pytest.fixture(scope="module")
def joint_probability_tree() -> ProbabilisticCircuit:
    """
    A joint probability tree over three variables, which is a much larger and more
    irregular graph than the hand built circuits.
    """
    np.random.seed(69)
    return fit_joint_probability_tree(
        correlated_gaussian_samples(3), min_samples_per_quantile=100
    )


@pytest.fixture(scope="module")
def layered_joint_probability_tree(
    joint_probability_tree,
) -> LayeredProbabilisticCircuit:
    """
    :func:`joint_probability_tree` converted to layers.
    """
    return RustworkxCircuitToLayeredCircuitConverter.convert(joint_probability_tree)


def test_conversion_of_a_learned_circuit_is_valid(
    joint_probability_tree, layered_joint_probability_tree
):
    layered_joint_probability_tree.validate()
    assert list(layered_joint_probability_tree.variables) == list(
        joint_probability_tree.variables
    )
    assert layered_joint_probability_tree.is_decomposable()


def test_log_likelihood_of_a_learned_circuit(
    joint_probability_tree, layered_joint_probability_tree
):
    samples = joint_probability_tree.sample(500)
    np.testing.assert_allclose(
        layered_joint_probability_tree.log_likelihood(samples),
        joint_probability_tree.log_likelihood(samples),
    )


def test_probability_of_a_simple_event_of_a_learned_circuit(
    joint_probability_tree, layered_joint_probability_tree
):
    event = joint_probability_tree.support.bounding_box()
    assert layered_joint_probability_tree.probability_of_simple_event(
        event
    ) == pytest.approx(joint_probability_tree.probability_of_simple_event(event))


def test_expectation_of_a_learned_circuit(
    joint_probability_tree, layered_joint_probability_tree
):
    expectation = layered_joint_probability_tree.expectation()
    expected = joint_probability_tree.expectation()
    for variable in layered_joint_probability_tree.variables:
        assert expectation[variable] == pytest.approx(expected[variable])


def test_samples_of_a_learned_circuit_lie_in_its_support(
    layered_joint_probability_tree,
):
    samples = layered_joint_probability_tree.sample(2000)
    assert not np.any(np.isnan(samples))
    assert np.all(layered_joint_probability_tree.log_likelihood(samples) > -np.inf)


def test_truncation_of_a_learned_circuit(
    joint_probability_tree, layered_joint_probability_tree
):
    bounding_box = joint_probability_tree.support.bounding_box()
    variable = layered_joint_probability_tree.variables[0]
    interval = bounding_box[variable].simple_sets[0]
    half = SimpleInterval.from_data(
        interval.lower,
        (interval.lower + interval.upper) / 2,
        interval.left,
        interval.right,
    )
    event = SimpleEvent.from_data(
        {variable: half.as_composite_set()}
    ).as_composite_set()

    rx_truncation = joint_probability_tree.truncated(event.__deepcopy__())
    truncation = layered_joint_probability_tree.truncated(event.__deepcopy__())
    assert truncation.probability == pytest.approx(rx_truncation.probability)
    truncation.distribution.validate()
    samples = truncation.distribution.sample(300)
    np.testing.assert_allclose(
        truncation.distribution.log_likelihood(samples),
        rx_truncation.distribution.log_likelihood(samples),
    )


def test_round_trip_of_a_learned_circuit_through_rustworkx(
    joint_probability_tree, layered_joint_probability_tree
):
    back = LayeredCircuitToRustworkxCircuitConverter.convert(
        layered_joint_probability_tree
    )
    samples = joint_probability_tree.sample(300)
    np.testing.assert_allclose(
        back.log_likelihood(samples), joint_probability_tree.log_likelihood(samples)
    )


def test_json_round_trip_of_a_learned_circuit(
    joint_probability_tree, layered_joint_probability_tree
):
    restored = from_json(to_json(layered_joint_probability_tree))
    samples = joint_probability_tree.sample(300)
    np.testing.assert_allclose(
        restored.log_likelihood(samples),
        layered_joint_probability_tree.log_likelihood(samples),
    )


def test_a_learned_circuit_has_fewer_layers_than_nodes(
    layered_joint_probability_tree,
):
    # the point of the layout: many nodes are folded into few parameter blocks
    assert (
        len(layered_joint_probability_tree.layers)
        < layered_joint_probability_tree.number_of_nodes
    )
