from __future__ import annotations

import json
import math
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import sklearn.datasets
from jpt.learning.preprocessing import preprocess_data as old_preprocess_data
from jpt.trees import JPT as OldJPT
from jpt.variables import infer_from_dataframe as old_infer_from_dataframe
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.distributions.distributions import (
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import (
    AnnotatedVariable,
    infer_variables_from_dataframe,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)


class Letter(Enum):
    """
    The categories of a symbolic column.
    """

    A = 0
    B = 1
    C = 2

    def __hash__(self) -> int:
        return hash(self.value)


# %% annotated variables


@pytest.mark.parametrize(
    "annotated_variable",
    [
        AnnotatedVariable(Integer(name="x"), 2, 1),
        AnnotatedVariable(Continuous(name="x"), 2, 3, 1.0, 0.1, 10),
        AnnotatedVariable(Symbolic(name="x", domain=Set.from_iterable(Letter))),
    ],
    ids=["integer", "continuous", "symbolic"],
)
def test_annotated_variable_serialization_round_trip(annotated_variable):
    assert from_json(to_json(annotated_variable)) == annotated_variable


# %% inference of variables from a data frame


@pytest.fixture
def mixed_data() -> pd.DataFrame:
    """
    A real, an integer and a symbolic column of 100 rows.
    """
    letters = list(Letter)
    return pd.DataFrame(
        {
            "integer": np.concatenate(
                (
                    np.random.randint(low=0, high=4, size=50),
                    np.random.randint(7, 10, 50),
                )
            ).astype(int),
            "real": np.random.normal(2, 4, 100).astype(np.float32),
            "symbol": [letters[index] for index in np.random.randint(0, 3, 100)],
        }
    )


def test_inferred_variables_are_named_after_their_columns(mixed_data):
    integer, real, symbol = infer_variables_from_dataframe(mixed_data)
    assert integer.variable.name == "integer"
    assert real.variable.name == "real"
    assert symbol.variable.name == "symbol"
    assert real.minimal_distance <= 1.0


def test_a_column_of_unknown_type_is_refused():
    with pytest.raises(ValueError):
        infer_variables_from_dataframe(pd.DataFrame({"time": [datetime.now()]}))


def test_string_dtype_column_is_treated_as_symbolic(monkeypatch):
    """
    With pandas' ``future.infer_string`` option enabled, the default from pandas 3.0
    onward, the dtype of a string column no longer equals ``object``.
    """
    monkeypatch.setattr(pd.options.future, "infer_string", True)
    (annotated_variable,) = infer_variables_from_dataframe(
        pd.DataFrame({"symbol": ["a", "b", "c"]})
    )
    assert annotated_variable.variable.name == "symbol"
    assert isinstance(annotated_variable.variable, Symbolic)


# %% joint probability trees


@pytest.fixture
def tree(mixed_data) -> JointProbabilityTree:
    """
    An unfitted tree over the columns of :func:`mixed_data`.
    """
    return JointProbabilityTree(
        annotated_variables=infer_variables_from_dataframe(mixed_data)
    )


def test_leaf_node_has_a_distribution_per_variable(tree, mixed_data):
    leaf_node = tree.create_leaf_node(tree.preprocess_data(mixed_data))
    assert len(leaf_node.subcircuits) == 3
    assert isinstance(leaf_node.subcircuits[2].distribution, IntegerDistribution)
    assert isinstance(leaf_node.subcircuits[1], SumUnit)
    assert isinstance(leaf_node.subcircuits[0].distribution, SymbolicDistribution)
    likelihood = leaf_node.probabilistic_circuit.likelihood(mixed_data.to_numpy())
    assert np.all(likelihood > 0)


def test_fit_without_an_improving_split_is_one_leaf(tree, mixed_data):
    tree.min_impurity_improvement = 1
    tree.fit(mixed_data)
    assert len(tree.root.subcircuits) == 1
    assert tree.root.log_weighted_subcircuits[0][0] == 0.0
    assert len(tree.root.subcircuits[0].subcircuits) == 3


def test_fit_has_at_most_one_leaf_per_minimal_number_of_samples(tree, mixed_data):
    tree.min_samples_per_leaf = 10
    circuit = tree.fit(mixed_data)
    assert len(circuit.root.subcircuits) <= math.floor(
        len(mixed_data) / tree.min_samples_leaf
    )
    assert all(weight > -np.inf for weight, _ in circuit.root.log_weighted_subcircuits)
    assert np.all(circuit.likelihood(mixed_data.to_numpy()) > 0)


def test_preprocessing_agrees_with_the_original_implementation(tree, mixed_data):
    variables = old_infer_from_dataframe(
        mixed_data, scale_numeric_types=False, precision=0.0
    )
    original_tree = OldJPT(
        variables,
        min_samples_leaf=tree.min_samples_leaf,
        min_impurity_improvement=tree.min_impurity_improvement,
    )
    original = old_preprocess_data(original_tree, mixed_data).to_numpy()
    # symbolic columns are not preprocessed in order in the original implementation
    assert np.all(original[:, :-1] == tree.preprocess_data(mixed_data)[:, :-1])


def test_fit_has_as_many_leaves_as_the_original_implementation(tree, mixed_data):
    tree.min_samples_per_leaf = 10
    tree.keep_sample_indices = True
    tree.fit(mixed_data)
    variables = old_infer_from_dataframe(
        mixed_data, scale_numeric_types=False, precision=0.0
    )
    original_tree = OldJPT(
        variables,
        min_samples_leaf=tree.min_samples_leaf,
        min_impurity_improvement=tree.min_impurity_improvement,
    ).learn(mixed_data, keep_samples=True)
    assert len(tree.root.subcircuits) == len(original_tree.leaves)


@pytest.mark.parametrize("column", ["real", "integer", "symbol"])
def test_a_tree_over_one_variable_is_one_leaf(mixed_data, column):
    data = mixed_data[[column]]
    circuit = JointProbabilityTree(infer_variables_from_dataframe(data)).fit(data)
    assert len(circuit.root.subcircuits) == 1


def test_variable_dependencies_to_json(tree):
    names = [variable.name for variable in tree.variables]
    assert tree._variable_dependencies_to_json() == {name: names for name in names}


def test_serialization_round_trip(tree, mixed_data):
    tree.min_samples_per_leaf = 10
    tree.fit(mixed_data)
    assert from_json(to_json(tree)) == tree


def test_leaves_stay_deterministic_when_data_is_denser_than_tolerance():
    """
    Leaves whose data sits closer to a decision boundary than the tolerance of a Nyga
    distribution reserve the data of the neighbouring leaf instead of overlapping it.
    """
    # smaller than twice the default tolerance at the extremes, so widening both sides
    # by the full tolerance would make them overlap
    gap = 2e-7
    data = pd.DataFrame(
        {
            "x": np.concatenate(
                [
                    np.linspace(0.0, 5.0 - gap / 2, 50),
                    np.linspace(5.0 + gap / 2, 10.0, 50),
                ]
            )
        }
    )
    (variable,) = infer_variables_from_dataframe(data)
    model = JointProbabilityTree(annotated_variables=[variable])
    preprocessed = model.preprocess_data(data)
    left_rows, right_rows = preprocessed[:50], preprocessed[50:]

    model.root = SumUnit(probabilistic_circuit=model.probabilistic_circuit)
    model.root.add_subcircuit(model.create_leaf_node(left_rows, right_rows), 0.0)
    model.root.add_subcircuit(model.create_leaf_node(right_rows, left_rows), 0.0)
    assert model.probabilistic_circuit.is_deterministic()


def test_dependent_continuous_variables_are_split_apart():
    """
    Small *x* always comes with large *y* and vice versa, so a single leaf, which models
    them as independent, is not a fit of this data.
    """
    data = pd.DataFrame(
        {
            "x": np.concatenate([np.linspace(0, 1, 50), np.linspace(10, 11, 50)]),
            "y": np.concatenate([np.linspace(5, 6, 50), np.linspace(-5, -4, 50)]),
        }
    )
    circuit = JointProbabilityTree(min_samples_per_leaf=10).fit(data)
    assert len(circuit.root.subcircuits) > 1


def test_splitting_stops_once_every_leaf_is_precise_enough():
    """
    A maximum standard deviation is a precision to reach: once every numeric target in a
    node is at least that precise, the node is not split any further.
    """
    generator = np.random.default_rng(0)
    x = generator.uniform(0, 100, 500)
    data = pd.DataFrame({"x": x, "y": x + generator.normal(0, 1, 500)})
    variables = infer_variables_from_dataframe(data)
    for variable in variables:
        variable.max_standard_deviation = 5

    circuit = JointProbabilityTree(
        min_samples_per_leaf=20, keep_sample_indices=True
    ).fit(data, variables)
    unlimited = JointProbabilityTree(min_samples_per_leaf=20).fit(data)

    for leaf_node in circuit.root.subcircuits:
        assert data.iloc[leaf_node.sample_indices].std().max() <= 5
    assert len(circuit.root.subcircuits) < len(unlimited.root.subcircuits)


# %% a tree of a real dataset


@pytest.fixture(scope="module")
def breast_cancer_data() -> pd.DataFrame:
    """
    The breast cancer dataset of :mod:`sklearn` with its target as a symbolic column.
    """
    dataset = sklearn.datasets.load_breast_cancer(as_frame=True)
    data = dataset.data
    target = dataset.target.astype(str)
    target[target == "1"] = "malignant"
    target[target == "0"] = "friendly"
    data["malignant"] = target
    return data


@pytest.fixture(scope="module")
def breast_cancer_tree(breast_cancer_data) -> JointProbabilityTree:
    """
    A tree fitted to :func:`breast_cancer_data`.
    """
    tree = JointProbabilityTree(
        annotated_variables=infer_variables_from_dataframe(
            breast_cancer_data, min_samples_per_quantile=600
        ),
        min_samples_per_leaf=0.4,
    )
    tree.fit(breast_cancer_data)
    return tree


@pytest.fixture
def whole_domain(breast_cancer_tree) -> Event:
    """
    The event that holds every value of every variable of the tree.
    """
    return SimpleEvent.from_data(
        {variable: variable.domain for variable in breast_cancer_tree.variables}
    ).as_composite_set()


def test_tree_survives_a_round_trip_through_a_json_file(breast_cancer_tree, tmp_path):
    file = tmp_path / "tree.json"
    file.write_text(json.dumps(to_json(breast_cancer_tree)))
    assert from_json(json.loads(file.read_text())) == breast_cancer_tree


def test_circuit_of_a_tree_survives_serialization(breast_cancer_tree, whole_domain):
    circuit = from_json(to_json(breast_cancer_tree.probabilistic_circuit))
    assert circuit.probability(whole_domain) == pytest.approx(1.0)


def test_truncation_to_the_whole_domain_keeps_all_mass(
    breast_cancer_tree, whole_domain
):
    truncation = breast_cancer_tree.probabilistic_circuit.truncated(whole_domain)
    assert truncation.probability == pytest.approx(1.0, abs=1e-5)
    assert truncation.distribution.probability(whole_domain) == pytest.approx(
        1.0, abs=1e-5
    )


def test_truncation_to_an_interval_holds_all_mass_inside_it(breast_cancer_tree):
    first = breast_cancer_tree.variables[0]
    event = SimpleEvent.from_data({first: closed(0, 10)}).as_composite_set()
    truncated = breast_cancer_tree.probabilistic_circuit.truncated(event).distribution
    event.fill_missing_variables(truncated.variables)
    assert truncated.probability(event) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "variable_of",
    [
        lambda variables: variables[0],
        lambda variables: next(
            variable for variable in variables if variable.name == "malignant"
        ),
    ],
    ids=["continuous", "symbolic"],
)
def test_univariate_marginal(breast_cancer_tree, variable_of):
    variable = variable_of(breast_cancer_tree.variables)
    marginal = breast_cancer_tree.probabilistic_circuit.marginal([variable])
    assert list(marginal.variables) == [variable]
    assert isinstance(marginal.root, SumUnit)


def test_mode_of_a_tree(breast_cancer_tree):
    mode = breast_cancer_tree.probabilistic_circuit.log_mode(
        check_determinism=False
    ).event
    assert len(mode.simple_sets) > 0


# %% plots of two dimensional circuits


@pytest.fixture
def product_of_gaussians(x, y) -> ProbabilisticCircuit:
    """
    The product of two Gaussians with location two and scale four.
    """
    circuit = ProbabilisticCircuit()
    product = ProductUnit(probabilistic_circuit=circuit)
    for variable in (x, y):
        product.add_subcircuit(
            leaf(GaussianDistribution(variable=variable, location=2, scale=4), circuit)
        )
    return circuit


@pytest.fixture
def tree_of_gaussian_samples(product_of_gaussians) -> ProbabilisticCircuit:
    """
    A tree fitted to samples of :func:`product_of_gaussians`.
    """
    data = pd.DataFrame(
        product_of_gaussians.sample(1000),
        columns=[variable.name for variable in product_of_gaussians.variables],
    )
    variables = infer_variables_from_dataframe(data, min_samples_per_quantile=20)
    return JointProbabilityTree(variables, min_samples_per_leaf=0.9).fit(data)


@pytest.mark.parametrize(
    "circuit_fixture", ["product_of_gaussians", "tree_of_gaussian_samples"]
)
def test_surface_plot_of_a_two_dimensional_circuit_has_traces(circuit_fixture, request):
    circuit = request.getfixturevalue(circuit_fixture)
    figure = go.Figure(circuit.plot(500, surface=True), circuit.plotly_layout())
    assert len(figure.data) > 0
