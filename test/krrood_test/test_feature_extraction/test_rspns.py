from unittest.mock import patch

import numpy as np
import pytest
from sortedcontainers import SortedSet

from krrood.entity_query_language.factories import a, an
from krrood.ormatic.data_access_objects.helper import to_dao
from probabilistic_model.distributions.distributions import IntegerDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    CircuitNotFittedError,
    InvalidMonteCarloSampleCountError,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    GroundingMode,
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.utils import MissingDict
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous, Integer
from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneObjectType,
    SceneRoom,
)


@pytest.fixture
def scenario():
    objects = [
        SceneObject(type=SceneObjectType.TABLE),
        SceneObject(type=SceneObjectType.CHAIR),
        SceneObject(type=SceneObjectType.CHAIR),
        SceneObject(type=SceneObjectType.CHAIR),
    ]
    room = SceneRoom(
        position=KRROODPosition(x=2.0, y=1.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=objects[:3],
    )
    room2 = SceneRoom(
        position=KRROODPosition(x=4.0, y=3.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=objects,
    )
    return to_dao(room), to_dao(room2)


@pytest.fixture
def rpc(scenario):
    room_dao, room2_dao = scenario
    model = RelationalProbabilisticCircuit(SceneRoom)
    model.fit([room_dao, room2_dao])
    return model


@pytest.fixture
def room_query_4():
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    query.resolve()
    return query


def test_ground_before_fit_raises(room_query_4):
    model = RelationalProbabilisticCircuit(SceneRoom)
    with pytest.raises(CircuitNotFittedError):
        model.ground(room_query_4)


def test_fit_class_circuit_is_valid(rpc):
    assert rpc.class_probabilistic_circuit is not None
    assert rpc.class_probabilistic_circuit.is_valid()


def test_fit_class_circuit_has_room_scalar_variables(rpc):
    names = {v.name for v in rpc.class_probabilistic_circuit.variables}
    assert "SceneRoom.position.x" in names
    assert "SceneRoom.position.y" in names
    assert "SceneRoom.position.z" in names
    assert "SceneRoom.orientation.x" in names
    assert "SceneRoom.orientation.y" in names
    assert "SceneRoom.orientation.z" in names
    assert "SceneRoom.orientation.w" in names


def test_fit_class_circuit_has_aggregation_variable(rpc):
    names = {v.name for v in rpc.class_probabilistic_circuit.variables}
    assert "SceneRoomAggregations.total_count()" in names


def test_fit_creates_exchangeable_template_for_objects(rpc):
    assert "objects" in rpc.exchangeable_distribution_templates
    template = rpc.exchangeable_distribution_templates["objects"]
    assert template.template_distribution.class_probabilistic_circuit is not None


def test_fit_exchangeable_template_latent_is_total_count(rpc):
    template = rpc.exchangeable_distribution_templates["objects"]
    latent_names = {v.name for v in template.latent_variables}
    assert "SceneRoomAggregations.total_count()" in latent_names


def test_fit_exchangeable_template_models_object_type(rpc):
    template = rpc.exchangeable_distribution_templates["objects"]
    pc = template.template_distribution.class_probabilistic_circuit
    names = {v.name for v in pc.variables}
    assert "type" in names


def test_ground_circuit_is_valid(rpc, room_query_4):
    model = rpc.ground(room_query_4)
    assert model.is_valid()


def test_ground_has_per_object_type_variables(rpc, room_query_4):
    model = rpc.ground(room_query_4)
    names = {v.name for v in model.variables}
    for i in range(4):
        assert f"SceneRoom.objects[{i}].type" in names


def test_ground_preserves_room_scalar_variables(rpc, room_query_4):
    model = rpc.ground(room_query_4)
    names = {v.name for v in model.variables}
    assert "SceneRoom.position.x" in names
    assert "SceneRoom.orientation.w" in names


def test_ground_integrates_out_unavailable_aggregates(rpc, room_query_4):
    """
    ``chair_count`` and ``table_count`` cannot be determined from the underspecified
    query, so the Monte-Carlo path must integrate them out: they must not survive as
    variables, while the object-type variables remain.
    """
    model = rpc.ground(room_query_4)
    names = {v.name for v in model.variables}
    assert "SceneRoomAggregations.chair_count()" not in names
    assert "SceneRoomAggregations.table_count()" not in names
    for i in range(4):
        assert f"SceneRoom.objects[{i}].type" in names


def test_ground_with_unavailable_aggregate_is_valid(rpc, room_query_4):
    np.random.seed(0)
    assert rpc.ground(room_query_4).is_valid()


def test_non_positive_sample_count_raises_when_integration_needed(rpc, room_query_4):
    """
    Monte-Carlo integration cannot be disabled: a non-positive sample count is rejected
    when undetermined aggregates must be integrated out.
    """
    rpc.monte_carlo_sample_count = 0
    with pytest.raises(InvalidMonteCarloSampleCountError):
        rpc.ground(room_query_4)


def test_monte_carlo_sample_count_controls_mixture_size(rpc, room_query_4):
    """
    Drawing more samples discovers more distinct aggregate values, each adding an
    exchangeable-distribution instance (and its sum units) to the mixture.
    """
    np.random.seed(0)
    rpc.monte_carlo_sample_count = 1
    single = sum(1 for n in rpc.ground(room_query_4).nodes() if isinstance(n, SumUnit))
    np.random.seed(0)
    rpc.monte_carlo_sample_count = 50
    many = sum(1 for n in rpc.ground(room_query_4).nodes() if isinstance(n, SumUnit))
    assert many > single


def test_ground_variable_count_scales_with_query_size(rpc):
    query_2 = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(2)],
    )
    query_2.resolve()
    query_4 = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    query_4.resolve()
    assert len(rpc.ground(query_4).variables) > len(rpc.ground(query_2).variables)


# %% GroundingMode.CAUSAL_SAMPLED retains undetermined latents instead of discarding them


def test_predictive_grounding_is_unaffected_by_the_explicit_default(rpc, room_query_4):
    np.random.seed(0)
    implicit_default = {v.name for v in rpc.ground(room_query_4).variables}
    np.random.seed(0)
    explicit_predictive = {
        v.name
        for v in rpc.ground(
            room_query_4, grounding_mode=GroundingMode.PREDICTIVE
        ).variables
    }
    assert implicit_default == explicit_predictive


def test_causal_sampled_grounding_retains_undetermined_latents_as_variables(
    rpc, room_query_4
):
    np.random.seed(0)
    model = rpc.ground(room_query_4, grounding_mode=GroundingMode.CAUSAL_SAMPLED)
    names = {v.name for v in model.variables}
    assert "SceneRoomAggregations.chair_count()" in names
    assert "SceneRoomAggregations.table_count()" in names


def test_causal_sampled_grounding_preserves_object_type_variables(rpc, room_query_4):
    np.random.seed(0)
    model = rpc.ground(room_query_4, grounding_mode=GroundingMode.CAUSAL_SAMPLED)
    names = {v.name for v in model.variables}
    for i in range(4):
        assert f"SceneRoom.objects[{i}].type" in names


def test_causal_sampled_grounding_is_valid(rpc, room_query_4):
    np.random.seed(0)
    model = rpc.ground(room_query_4, grounding_mode=GroundingMode.CAUSAL_SAMPLED)
    assert model.is_valid()


def test_causal_sampled_grounding_supports_causal_circuit_registration(
    rpc, room_query_4
):
    """
    The whole point of ``GroundingMode.CAUSAL_SAMPLED``: a latent that predictive
    grounding would have discarded must be usable as a registered cause in a
    ``CausalCircuit`` -- i.e. verified support-deterministic against it, the structural
    precondition ``backdoor_adjustment`` relies on.
    """
    np.random.seed(0)
    model = rpc.ground(room_query_4, grounding_mode=GroundingMode.CAUSAL_SAMPLED)
    chair_count_variable = next(
        v for v in model.variables if v.name == "SceneRoomAggregations.chair_count()"
    )
    object_type_variable = next(
        v for v in model.variables if v.name == "SceneRoom.objects[0].type"
    )

    tree = MarginalDeterminismTreeNode.from_causal_graph(
        [chair_count_variable], [object_type_variable]
    )
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        model, tree, [chair_count_variable], [object_type_variable]
    )

    result = causal_circuit.verify_support_determinism()
    assert result.passed


def test_causal_sampled_grounding_backdoor_adjustment_runs(rpc, room_query_4):
    """
    End-to-end regression test: computing ``P(effect | do(cause))`` on a
    ``CAUSAL_SAMPLED``-grounded circuit must not raise.

    This exercises every renamed exchangeable-instance leaf, including any query part
    whose grounded circuit happens to collapse to a single leaf as its own root.
    """
    np.random.seed(0)
    model = rpc.ground(room_query_4, grounding_mode=GroundingMode.CAUSAL_SAMPLED)
    chair_count_variable = next(
        v for v in model.variables if v.name == "SceneRoomAggregations.chair_count()"
    )
    object_type_variable = next(
        v for v in model.variables if v.name == "SceneRoom.objects[0].type"
    )

    tree = MarginalDeterminismTreeNode.from_causal_graph(
        [chair_count_variable], [object_type_variable]
    )
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        model, tree, [chair_count_variable], [object_type_variable]
    )

    interventional_circuit = causal_circuit.backdoor_adjustment(
        cause_variable=chair_count_variable, effect_variable=object_type_variable
    )
    assert interventional_circuit.is_valid()


# %% GroundingMode.CAUSAL_EXACT retains undetermined latents via exact partition


def test_causal_exact_grounding_retains_undetermined_latents_as_variables(
    rpc, room_query_4
):
    model = rpc.ground(room_query_4, grounding_mode=GroundingMode.CAUSAL_EXACT)
    names = {v.name for v in model.variables}
    assert "SceneRoomAggregations.chair_count()" in names
    assert "SceneRoomAggregations.table_count()" in names


def test_causal_exact_grounding_is_valid(rpc, room_query_4):
    model = rpc.ground(room_query_4, grounding_mode=GroundingMode.CAUSAL_EXACT)
    assert model.is_valid()


def test_causal_exact_grounding_is_reproducible_across_calls(rpc, room_query_4):
    """
    ``CAUSAL_EXACT`` grounding -- whether it takes its own exact-partition path or falls
    back to ``CAUSAL_SAMPLED`` -- must ground the identical set of variables across
    calls, even under different random state.
    """
    np.random.seed(0)
    first = {
        v.name
        for v in rpc.ground(
            room_query_4, grounding_mode=GroundingMode.CAUSAL_EXACT
        ).variables
    }
    np.random.seed(123)
    second = {
        v.name
        for v in rpc.ground(
            room_query_4, grounding_mode=GroundingMode.CAUSAL_EXACT
        ).variables
    }
    assert first == second


def test_causal_exact_grounding_supports_causal_circuit_registration(rpc, room_query_4):
    model = rpc.ground(room_query_4, grounding_mode=GroundingMode.CAUSAL_EXACT)
    chair_count_variable = next(
        v for v in model.variables if v.name == "SceneRoomAggregations.chair_count()"
    )
    object_type_variable = next(
        v for v in model.variables if v.name == "SceneRoom.objects[0].type"
    )

    tree = MarginalDeterminismTreeNode.from_causal_graph(
        [chair_count_variable], [object_type_variable]
    )
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        model, tree, [chair_count_variable], [object_type_variable]
    )

    result = causal_circuit.verify_support_determinism()
    assert result.passed


def test_causal_exact_grounding_backdoor_adjustment_runs(rpc, room_query_4):
    model = rpc.ground(room_query_4, grounding_mode=GroundingMode.CAUSAL_EXACT)
    chair_count_variable = next(
        v for v in model.variables if v.name == "SceneRoomAggregations.chair_count()"
    )
    object_type_variable = next(
        v for v in model.variables if v.name == "SceneRoom.objects[0].type"
    )

    tree = MarginalDeterminismTreeNode.from_causal_graph(
        [chair_count_variable], [object_type_variable]
    )
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        model, tree, [chair_count_variable], [object_type_variable]
    )

    interventional_circuit = causal_circuit.backdoor_adjustment(
        cause_variable=chair_count_variable, effect_variable=object_type_variable
    )
    assert interventional_circuit.is_valid()


def test_causal_exact_grounding_falls_back_to_causal_sampled_when_partition_overlaps(
    rpc, room_query_4, caplog
):
    """
    When the fitted circuit's partition over the undetermined latents is not disjoint,
    ``CAUSAL_EXACT`` must fall back to ``CAUSAL_SAMPLED`` rather than raise or produce
    an unsound circuit.
    """
    np.random.seed(0)
    with patch.object(
        RelationalProbabilisticCircuit,
        "_undetermined_latents_partition_disjointly",
        return_value=False,
    ):
        with caplog.at_level("WARNING"):
            model = rpc.ground(room_query_4, grounding_mode=GroundingMode.CAUSAL_EXACT)

    assert model.is_valid()
    names = {v.name for v in model.variables}
    assert "SceneRoomAggregations.chair_count()" in names
    assert any("falling back" in message.lower() for message in caplog.messages)


# %% GroundingMode.CAUSAL_EXACT preserves the actual correlation between the retained
# latent and the exchangeable relation's own attributes, not just its variable set


def _room_with_chair_count(rng: np.random.Generator, chair_count: int) -> SceneRoom:
    """
    A three-object room whose first object's type is CHAIR whenever chair_count is at
    least 2, TABLE otherwise, and whose remaining objects are padded to match
    chair_count exactly.
    """
    first_type = SceneObjectType.CHAIR if chair_count >= 2 else SceneObjectType.TABLE
    remaining_chairs = max(
        chair_count - (1 if first_type == SceneObjectType.CHAIR else 0), 0
    )
    remaining_types = [SceneObjectType.CHAIR] * remaining_chairs
    while len(remaining_types) < 2:
        remaining_types.append(SceneObjectType.TABLE)
    objects = [SceneObject(type=first_type)] + [
        SceneObject(type=object_type) for object_type in remaining_types[:2]
    ]
    return SceneRoom(
        position=KRROODPosition(
            x=float(rng.uniform(0, 5)), y=float(rng.uniform(0, 5)), z=0.0
        ),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=objects,
    )


@pytest.fixture
def correlated_rpc() -> RelationalProbabilisticCircuit:
    rng = np.random.default_rng(0)
    rooms = [_room_with_chair_count(rng, 1) for _ in range(20)] + [
        _room_with_chair_count(rng, 3) for _ in range(20)
    ]
    model = RelationalProbabilisticCircuit(SceneRoom)
    model.fit([to_dao(room) for room in rooms])
    return model


@pytest.fixture
def correlated_room_query():
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(3)],
    )
    query.resolve()
    return query


def test_causal_exact_grounding_preserves_correlation_with_the_retained_latent(
    correlated_rpc, correlated_room_query
):
    """
    Regression test: the retained chair_count latent must stay statistically tied to the
    object-type distribution it was fitted alongside.

    Before this was fixed, _representative_value passed a whole mode region (not a
    point) into conditioning, which always failed and silently fell back to grounding
    every branch from the same unconditioned distribution -- and even after fixing that,
    a single, undifferentiated partition branch was treated as trivially valid instead
    of triggering a fall back to sampling, discarding the correlation either way.
    P(objects[0].type=CHAIR | do(chair_count=1)) and P(objects[0].type=CHAIR |
    do(chair_count=3)) must therefore differ, reflecting chair_count=1 rooms never
    having their first object be a chair and chair_count=3 rooms always having it be
    one.
    """
    np.random.seed(0)
    grounded = correlated_rpc.ground(
        correlated_room_query, grounding_mode=GroundingMode.CAUSAL_EXACT
    )
    chair_count_variable = next(
        v for v in grounded.variables if v.name == "SceneRoomAggregations.chair_count()"
    )
    object_type_variable = next(
        v for v in grounded.variables if v.name == "SceneRoom.objects[0].type"
    )

    tree = MarginalDeterminismTreeNode.from_causal_graph(
        [chair_count_variable], [object_type_variable]
    )
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        grounded, tree, [chair_count_variable], [object_type_variable]
    )
    interventional_circuit = causal_circuit.backdoor_adjustment(
        cause_variable=chair_count_variable, effect_variable=object_type_variable
    )

    def probability_of_chair_given_chair_count(chair_count: int) -> float:
        cause_event = (
            SimpleEvent.from_data({chair_count_variable: chair_count})
            .as_composite_set()
            .fill_missing_variables_pure(interventional_circuit.variables)
        )
        chair_event = (
            SimpleEvent.from_data({object_type_variable: SceneObjectType.CHAIR})
            .as_composite_set()
            .fill_missing_variables_pure(interventional_circuit.variables)
        )
        cause_probability = interventional_circuit.probability(cause_event)
        assert cause_probability > 0
        return (
            interventional_circuit.probability(cause_event & chair_event)
            / cause_probability
        )

    assert probability_of_chair_given_chair_count(
        1
    ) < probability_of_chair_given_chair_count(3)


def test_representative_value_returns_a_point_not_a_region():
    """
    Regression test: _representative_value must collapse each leaf's mode to a single
    point.

    Passing the mode region itself into conditioning always fails silently (see
    test_causal_exact_grounding_preserves_correlation_with_the_retained_latent).
    """
    variable = Integer("value")
    circuit = ProbabilisticCircuit()
    branch = _integer_leaf(variable, {2: 0.5, 3: 0.5}, circuit)

    representative_value = RelationalProbabilisticCircuit._representative_value(
        branch, SortedSet([variable])
    )

    assert representative_value == {variable: 2.0}
    conditioning_result, log_likelihood = branch.distribution.log_conditional(
        representative_value
    )
    assert conditioning_result is not None
    assert log_likelihood > -np.inf


def test_node_local_branch_log_probabilities_reflect_each_nodes_own_correlation():
    """
    Regression test: two mounting nodes that each correlate the undetermined latent with
    a different other variable must get different weights over the same global partition
    branches, not one weighting shared across every node.
    """
    other_variable = Continuous("other_variable")
    chair_count = Integer("chair_count")

    circuit = ProbabilisticCircuit()
    node_favoring_one = ProductUnit(probabilistic_circuit=circuit)
    node_favoring_one.add_subcircuit(
        leaf(
            UniformDistribution(
                variable=other_variable, interval=closed(0, 1).simple_sets[0]
            ),
            circuit,
        )
    )
    node_favoring_one.add_subcircuit(_integer_leaf(chair_count, {1: 1.0}, circuit))

    node_favoring_three = ProductUnit(probabilistic_circuit=circuit)
    node_favoring_three.add_subcircuit(
        leaf(
            UniformDistribution(
                variable=other_variable, interval=closed(2, 3).simple_sets[0]
            ),
            circuit,
        )
    )
    node_favoring_three.add_subcircuit(_integer_leaf(chair_count, {3: 1.0}, circuit))

    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(node_favoring_one, 0.0)
    root.add_subcircuit(node_favoring_three, 0.0)
    root.normalize()

    region_one = SimpleEvent.from_data({chair_count: 1}).as_composite_set()
    region_three = SimpleEvent.from_data({chair_count: 3}).as_composite_set()

    weights_for_node_favoring_one = (
        RelationalProbabilisticCircuit._node_local_branch_log_probabilities(
            node_favoring_one, SortedSet([chair_count]), [region_one, region_three]
        )
    )
    weights_for_node_favoring_three = (
        RelationalProbabilisticCircuit._node_local_branch_log_probabilities(
            node_favoring_three, SortedSet([chair_count]), [region_one, region_three]
        )
    )

    assert weights_for_node_favoring_one[0] > weights_for_node_favoring_one[1]
    assert weights_for_node_favoring_three[1] > weights_for_node_favoring_three[0]


# %% RelationalProbabilisticCircuit._undetermined_latents_partition_disjointly


def _integer_leaf(variable, probabilities, circuit):
    return leaf(
        IntegerDistribution(
            variable=variable, probabilities=MissingDict(float, probabilities)
        ),
        circuit,
    )


def test_partition_disjointly_false_for_a_single_branch():
    """
    A single, undifferentiated branch fails the precondition rather than trivially
    passing it: the fitted circuit never actually split on this latent, so every
    exchangeable instance would be grounded from the same representative point
    regardless of which value the latent takes, discarding the correlation between
    them.
    """
    variable = Integer("value")
    circuit = ProbabilisticCircuit()
    _integer_leaf(variable, {1: 1.0}, circuit)
    assert (
        not RelationalProbabilisticCircuit._undetermined_latents_partition_disjointly(
            circuit
        )
    )


def test_partition_disjointly_true_for_disjoint_branches():
    variable = Integer("value")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(_integer_leaf(variable, {1: 1.0}, circuit), 0.0)
    root.add_subcircuit(_integer_leaf(variable, {2: 1.0}, circuit), 0.0)
    root.normalize()
    assert RelationalProbabilisticCircuit._undetermined_latents_partition_disjointly(
        circuit
    )


def test_partition_disjointly_false_for_overlapping_branches():
    variable = Integer("value")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(_integer_leaf(variable, {1: 0.5, 2: 0.5}, circuit), 0.0)
    root.add_subcircuit(_integer_leaf(variable, {2: 0.5, 3: 0.5}, circuit), 0.0)
    root.normalize()
    assert (
        not RelationalProbabilisticCircuit._undetermined_latents_partition_disjointly(
            circuit
        )
    )
