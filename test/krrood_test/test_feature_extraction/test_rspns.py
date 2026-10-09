import functools
import json
import operator
import random
from unittest.mock import patch

import numpy as np
import pytest
from sortedcontainers import SortedSet
from typing_extensions import Any

from krrood.adapters.json_serializer import from_json, to_json
from krrood.entity_query_language.factories import a, an
from probabilistic_model.distributions.distributions import IntegerDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    CircuitNotFittedError,
    InvalidMonteCarloSampleCountError,
    MixedCircuitTypesError,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.probabilistic_circuit.relational.exchangeable_grounding import (
    ExchangeablePartGrounder,
    GroundingMode,
    InstanceMixture,
    WeightedAssignments,
)
from probabilistic_model.probabilistic_circuit.relational.layered_grounding import (
    LayeredGrounding,
)
from probabilistic_model.probabilistic_circuit.relational.rustworkx_grounding import (
    RustworkxExchangeablePartGrounder,
    RustworkxGrounding,
)
from probabilistic_model.learning.learning_method import LayeredLearning
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.utils import MissingDict
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous, Integer, Symbolic
from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneObjectType,
    SceneRoom,
    TestExParts,
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
    return room, room2


@pytest.fixture
def relational_probabilistic_circuit(scenario):
    room, room2 = scenario
    model = RelationalProbabilisticCircuit(SceneRoom)
    model.fit([room, room2])
    return model


@pytest.fixture
def layered_relational_probabilistic_circuit(scenario):
    return RelationalProbabilisticCircuit(
        SceneRoom, learning_method=LayeredLearning(JointProbabilityTree())
    ).fit(list(scenario))


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


def test_fit_class_circuit_is_valid(relational_probabilistic_circuit):
    assert relational_probabilistic_circuit.class_probabilistic_circuit is not None
    assert relational_probabilistic_circuit.class_probabilistic_circuit.is_valid()


def test_fit_class_circuit_has_room_scalar_variables(relational_probabilistic_circuit):
    names = {
        variable.name
        for variable in relational_probabilistic_circuit.class_probabilistic_circuit.variables
    }
    assert "SceneRoom.position.x" in names
    assert "SceneRoom.position.y" in names
    assert "SceneRoom.position.z" in names
    assert "SceneRoom.orientation.x" in names
    assert "SceneRoom.orientation.y" in names
    assert "SceneRoom.orientation.z" in names
    assert "SceneRoom.orientation.w" in names


def test_fit_class_circuit_has_aggregation_variable(relational_probabilistic_circuit):
    names = {
        variable.name
        for variable in relational_probabilistic_circuit.class_probabilistic_circuit.variables
    }
    assert "SceneRoomAggregations.total_count()" in names


def test_fit_creates_exchangeable_template_for_objects(
    relational_probabilistic_circuit,
):
    assert (
        "objects"
        in relational_probabilistic_circuit.exchangeable_distribution_templates
    )
    template = relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]
    assert template.template_distribution.class_probabilistic_circuit is not None


def test_fit_exchangeable_template_latent_is_total_count(
    relational_probabilistic_circuit,
):
    template = relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]
    latent_names = {variable.name for variable in template.latent_variables}
    assert "SceneRoomAggregations.total_count()" in latent_names


def test_fit_exchangeable_template_models_object_type(relational_probabilistic_circuit):
    template = relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]
    probabilistic_circuit = template.template_distribution.class_probabilistic_circuit
    names = {variable.name for variable in probabilistic_circuit.variables}
    assert "type" in names


def test_ground_circuit_is_valid(relational_probabilistic_circuit, room_query_4):
    model = relational_probabilistic_circuit.ground(room_query_4)
    assert model.is_valid()


def test_ground_has_per_object_type_variables(
    relational_probabilistic_circuit, room_query_4
):
    model = relational_probabilistic_circuit.ground(room_query_4)
    names = {variable.name for variable in model.variables}
    for i in range(4):
        assert f"SceneRoom.objects[{i}].type" in names


def test_ground_preserves_room_scalar_variables(
    relational_probabilistic_circuit, room_query_4
):
    model = relational_probabilistic_circuit.ground(room_query_4)
    names = {variable.name for variable in model.variables}
    assert "SceneRoom.position.x" in names
    assert "SceneRoom.orientation.w" in names


def test_ground_integrates_out_unavailable_aggregates(
    relational_probabilistic_circuit, room_query_4
):
    """
    ``chair_count`` and ``table_count`` cannot be determined from the underspecified
    query, so the Monte-Carlo path must retain them as variables (grounding never
    integrates undetermined latents out), alongside the object-type variables.
    """
    model = relational_probabilistic_circuit.ground(room_query_4)
    names = {variable.name for variable in model.variables}
    assert "SceneRoomAggregations.chair_count()" in names
    assert "SceneRoomAggregations.table_count()" in names
    for i in range(4):
        assert f"SceneRoom.objects[{i}].type" in names


def test_ground_with_unavailable_aggregate_is_valid(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    assert relational_probabilistic_circuit.ground(room_query_4).is_valid()


def test_non_positive_sample_count_raises_when_integration_needed(
    relational_probabilistic_circuit, room_query_4
):
    """
    Monte-Carlo integration cannot be disabled: a non-positive sample count is rejected
    when undetermined aggregates must be integrated out.
    """
    relational_probabilistic_circuit.monte_carlo_sample_count = 0
    with pytest.raises(InvalidMonteCarloSampleCountError):
        relational_probabilistic_circuit.ground(room_query_4)


def _rooms_with_ambiguous_total_count_4() -> list[SceneRoom]:
    three_chairs_one_table = SceneRoom(
        position=KRROODPosition(x=4.0, y=3.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=[
            SceneObject(type=SceneObjectType.TABLE),
            SceneObject(type=SceneObjectType.CHAIR),
            SceneObject(type=SceneObjectType.CHAIR),
            SceneObject(type=SceneObjectType.CHAIR),
        ],
    )
    two_chairs_two_tables = SceneRoom(
        position=KRROODPosition(x=5.0, y=2.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=[
            SceneObject(type=SceneObjectType.TABLE),
            SceneObject(type=SceneObjectType.TABLE),
            SceneObject(type=SceneObjectType.CHAIR),
            SceneObject(type=SceneObjectType.CHAIR),
        ],
    )
    return [three_chairs_one_table, two_chairs_two_tables]


@pytest.fixture
def relational_probabilistic_circuit_with_ambiguous_total_count_4():
    """
    Two rooms share ``total_count() == 4`` but split it differently between chairs and
    tables (3 chairs/1 table vs.

    2 chairs/2 tables), so conditioning on 4 objects still
    leaves genuine ambiguity between two distinct ``(chair_count, table_count)``
    aggregate values for :func:`test_monte_carlo_sample_count_controls_mixture_size` to
    discover -- ``relational_probabilistic_circuit``'s own two rooms have distinct ``total_count()`` values (3 and
    4), so conditioning on 4 objects there pins the aggregates down to a single value
    regardless of sample count.

    Both rooms are fitted into one leaf (``min_samples_per_leaf=2``): grounding draws a
    leaf's own samples whenever the shared ones miss its values, so an ambiguity has to
    sit within one leaf for the sample count to decide how much of it is discovered.
    """
    model = RelationalProbabilisticCircuit(
        SceneRoom, learning_method=JointProbabilityTree(min_samples_per_leaf=2)
    )
    model.fit(_rooms_with_ambiguous_total_count_4())
    return model


@pytest.fixture
def layered_relational_probabilistic_circuit_with_ambiguous_total_count_4():
    return RelationalProbabilisticCircuit(
        SceneRoom,
        learning_method=LayeredLearning(JointProbabilityTree(min_samples_per_leaf=2)),
    ).fit(_rooms_with_ambiguous_total_count_4())


def test_monte_carlo_sample_count_controls_mixture_size(
    relational_probabilistic_circuit_with_ambiguous_total_count_4, room_query_4
):
    """
    Drawing more samples discovers more distinct aggregate values, each adding an
    exchangeable-distribution instance (and its sum units) to the mixture.
    """
    np.random.seed(0)
    relational_probabilistic_circuit_with_ambiguous_total_count_4.monte_carlo_sample_count = (
        1
    )
    single = len(
        relational_probabilistic_circuit_with_ambiguous_total_count_4.ground(
            room_query_4
        ).nodes()
    )
    np.random.seed(0)
    relational_probabilistic_circuit_with_ambiguous_total_count_4.monte_carlo_sample_count = (
        50
    )
    many = len(
        relational_probabilistic_circuit_with_ambiguous_total_count_4.ground(
            room_query_4
        ).nodes()
    )
    assert many > single


@pytest.fixture
def deserialized_relational_probabilistic_circuit(relational_probabilistic_circuit):
    """
    The circuit after a round-trip through actual JSON text.

    Going through :func:`json.dumps` and :func:`json.loads` rather than only through the
    intermediate dict is what exposes encoding losses such as integer node keys becoming
    strings.
    """
    return from_json(json.loads(json.dumps(to_json(relational_probabilistic_circuit))))


def test_deserialization_restores_class(deserialized_relational_probabilistic_circuit):
    assert isinstance(
        deserialized_relational_probabilistic_circuit, RelationalProbabilisticCircuit
    )
    assert deserialized_relational_probabilistic_circuit.class_ is SceneRoom


def test_deserialization_restores_class_circuit_variables(
    relational_probabilistic_circuit, deserialized_relational_probabilistic_circuit
):
    original_names = {
        variable.name
        for variable in relational_probabilistic_circuit.class_probabilistic_circuit.variables
    }
    restored_names = {
        variable.name
        for variable in deserialized_relational_probabilistic_circuit.class_probabilistic_circuit.variables
    }
    assert restored_names == original_names


def test_deserialization_restores_exchangeable_templates(
    relational_probabilistic_circuit, deserialized_relational_probabilistic_circuit
):
    assert (
        deserialized_relational_probabilistic_circuit.exchangeable_distribution_templates.keys()
        == relational_probabilistic_circuit.exchangeable_distribution_templates.keys()
    )
    template = deserialized_relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]
    latent_names = {variable.name for variable in template.latent_variables}
    assert latent_names == {
        variable.name
        for variable in relational_probabilistic_circuit.exchangeable_distribution_templates[
            "objects"
        ].latent_variables
    }


def test_deserialized_circuit_grounds_to_the_same_variables(
    relational_probabilistic_circuit,
    deserialized_relational_probabilistic_circuit,
    room_query_4,
):
    np.random.seed(0)
    original = relational_probabilistic_circuit.ground(room_query_4)
    np.random.seed(0)
    restored = deserialized_relational_probabilistic_circuit.ground(room_query_4)
    assert restored.is_valid()
    assert {variable.name for variable in restored.variables} == {
        variable.name for variable in original.variables
    }


def test_deserialized_circuit_preserves_likelihoods(
    relational_probabilistic_circuit, deserialized_relational_probabilistic_circuit
):
    """
    The class distribution itself must be preserved numerically, not only structurally.
    """
    samples = relational_probabilistic_circuit.class_probabilistic_circuit.sample(10)
    assert np.allclose(
        relational_probabilistic_circuit.class_probabilistic_circuit.log_likelihood(
            samples
        ),
        deserialized_relational_probabilistic_circuit.class_probabilistic_circuit.log_likelihood(
            samples
        ),
    )


def test_ground_variable_count_scales_with_query_size(relational_probabilistic_circuit):
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
    assert len(relational_probabilistic_circuit.ground(query_4).variables) > len(
        relational_probabilistic_circuit.ground(query_2).variables
    )


# %% GroundingMode.SAMPLED retains undetermined latents instead of discarding them


def test_sampled_grounding_retains_undetermined_latents_as_variables(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    model = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    names = {variable.name for variable in model.variables}
    assert "SceneRoomAggregations.chair_count()" in names
    assert "SceneRoomAggregations.table_count()" in names


def test_sampled_grounding_preserves_object_type_variables(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    model = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    names = {variable.name for variable in model.variables}
    for i in range(4):
        assert f"SceneRoom.objects[{i}].type" in names


def test_sampled_grounding_is_valid(relational_probabilistic_circuit, room_query_4):
    np.random.seed(0)
    model = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    assert model.is_valid()


def _causal_circuit_for(model):
    """
    Build a CausalCircuit registering ``chair_count()`` as the cause and
    ``objects[0].type`` as the effect over a grounded circuit, the pair every
    GroundingMode.SAMPLED/EXACT causal-registration test below exercises.
    """
    chair_count_variable = next(
        variable
        for variable in model.variables
        if variable.name == "SceneRoomAggregations.chair_count()"
    )
    object_type_variable = next(
        variable
        for variable in model.variables
        if variable.name == "SceneRoom.objects[0].type"
    )
    tree = MarginalDeterminismTreeNode.from_causal_graph(
        [chair_count_variable], [object_type_variable]
    )
    return CausalCircuit.from_probabilistic_circuit(
        model, tree, [chair_count_variable], [object_type_variable]
    )


def test_sampled_grounding_supports_causal_circuit_registration(
    relational_probabilistic_circuit, room_query_4
):
    """
    The whole point of ``GroundingMode.SAMPLED``: a latent that predictive
    grounding would have discarded must be usable as a registered cause in a
    ``CausalCircuit`` -- i.e. verified support-deterministic against it, the structural
    precondition ``backdoor_adjustment`` relies on.
    """
    np.random.seed(0)
    model = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    causal_circuit = _causal_circuit_for(model)

    result = causal_circuit.verify_support_determinism()
    assert result.passed


def test_sampled_grounding_backdoor_adjustment_runs(
    relational_probabilistic_circuit, room_query_4
):
    """
    End-to-end regression test: computing ``P(effect | do(cause))`` on a
    ``SAMPLED``-grounded circuit must not raise.

    This exercises every renamed exchangeable-instance leaf, including any query part
    whose grounded circuit happens to collapse to a single leaf as its own root.
    """
    np.random.seed(0)
    model = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    causal_circuit = _causal_circuit_for(model)

    interventional_circuit = causal_circuit.backdoor_adjustment(
        cause_variable=causal_circuit.causal_variables[0],
        effect_variable=causal_circuit.effect_variables[0],
    )
    assert interventional_circuit.is_valid()


# %% GroundingMode.EXACT retains undetermined latents via exact partition


def test_exact_grounding_retains_undetermined_latents_as_variables(
    relational_probabilistic_circuit, room_query_4
):
    model = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.EXACT
    )
    names = {variable.name for variable in model.variables}
    assert "SceneRoomAggregations.chair_count()" in names
    assert "SceneRoomAggregations.table_count()" in names


def test_exact_grounding_is_valid(relational_probabilistic_circuit, room_query_4):
    model = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.EXACT
    )
    assert model.is_valid()


def test_exact_grounding_is_reproducible_across_calls(
    relational_probabilistic_circuit, room_query_4
):
    """
    ``EXACT`` grounding -- whether it takes its own exact-partition path or falls back
    to ``SAMPLED`` -- must ground the identical set of variables across calls, even
    under different random state.
    """
    np.random.seed(0)
    first = {
        variable.name
        for variable in relational_probabilistic_circuit.ground(
            room_query_4, grounding_mode=GroundingMode.EXACT
        ).variables
    }
    np.random.seed(123)
    second = {
        variable.name
        for variable in relational_probabilistic_circuit.ground(
            room_query_4, grounding_mode=GroundingMode.EXACT
        ).variables
    }
    assert first == second


def test_exact_grounding_supports_causal_circuit_registration(
    relational_probabilistic_circuit, room_query_4
):
    model = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.EXACT
    )
    causal_circuit = _causal_circuit_for(model)

    result = causal_circuit.verify_support_determinism()
    assert result.passed


def test_exact_grounding_backdoor_adjustment_runs(
    relational_probabilistic_circuit, room_query_4
):
    model = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.EXACT
    )
    causal_circuit = _causal_circuit_for(model)

    interventional_circuit = causal_circuit.backdoor_adjustment(
        cause_variable=causal_circuit.causal_variables[0],
        effect_variable=causal_circuit.effect_variables[0],
    )
    assert interventional_circuit.is_valid()


def test_exact_grounding_falls_back_to_sampled_when_partition_overlaps(
    relational_probabilistic_circuit, room_query_4, caplog
):
    """
    When the fitted circuit's partition over the undetermined latents is not disjoint,
    ``EXACT`` must fall back to ``SAMPLED`` rather than raise or produce an unsound
    circuit.
    """
    np.random.seed(0)
    with patch.object(
        ExchangeablePartGrounder,
        "_undetermined_latents_partition_disjointly",
        return_value=False,
    ):
        with caplog.at_level("WARNING"):
            model = relational_probabilistic_circuit.ground(
                room_query_4, grounding_mode=GroundingMode.EXACT
            )

    assert model.is_valid()
    names = {variable.name for variable in model.variables}
    assert "SceneRoomAggregations.chair_count()" in names
    assert any("falling back" in message.lower() for message in caplog.messages)


# %% GroundingMode.EXACT preserves the actual correlation between the retained
# latent and the exchangeable relation's own attributes, not just its variable set


def _room_with_chair_count(
    random_generator: np.random.Generator, chair_count: int
) -> SceneRoom:
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
            x=float(random_generator.uniform(0, 5)),
            y=float(random_generator.uniform(0, 5)),
            z=0.0,
        ),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=objects,
    )


def _correlated_rooms() -> list[SceneRoom]:
    random_generator = np.random.default_rng(0)
    return [_room_with_chair_count(random_generator, 1) for _ in range(20)] + [
        _room_with_chair_count(random_generator, 3) for _ in range(20)
    ]


@pytest.fixture
def correlated_relational_probabilistic_circuit() -> RelationalProbabilisticCircuit:
    model = RelationalProbabilisticCircuit(SceneRoom)
    model.fit(_correlated_rooms())
    return model


@pytest.fixture
def layered_correlated_relational_probabilistic_circuit() -> (
    RelationalProbabilisticCircuit
):
    return RelationalProbabilisticCircuit(
        SceneRoom, learning_method=LayeredLearning(JointProbabilityTree())
    ).fit(_correlated_rooms())


@pytest.fixture
def correlated_room_query():
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(3)],
    )
    query.resolve()
    return query


def test_exact_grounding_preserves_correlation_with_the_retained_latent(
    correlated_relational_probabilistic_circuit, correlated_room_query
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
    grounded = correlated_relational_probabilistic_circuit.ground(
        correlated_room_query, grounding_mode=GroundingMode.EXACT
    )
    chair_count_variable = next(
        variable
        for variable in grounded.variables
        if variable.name == "SceneRoomAggregations.chair_count()"
    )
    object_type_variable = next(
        variable
        for variable in grounded.variables
        if variable.name == "SceneRoom.objects[0].type"
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
    test_exact_grounding_preserves_correlation_with_the_retained_latent).
    """
    variable = Integer("value")
    circuit = ProbabilisticCircuit()
    branch = _integer_leaf(variable, {2: 0.5, 3: 0.5}, circuit)

    representative_value = ExchangeablePartGrounder._representative_value(
        circuit, SortedSet([variable])
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

    weights_for_node_favoring_one = ExchangeablePartGrounder._branch_log_probabilities(
        _circuit_below(node_favoring_one),
        SortedSet([chair_count]),
        [region_one, region_three],
    )
    weights_for_node_favoring_three = (
        ExchangeablePartGrounder._branch_log_probabilities(
            _circuit_below(node_favoring_three),
            SortedSet([chair_count]),
            [region_one, region_three],
        )
    )

    assert weights_for_node_favoring_one[0] > weights_for_node_favoring_one[1]
    assert weights_for_node_favoring_three[1] > weights_for_node_favoring_three[0]


# %% ExchangeablePartGrounder._undetermined_latents_partition_disjointly


def _circuit_below(node: ProductUnit) -> ProbabilisticCircuit:
    result = ProbabilisticCircuit()
    result.mount(node)
    return result


def _partitions_disjointly(circuit: ProbabilisticCircuit) -> bool:
    branches = RustworkxExchangeablePartGrounder.partition_branches(circuit)
    return ExchangeablePartGrounder._undetermined_latents_partition_disjointly(
        [branch.support for branch in branches]
    )


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
    assert not _partitions_disjointly(circuit)


def test_partition_disjointly_true_for_disjoint_branches():
    variable = Integer("value")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(_integer_leaf(variable, {1: 1.0}, circuit), 0.0)
    root.add_subcircuit(_integer_leaf(variable, {2: 1.0}, circuit), 0.0)
    root.normalize()
    assert _partitions_disjointly(circuit)


def test_partition_disjointly_false_for_overlapping_branches():
    variable = Integer("value")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(_integer_leaf(variable, {1: 0.5, 2: 0.5}, circuit), 0.0)
    root.add_subcircuit(_integer_leaf(variable, {2: 0.5, 3: 0.5}, circuit), 0.0)
    root.normalize()
    assert not _partitions_disjointly(circuit)


# %% exchangeable relations of exchangeable relations


def _room(x: float, object_types: list[SceneObjectType]) -> SceneRoom:
    return SceneRoom(
        position=KRROODPosition(x=x, y=1.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=[SceneObject(type=object_type) for object_type in object_types],
    )


def _nested_scenes() -> list[TestExParts]:
    """
    Scenes whose rooms are an exchangeable relation with an exchangeable relation of
    their own, the objects of every room.
    """
    table, chair = SceneObjectType.TABLE, SceneObjectType.CHAIR
    return [
        TestExParts(
            objects=[SceneObject(type=table)],
            rooms=[_room(1.0, [table, chair]), _room(2.0, [chair, chair, chair])],
        ),
        TestExParts(
            objects=[SceneObject(type=chair), SceneObject(type=chair)],
            rooms=[
                _room(3.0, [table]),
                _room(4.0, [table, chair, chair]),
                _room(5.0, [chair]),
            ],
        ),
        TestExParts(
            objects=[SceneObject(type=table), SceneObject(type=chair)],
            rooms=[_room(2.5, [table, table, chair])],
        ),
    ]


@pytest.fixture
def nested_relational_probabilistic_circuit() -> RelationalProbabilisticCircuit:
    return RelationalProbabilisticCircuit(TestExParts).fit(_nested_scenes())


@pytest.fixture
def layered_nested_relational_probabilistic_circuit() -> RelationalProbabilisticCircuit:
    return RelationalProbabilisticCircuit(
        TestExParts, learning_method=LayeredLearning(JointProbabilityTree())
    ).fit(_nested_scenes())


def _room_query(object_types: list) -> Any:
    return a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=object_type) for object_type in object_types],
    )


@pytest.fixture
def nested_query_with_determined_room_statistics():
    """
    A scene whose rooms list the type of every one of their objects, so the query
    determines the aggregation statistics of every room.
    """
    table, chair = SceneObjectType.TABLE, SceneObjectType.CHAIR
    query = a(TestExParts)(
        objects=[a(SceneObject)(type=table), a(SceneObject)(type=chair)],
        rooms=[_room_query([table, chair]), _room_query([chair, chair, table])],
    )
    query.resolve()
    return query


@pytest.fixture
def nested_query_with_undetermined_room_statistics():
    """
    A scene with one room whose object types are left open, so grounding that room
    retains its aggregation statistics.
    """
    query = a(TestExParts)(
        objects=[a(SceneObject)(type=SceneObjectType.TABLE)],
        rooms=[_room_query([SceneObjectType.CHAIR]), _room_query([..., ...])],
    )
    query.resolve()
    return query


@pytest.fixture
def nested_query_with_rooms_of_one_shape():
    """
    A scene whose first two rooms have the same objects in a different order, so their
    queries have the same shape, and whose third room has other objects.
    """
    table, chair = SceneObjectType.TABLE, SceneObjectType.CHAIR
    query = a(TestExParts)(
        objects=[a(SceneObject)(type=table)],
        rooms=[
            _room_query([table, chair]),
            _room_query([chair, table]),
            _room_query([chair, chair, table]),
        ],
    )
    query.resolve()
    return query


@pytest.fixture
def nested_query_with_undetermined_rooms_of_one_shape():
    """
    A scene with two rooms whose object types are left open.
    """
    query = a(TestExParts)(
        objects=[a(SceneObject)(type=SceneObjectType.TABLE)],
        rooms=[_room_query([..., ...]), _room_query([..., ...])],
    )
    query.resolve()
    return query


def test_room_template_models_the_aggregation_statistics_of_its_objects(
    nested_relational_probabilistic_circuit,
):
    room_circuit = (
        nested_relational_probabilistic_circuit.exchangeable_distribution_templates[
            "rooms"
        ].template_distribution
    )
    object_template = room_circuit.exchangeable_distribution_templates["objects"]
    assert set(object_template.latent_variables) <= set(
        room_circuit.class_probabilistic_circuit.variables
    )


def test_ground_nested_relations_is_valid(
    nested_relational_probabilistic_circuit,
    nested_query_with_determined_room_statistics,
):
    grounded = nested_relational_probabilistic_circuit.ground(
        nested_query_with_determined_room_statistics
    )
    assert grounded.is_valid()


def test_ground_keeps_a_relation_whose_successor_has_impossible_statistics(
    nested_relational_probabilistic_circuit,
    nested_query_with_determined_room_statistics,
):
    """
    No scene with a table and a chair as objects has two rooms, so the class circuit
    deems the statistics of the rooms impossible after conditioning on those of the
    objects.

    Grounding the rooms must still keep the grounded objects.
    """
    grounded = nested_relational_probabilistic_circuit.ground(
        nested_query_with_determined_room_statistics
    )
    object_type_names = {
        f"{object_part._variable_}.type"
        for object_part in nested_query_with_determined_room_statistics._kwargs_[
            "objects"
        ]
    }
    assert object_type_names <= {variable.name for variable in grounded.variables}


def test_ground_names_the_objects_of_every_room_by_their_query_path(
    nested_relational_probabilistic_circuit,
    nested_query_with_determined_room_statistics,
):
    grounded = nested_relational_probabilistic_circuit.ground(
        nested_query_with_determined_room_statistics
    )
    object_type_names = {
        f"{object_part._variable_}.type"
        for room_part in nested_query_with_determined_room_statistics._kwargs_["rooms"]
        for object_part in room_part._kwargs_["objects"]
    }
    assert object_type_names <= {variable.name for variable in grounded.variables}


def test_rooms_of_one_shape_share_their_grounding(
    nested_relational_probabilistic_circuit, nested_query_with_rooms_of_one_shape
):
    template = (
        nested_relational_probabilistic_circuit.exchangeable_distribution_templates[
            "rooms"
        ]
    )
    room_parts = nested_query_with_rooms_of_one_shape._kwargs_["rooms"]
    grounded = template.grounded_templates_of_parts(
        room_parts, template.template_distribution.ground
    )
    prefixes = [
        template._prefix_for_part(part, index) for index, part in enumerate(room_parts)
    ]
    assert grounded[0].circuit is grounded[1].circuit
    assert grounded[2].circuit is not grounded[0].circuit
    assert [part.grounded_prefix for part in grounded] == [
        prefixes[0],
        prefixes[0],
        prefixes[2],
    ]
    assert [part.prefix for part in grounded] == prefixes


def test_rooms_of_one_shape_name_their_objects_by_their_own_query_path(
    nested_relational_probabilistic_circuit, nested_query_with_rooms_of_one_shape
):
    grounded = nested_relational_probabilistic_circuit.ground(
        nested_query_with_rooms_of_one_shape
    )
    object_type_names = {
        f"{object_part._variable_}.type"
        for room_part in nested_query_with_rooms_of_one_shape._kwargs_["rooms"]
        for object_part in room_part._kwargs_["objects"]
    }
    assert object_type_names <= {variable.name for variable in grounded.variables}


# %% grounding into a layered circuit


def events_over_discrete_variables(
    variables: SortedSet, number_of_events: int
) -> list[Event]:
    """
    Events that pick a random subset of the values of every symbolic and integer
    variable, and leave every continuous variable free.

    :param variables: The variables of a grounded circuit.
    :param number_of_events: How many events to make.
    :return: The events.
    """
    generator = random.Random(0)
    events = []
    for _ in range(number_of_events):
        assignment = {}
        for variable in variables:
            if isinstance(variable, Symbolic):
                elements = list(variable.domain.simple_sets)
                chosen = generator.sample(elements, generator.randint(1, len(elements)))
                assignment[variable] = functools.reduce(
                    operator.or_, [element.as_composite_set() for element in chosen]
                )
            elif isinstance(variable, Integer):
                lower = generator.randint(0, 4)
                assignment[variable] = closed(lower, lower + generator.randint(0, 3))
            else:
                assignment[variable] = variable.domain
        events.append(SimpleEvent.from_data(assignment).as_composite_set())
    return events


def assert_same_distribution(
    grounded: ProbabilisticCircuit, layered: LayeredProbabilisticCircuit
):
    """
    Assert that a grounding and a layered grounding have the same variables and give
    every event over their discrete variables the same probability.
    """
    assert list(layered.variables) == list(grounded.variables)
    events = events_over_discrete_variables(grounded.variables, 50)
    np.testing.assert_allclose(
        [layered.probability(event) for event in events],
        [grounded.probability(event) for event in events],
        atol=1e-12,
    )


def ground_with_the_same_samples(
    model: RelationalProbabilisticCircuit,
    layered_model: RelationalProbabilisticCircuit,
    query: Any,
    grounding_mode: GroundingMode = GroundingMode.SAMPLED,
) -> tuple[ProbabilisticCircuit, LayeredProbabilisticCircuit]:
    """
    Ground a model and its layered twin, the twin with the samples of the undetermined
    statistics that grounding the model drew, since the two circuit types sample
    differently.
    """
    drawn = []
    sample = ExchangeablePartGrounder._sample_undetermined_latents

    def recording_sample(grounder, node=None):
        drawn.append(sample(grounder, node))
        return drawn[-1]

    with patch.object(
        ExchangeablePartGrounder, "_sample_undetermined_latents", recording_sample
    ):
        grounded = model.ground(query, grounding_mode=grounding_mode)
    replayed = iter(drawn)
    with patch.object(
        ExchangeablePartGrounder,
        "_sample_undetermined_latents",
        lambda grounder, node=None: next(replayed),
    ):
        layered = layered_model.ground(query, grounding_mode=grounding_mode)
    assert next(replayed, None) is None
    return grounded, layered


def test_layered_grounding_with_sampled_latents_is_the_grounding(
    relational_probabilistic_circuit_with_ambiguous_total_count_4,
    layered_relational_probabilistic_circuit_with_ambiguous_total_count_4,
    room_query_4,
):
    assert_same_distribution(
        *ground_with_the_same_samples(
            relational_probabilistic_circuit_with_ambiguous_total_count_4,
            layered_relational_probabilistic_circuit_with_ambiguous_total_count_4,
            room_query_4,
        )
    )


def test_layered_grounding_over_the_exact_partition_is_the_grounding(
    correlated_relational_probabilistic_circuit,
    layered_correlated_relational_probabilistic_circuit,
    correlated_room_query,
):
    grounded = correlated_relational_probabilistic_circuit.ground(
        correlated_room_query, grounding_mode=GroundingMode.EXACT
    )
    layered = layered_correlated_relational_probabilistic_circuit.ground(
        correlated_room_query, grounding_mode=GroundingMode.EXACT
    )
    assert_same_distribution(grounded, layered)


def test_layered_grounding_of_a_query_that_determines_every_statistic_is_the_grounding(
    relational_probabilistic_circuit, layered_relational_probabilistic_circuit
):
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[
            a(SceneObject)(type=SceneObjectType.TABLE),
            a(SceneObject)(type=SceneObjectType.CHAIR),
            a(SceneObject)(type=SceneObjectType.CHAIR),
        ],
    )
    query.resolve()
    grounded = relational_probabilistic_circuit.ground(query)
    layered = layered_relational_probabilistic_circuit.ground(query)
    assert_same_distribution(grounded, layered)


def test_layered_grounding_of_nested_relations_is_the_grounding(
    nested_relational_probabilistic_circuit,
    layered_nested_relational_probabilistic_circuit,
    nested_query_with_determined_room_statistics,
):
    grounded = nested_relational_probabilistic_circuit.ground(
        nested_query_with_determined_room_statistics
    )
    layered = layered_nested_relational_probabilistic_circuit.ground(
        nested_query_with_determined_room_statistics
    )
    assert_same_distribution(grounded, layered)


def test_layered_grounding_of_nested_relations_with_sampled_latents_is_the_grounding(
    nested_relational_probabilistic_circuit,
    layered_nested_relational_probabilistic_circuit,
    nested_query_with_undetermined_room_statistics,
):
    assert_same_distribution(
        *ground_with_the_same_samples(
            nested_relational_probabilistic_circuit,
            layered_nested_relational_probabilistic_circuit,
            nested_query_with_undetermined_room_statistics,
        )
    )


def test_layered_grounding_of_nested_relations_over_the_exact_partition_is_the_grounding(
    nested_relational_probabilistic_circuit,
    layered_nested_relational_probabilistic_circuit,
    nested_query_with_undetermined_room_statistics,
):
    assert_same_distribution(
        *ground_with_the_same_samples(
            nested_relational_probabilistic_circuit,
            layered_nested_relational_probabilistic_circuit,
            nested_query_with_undetermined_room_statistics,
            GroundingMode.EXACT,
        )
    )


def test_layered_grounding_of_rooms_of_one_shape_is_the_grounding(
    nested_relational_probabilistic_circuit,
    layered_nested_relational_probabilistic_circuit,
    nested_query_with_rooms_of_one_shape,
):
    grounded = nested_relational_probabilistic_circuit.ground(
        nested_query_with_rooms_of_one_shape
    )
    layered = layered_nested_relational_probabilistic_circuit.ground(
        nested_query_with_rooms_of_one_shape
    )
    assert_same_distribution(grounded, layered)


def test_layered_grounding_of_undetermined_rooms_of_one_shape_is_the_grounding(
    nested_relational_probabilistic_circuit,
    layered_nested_relational_probabilistic_circuit,
    nested_query_with_undetermined_rooms_of_one_shape,
):
    assert_same_distribution(
        *ground_with_the_same_samples(
            nested_relational_probabilistic_circuit,
            layered_nested_relational_probabilistic_circuit,
            nested_query_with_undetermined_rooms_of_one_shape,
        )
    )


def test_layered_grounding_leaves_out_branches_a_later_relation_rules_out(
    nested_relational_probabilistic_circuit,
    layered_nested_relational_probabilistic_circuit,
):
    """
    Conditioning on the statistics of the rooms removes branches of the class circuit
    that the objects were already mounted at.
    """
    chair, table = SceneObjectType.CHAIR, SceneObjectType.TABLE
    query = a(TestExParts)(
        objects=[a(SceneObject)(type=chair)],
        rooms=[_room_query([chair, chair]), _room_query([table])],
    )
    query.resolve()
    grounded = nested_relational_probabilistic_circuit.ground(query)
    layered = layered_nested_relational_probabilistic_circuit.ground(query)
    assert_same_distribution(grounded, layered)


def test_instance_mixture_gives_every_distinct_assignment_one_instance():
    variable = Integer("value")
    first, second, third = {variable: 1}, {variable: 2}, {variable: 3}
    mixture = InstanceMixture.of_node_local_assignments(
        [
            WeightedAssignments([first, second], [-1.0, -2.0]),
            WeightedAssignments([second, third], [-3.0, -4.0]),
        ],
        SortedSet([variable]),
    )
    assert mixture.assignments == [first, second, third]
    np.testing.assert_array_equal(
        mixture.log_weights, [[-1.0, -2.0, -np.inf], [-np.inf, -3.0, -4.0]]
    )


def test_layered_learning_fits_every_class_circuit_as_a_layered_circuit(
    layered_nested_relational_probabilistic_circuit,
):
    rooms = layered_nested_relational_probabilistic_circuit.exchangeable_distribution_templates[
        "rooms"
    ].template_distribution
    objects_of_rooms = rooms.exchangeable_distribution_templates[
        "objects"
    ].template_distribution
    for model in [
        layered_nested_relational_probabilistic_circuit,
        rooms,
        objects_of_rooms,
    ]:
        assert isinstance(
            model.class_probabilistic_circuit, LayeredProbabilisticCircuit
        )


def test_a_relational_circuit_is_grounded_in_the_circuit_type_it_was_fitted_in(
    nested_relational_probabilistic_circuit,
    layered_nested_relational_probabilistic_circuit,
    nested_query_with_rooms_of_one_shape,
):
    assert isinstance(
        nested_relational_probabilistic_circuit.grounding(), RustworkxGrounding
    )
    assert isinstance(
        layered_nested_relational_probabilistic_circuit.grounding(), LayeredGrounding
    )
    assert isinstance(
        layered_nested_relational_probabilistic_circuit.ground(
            nested_query_with_rooms_of_one_shape
        ),
        LayeredProbabilisticCircuit,
    )


def test_fitting_parts_in_another_circuit_type_raises():
    model = RelationalProbabilisticCircuit(
        TestExParts,
        learning_method=LayeredLearning(JointProbabilityTree()),
        part_learning_methods={"rooms": JointProbabilityTree()},
    )
    with pytest.raises(MixedCircuitTypesError):
        model.fit(_nested_scenes())


def test_nested_relations_are_grounded_in_the_grounding_mode_of_the_parent(
    nested_relational_probabilistic_circuit,
    nested_query_with_undetermined_room_statistics,
):
    """
    The objects of a room whose object types are left open are grounded in the mode the
    scene is grounded in.
    """
    grounding_modes = []
    ground_part = ExchangeablePartGrounder.ground

    def recording_ground(grounder, grounding_mode):
        grounding_modes.append(grounding_mode)
        return ground_part(grounder, grounding_mode)

    with patch.object(ExchangeablePartGrounder, "ground", recording_ground):
        nested_relational_probabilistic_circuit.ground(
            nested_query_with_undetermined_room_statistics,
            grounding_mode=GroundingMode.EXACT,
        )
    assert len(grounding_modes) > 2
    assert set(grounding_modes) == {GroundingMode.EXACT}
