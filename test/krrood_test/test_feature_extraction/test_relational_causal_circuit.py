"""
Tests for :mod:`probabilistic_model.probabilistic_circuit.relational.causal`: the bridge
from ``RelationalProbabilisticCircuit`` grounding to ``CausalCircuit`` construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

import numpy as np
import pytest

from krrood.entity_query_language.factories import variable
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.probabilistic_circuit.causal.exceptions import (
    SupportDeterminismVerificationResult,
)
from probabilistic_model.learning.learning_method import StratifiedLearning
from probabilistic_model.probabilistic_circuit.relational.part_learning_methods import (
    PartLearningMethods,
)
from probabilistic_model.probabilistic_circuit.relational.causal import (
    RelationalCausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    AmbiguousVariablePathError,
    VariableNotFoundError,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    GroundingMode,
    RelationalProbabilisticCircuit,
)
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from random_events.product_algebra import SimpleEvent
from random_events.variable import Variable
from ..dataset.example_classes import SceneRoom, SceneRoomAggregations
from ...probabilistic_model_test.test_causal.test_causal_circuit_types import (
    events_over,
    probabilities_of,
)
from .test_relational_probabilistic_circuit import (  # noqa: F401
    _room_with_chair_count,
    correlated_room_query,
    relational_probabilistic_circuit,
    room_query_4,
    scenario,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.query.match import Match

# %% resolve_variable


def test_resolve_variable_matches_the_full_name(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    variable = RelationalCausalCircuit.resolve_variable(
        grounded, "SceneRoomAggregations.chair_count()"
    )
    assert variable.name == "SceneRoomAggregations.chair_count()"


def test_resolve_variable_matches_an_unambiguous_suffix(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    variable = RelationalCausalCircuit.resolve_variable(grounded, "chair_count()")
    assert variable.name == "SceneRoomAggregations.chair_count()"


def test_resolve_variable_matches_a_relational_index_suffix(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    variable = RelationalCausalCircuit.resolve_variable(grounded, "objects[2].type")
    assert variable.name == "SceneRoom.objects[2].type"


def test_resolve_variable_raises_for_no_match(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    with pytest.raises(VariableNotFoundError):
        RelationalCausalCircuit.resolve_variable(grounded, "not_a_real_variable")


def test_resolve_variable_raises_for_ambiguous_suffix(
    relational_probabilistic_circuit, room_query_4
):
    """
    ``type`` alone matches every ``objects[i].type``, so it must be rejected rather than
    silently picking one.
    """
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    with pytest.raises(AmbiguousVariablePathError):
        RelationalCausalCircuit.resolve_variable(grounded, "type")


# %% RelationalCausalCircuit.ground


def _resolve_against_a_fresh_grounding(
    relational_probabilistic_circuit, query, paths, grounding_mode
):
    """
    Resolve each of ``paths`` against a circuit grounded just for that purpose.

    ``RelationalCausalCircuit.ground`` requires already-resolved Variables (see its
    docstring), since it accepts no path strings itself; a caller who only has names
    grounds once to resolve them, the way this helper does, before grounding again for
    the ``CausalCircuit`` it actually wants.

    :param relational_probabilistic_circuit: The relational circuit to ground.
    :param query: The grounding query.
    :param paths: Dotted access-path strings to resolve.
    :param grounding_mode: The grounding mode to resolve under.
    :return: The resolved Variables, in input order.
    """
    grounded = relational_probabilistic_circuit.ground(
        query, grounding_mode=grounding_mode
    )
    return [RelationalCausalCircuit.resolve_variable(grounded, path) for path in paths]


def test_relational_causal_circuit_ground_returns_a_causal_circuit(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    [chair_count_variable, object_type_variable] = _resolve_against_a_fresh_grounding(
        relational_probabilistic_circuit,
        room_query_4,
        ["chair_count()", "objects[0].type"],
        GroundingMode.SAMPLED,
    )

    np.random.seed(0)
    causal_circuit = RelationalCausalCircuit().ground(
        relational_probabilistic_circuit,
        room_query_4,
        causal_variables=[chair_count_variable],
        effect_variables=[object_type_variable],
    )
    assert isinstance(causal_circuit, CausalCircuit)


def test_relational_causal_circuit_ground_defaults_to_causal_sampled(
    relational_probabilistic_circuit, room_query_4
):
    """
    The default grounding mode must retain undetermined latents, since that is the
    entire point of grounding a CausalCircuit this way.
    """
    np.random.seed(0)
    [chair_count_variable, object_type_variable] = _resolve_against_a_fresh_grounding(
        relational_probabilistic_circuit,
        room_query_4,
        ["chair_count()", "objects[0].type"],
        GroundingMode.SAMPLED,
    )

    np.random.seed(0)
    causal_circuit = RelationalCausalCircuit().ground(
        relational_probabilistic_circuit,
        room_query_4,
        causal_variables=[chair_count_variable],
        effect_variables=[object_type_variable],
    )
    names = {v.name for v in causal_circuit.probabilistic_circuit.variables}
    assert "SceneRoomAggregations.chair_count()" in names


def test_relational_causal_circuit_ground_backdoor_adjustment_runs(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    [chair_count_variable, object_type_variable] = _resolve_against_a_fresh_grounding(
        relational_probabilistic_circuit,
        room_query_4,
        ["chair_count()", "objects[0].type"],
        GroundingMode.SAMPLED,
    )

    np.random.seed(0)
    causal_circuit = RelationalCausalCircuit().ground(
        relational_probabilistic_circuit,
        room_query_4,
        causal_variables=[chair_count_variable],
        effect_variables=[object_type_variable],
    )
    chair_count_variable = RelationalCausalCircuit.resolve_variable(
        causal_circuit.probabilistic_circuit, "chair_count()"
    )
    object_type_variable = RelationalCausalCircuit.resolve_variable(
        causal_circuit.probabilistic_circuit, "objects[0].type"
    )
    interventional_circuit = causal_circuit.backdoor_adjustment(
        cause_variable=chair_count_variable, effect_variable=object_type_variable
    )
    assert interventional_circuit.is_valid()


def test_relational_causal_circuit_ground_warns_on_expensive_adjustment_set(
    relational_probabilistic_circuit, room_query_4, caplog
):
    """
    Under GroundingMode.EXACT, registering more than one relational adjustment variable
    together must warn once the product of their region counts exceeds the configured
    threshold, rather than waiting to discover the cost at query time.
    """
    object_type_0, object_type_1, chair_count_variable, table_count_variable = (
        _resolve_against_a_fresh_grounding(
            relational_probabilistic_circuit,
            room_query_4,
            [
                "objects[0].type",
                "objects[1].type",
                "chair_count()",
                "table_count()",
            ],
            GroundingMode.EXACT,
        )
    )

    with caplog.at_level("WARNING"):
        RelationalCausalCircuit(adjustment_region_count_warning_threshold=0).ground(
            relational_probabilistic_circuit,
            room_query_4,
            causal_variables=[object_type_0],
            effect_variables=[object_type_1],
            adjustment_variables=[chair_count_variable, table_count_variable],
            grounding_mode=GroundingMode.EXACT,
        )
    assert any("Cartesian product" in message for message in caplog.messages)


def test_relational_causal_circuit_ground_does_not_warn_below_threshold(
    relational_probabilistic_circuit, room_query_4, caplog
):
    object_type_0, object_type_1, chair_count_variable, table_count_variable = (
        _resolve_against_a_fresh_grounding(
            relational_probabilistic_circuit,
            room_query_4,
            [
                "objects[0].type",
                "objects[1].type",
                "chair_count()",
                "table_count()",
            ],
            GroundingMode.EXACT,
        )
    )

    with caplog.at_level("WARNING"):
        RelationalCausalCircuit(
            adjustment_region_count_warning_threshold=10_000
        ).ground(
            relational_probabilistic_circuit,
            room_query_4,
            causal_variables=[object_type_0],
            effect_variables=[object_type_1],
            adjustment_variables=[chair_count_variable, table_count_variable],
            grounding_mode=GroundingMode.EXACT,
        )
    assert not any("Cartesian product" in message for message in caplog.messages)


# %% RelationalCausalCircuit.fit


@pytest.fixture
def many_chair_count_rooms():
    """
    Two large chair-count partitions (20 rooms each), with each room's continuous
    position varied -- enough rows and variance for JointProbabilityTree to split each
    partition further on its own, the precondition the regression test below needs.
    """
    rng = np.random.default_rng(0)
    return [_room_with_chair_count(rng, 1) for _ in range(20)] + [
        _room_with_chair_count(rng, 3) for _ in range(20)
    ]


def test_fit_stratifies_the_class_circuit_by_the_given_variable(many_chair_count_rooms):
    chair_count_variable = variable(SceneRoomAggregations).chair_count()
    model = RelationalProbabilisticCircuit(
        learning_method=StratifiedLearning(
            variables=[chair_count_variable._name_], method=JointProbabilityTree()
        ),
    )
    model.fit(many_chair_count_rooms)
    resolved_chair_count = next(
        v
        for v in model.class_probabilistic_circuit.variables
        if v.name == "SceneRoomAggregations.chair_count()"
    )
    for value in (1, 3):
        event = SimpleEvent.from_data({resolved_chair_count: value}).as_composite_set()
        probability = model.class_probabilistic_circuit.probability(
            event.fill_missing_variables_pure(
                model.class_probabilistic_circuit.variables
            )
        )
        assert probability == pytest.approx(0.5, abs=0.01)


def test_verify_support_determinism_survives_a_stratified_partitions_own_further_splits(
    many_chair_count_rooms, correlated_room_query
):
    """
    Regression test: a stratified partition's own further JointProbabilityTree splits on
    unrelated variables (here, room position) used to make
    ``verify_support_determinism`` fail even though the stratification itself was
    correct.

    ``CausalCircuit._check_support_disjointness`` used to compute each variable's
    marginal via ``ProbabilisticCircuit.marginal``, whose trailing
    ``SumUnit.simplify()`` flattens nested SumUnits together, merging a partition's own
    further-split branches into siblings of a different partition's branches and erasing
    which stratified value each one actually belonged to. Grounding also used to trigger
    the same flattening on the class circuit itself, via ``_condition_class_circuit``
    unconditionally calling ``log_conditional_in_place`` even when there was nothing to
    condition on.
    """
    chair_count_variable = variable(SceneRoomAggregations).chair_count()
    model = RelationalProbabilisticCircuit(
        learning_method=StratifiedLearning(
            variables=[chair_count_variable._name_], method=JointProbabilityTree()
        ),
    )
    relational_causal_circuit = RelationalCausalCircuit()
    model.fit(many_chair_count_rooms)

    np.random.seed(0)
    grounded = model.ground(correlated_room_query, grounding_mode=GroundingMode.SAMPLED)
    resolved_chair_count = next(
        v for v in grounded.variables if v.name == "SceneRoomAggregations.chair_count()"
    )
    object_type_variable = next(
        v for v in grounded.variables if v.name == "SceneRoom.objects[0].type"
    )

    causal_circuit = relational_causal_circuit.from_grounded_circuit(
        grounded,
        causal_variables=[resolved_chair_count],
        effect_variables=[object_type_variable],
        trim_to_registered_variables=True,
    )
    assert isinstance(causal_circuit, CausalCircuit)


# %% stratifying by several variables, and stratifying an exchangeable part's template


def _root_partition_count(circuit) -> int:
    """
    :return: How many partitions a stratified fit combined under ``circuit``'s root.
    """
    return len(circuit.root.subcircuits)


def test_fit_stratifies_the_class_circuit_by_every_given_variable(
    many_chair_count_rooms,
):
    """
    Stratifying by two variables partitions the training rows by their joint value, so
    the class circuit is support-deterministic over either one.
    """
    aggregations = variable(SceneRoomAggregations)
    model = RelationalProbabilisticCircuit(
        learning_method=StratifiedLearning(
            variables=[
                aggregations.chair_count()._name_,
                aggregations.table_count()._name_,
            ],
            method=JointProbabilityTree(),
        ),
    )
    model.fit(many_chair_count_rooms)
    joint_values = {
        (
            SceneRoomAggregations(role_taker=room).chair_count(),
            SceneRoomAggregations(role_taker=room).table_count(),
        )
        for room in many_chair_count_rooms
    }
    assert _root_partition_count(model.class_probabilistic_circuit) == len(joint_values)


def test_fit_stratifies_an_exchangeable_parts_template_by_the_given_variable(
    many_chair_count_rooms,
):
    """
    A per-object cause needs the part's template to be support-deterministic over it,
    which the plain template fit gives no guarantee of.
    """
    chair_count_variable = variable(SceneRoomAggregations).chair_count()
    model = RelationalProbabilisticCircuit(
        learning_method=StratifiedLearning(
            variables=[chair_count_variable._name_], method=JointProbabilityTree()
        ),
        part_learning_methods=PartLearningMethods.from_methods_and_default(
            {
                variable(SceneRoom).objects: StratifiedLearning(
                    variables=["type"], method=JointProbabilityTree()
                )
            },
            JointProbabilityTree(),
        ),
    )
    model.fit(many_chair_count_rooms)
    object_types = {
        scene_object.type
        for room in many_chair_count_rooms
        for scene_object in room.objects
    }
    template_circuit = model.exchangeable_distribution_templates[
        "objects"
    ].template_distribution.class_probabilistic_circuit
    assert _root_partition_count(template_circuit) == len(object_types)


# %% leaf size


def _tree_leaf_count(tree_root: SumUnit) -> int:
    """
    :param tree_root: The root sum a ``JointProbabilityTree`` fit produced.
    :return: How many leaves the tree grew: the product units its root sum mixes.
    """
    return len(tree_root.subcircuits)


def test_plain_fit_honours_the_minimum_samples_per_leaf(many_chair_count_rooms):
    """
    With the minimum set to the whole training set, the class circuit cannot split at
    all, so every variable is modeled by exactly one leaf.
    """
    model = RelationalProbabilisticCircuit(
        learning_method=JointProbabilityTree(
            min_samples_per_leaf=len(many_chair_count_rooms)
        ),
    )
    model.fit(many_chair_count_rooms)

    assert _tree_leaf_count(model.class_probabilistic_circuit.root) == 1


def test_stratified_fit_honours_the_minimum_samples_per_leaf(many_chair_count_rooms):
    """
    With the minimum set to a partition's size, each partition is one leaf, so a
    variable has as many leaves as there are partitions.
    """
    chair_count_variable = variable(SceneRoomAggregations).chair_count()
    model = RelationalProbabilisticCircuit(
        learning_method=StratifiedLearning(
            variables=[chair_count_variable._name_],
            method=JointProbabilityTree(min_samples_per_leaf=20),
        ),
    )
    model.fit(many_chair_count_rooms)

    partitions = model.class_probabilistic_circuit.root.subcircuits
    assert [_tree_leaf_count(partition) for partition in partitions] == [1, 1]


def test_minimum_samples_per_leaf_reaches_an_exchangeable_parts_template(
    many_chair_count_rooms,
):
    object_count = sum(len(room.objects) for room in many_chair_count_rooms)
    model = RelationalProbabilisticCircuit(
        part_learning_methods=PartLearningMethods.from_methods_and_default(
            {
                variable(SceneRoom).objects: JointProbabilityTree(
                    min_samples_per_leaf=object_count
                )
            },
            JointProbabilityTree(),
        ),
    )
    model.fit(many_chair_count_rooms)

    template_circuit = model.exchangeable_distribution_templates[
        "objects"
    ].template_distribution.class_probabilistic_circuit
    assert _tree_leaf_count(template_circuit.root) == 1


# %% latent coverage of a sampled grounding


def test_sampled_grounding_gives_every_stratum_its_own_latent_value(
    many_chair_count_rooms, correlated_room_query
):
    """
    Regression test: with too few Monte-Carlo samples to hit every stratum's latent
    value, a stratum none of the samples fitted used to be handed the first sampled
    instance regardless, so it claimed another stratum's value and the grounded circuit
    was no longer support-deterministic over the latent.
    """
    chair_count_variable = variable(SceneRoomAggregations).chair_count()
    model = RelationalProbabilisticCircuit(
        monte_carlo_sample_count=1,
        learning_method=StratifiedLearning(
            variables=[chair_count_variable._name_], method=JointProbabilityTree()
        ),
    )
    relational_causal_circuit = RelationalCausalCircuit()
    model.fit(many_chair_count_rooms)

    np.random.seed(0)
    grounded = model.ground(correlated_room_query, grounding_mode=GroundingMode.SAMPLED)
    resolved_chair_count = next(
        v for v in grounded.variables if v.name == "SceneRoomAggregations.chair_count()"
    )
    object_type_variable = next(
        v for v in grounded.variables if v.name == "SceneRoom.objects[0].type"
    )

    causal_circuit = relational_causal_circuit.from_grounded_circuit(
        grounded,
        causal_variables=[resolved_chair_count],
        effect_variables=[object_type_variable],
        trim_to_registered_variables=True,
    )
    for value in (1, 3):
        event = SimpleEvent.from_data({resolved_chair_count: value}).as_composite_set()
        probability = causal_circuit.probabilistic_circuit.probability(
            event.fill_missing_variables_pure(
                causal_circuit.probabilistic_circuit.variables
            )
        )
        assert probability == pytest.approx(0.5, abs=0.01)


@pytest.fixture
def rooms_with_every_object_type_in_every_partition():
    """
    Two chair-count partitions (20 rooms each) whose objects show both object types in
    either partition, so the object template conditioned on either sampled count keeps
    the same support on an object's type.
    """
    rng = np.random.default_rng(0)
    return [_room_with_chair_count(rng, 1) for _ in range(20)] + [
        _room_with_chair_count(rng, 2) for _ in range(20)
    ]


def _part_attribute_causal_circuit(rooms, correlated_room_query) -> CausalCircuit:
    """
    Register the first object's type as a cause of the room's x position on a model
    whose object template is stratified by type.
    """
    model = RelationalProbabilisticCircuit(
        learning_method=JointProbabilityTree(min_samples_per_leaf=5),
        part_learning_methods=PartLearningMethods.from_methods_and_default(
            {
                variable(SceneRoom).objects: StratifiedLearning(
                    variables=["type"],
                    method=JointProbabilityTree(min_samples_per_leaf=5),
                )
            },
            JointProbabilityTree(),
        ),
    )
    model.fit(rooms)

    np.random.seed(0)
    grounded = model.ground(correlated_room_query, grounding_mode=GroundingMode.SAMPLED)
    object_type_variable = next(
        v for v in grounded.variables if v.name == "SceneRoom.objects[0].type"
    )
    position_variable = next(
        v for v in grounded.variables if v.name == "SceneRoom.position.x"
    )
    return RelationalCausalCircuit().from_grounded_circuit(
        grounded,
        causal_variables=[object_type_variable],
        effect_variables=[position_variable],
        trim_to_registered_variables=True,
    )


def test_a_part_attribute_of_a_stratified_template_registers_as_a_cause(
    rooms_with_every_object_type_in_every_partition, correlated_room_query
):
    """
    Regression test: grounding a part used to condition the template on the latents
    through the simplifying conditional, which merged the template's own strata into one
    sum whose children overlapped on the stratified attribute, so a part attribute could
    never pass support-determinism verification as a cause.
    """
    causal_circuit = _part_attribute_causal_circuit(
        rooms_with_every_object_type_in_every_partition, correlated_room_query
    )
    assert isinstance(causal_circuit, CausalCircuit)


def test_a_part_attribute_whose_support_depends_on_the_sampled_count_is_no_cause(
    many_chair_count_rooms, correlated_room_query
):
    """
    Grounding with the chair count left open mixes one copy of the object template per
    sampled count.

    Here every object of a three-chair room is a chair, so the copy for
    that count supports only one type while the copy for one chair supports both: the
    copies overlap on the type without coinciding, and there is no disjoint region of
    the type to intervene on.
    """
    with pytest.raises(SupportDeterminismVerificationResult):
        _part_attribute_causal_circuit(many_chair_count_rooms, correlated_room_query)


# %% the circuit type a grounding is wrapped in


@dataclass
class GroundingInBothCircuitTypes:
    """
    One grounding of a relational circuit, wrapped as a causal circuit of each circuit
    type, with the variables both of them registered.
    """

    units: CausalCircuit[ProbabilisticCircuit]
    """
    The causal circuit over the grounding as a graph of units.
    """

    layers: CausalCircuit[LayeredProbabilisticCircuit]
    """
    The causal circuit over the same grounding, converted into layers.
    """

    cause: Variable
    """
    The registered cause Variable.
    """

    effect: Variable
    """
    The registered effect Variable.
    """


def _wrapped_in_both_circuit_types(
    relational_circuit: RelationalProbabilisticCircuit,
    query: Match,
    cause_path: str,
    effect_path: str,
    seed: int = 0,
) -> GroundingInBothCircuitTypes:
    """
    Ground a relational circuit once and wrap that one grounding twice, so that the two
    causal circuits differ in nothing but how their nodes are stored.

    :param relational_circuit: The relational circuit to ground.
    :param query: The grounding query.
    :param cause_path: The access path of the cause variable.
    :param effect_path: The access path of the effect variable.
    :param seed: The seed the undetermined statistics are sampled under.
    :return: Both causal circuits and the variables they registered.
    """
    np.random.seed(seed)
    grounded = relational_circuit.ground(query, GroundingMode.SAMPLED)
    cause = RelationalCausalCircuit.resolve_variable(grounded, cause_path)
    effect = RelationalCausalCircuit.resolve_variable(grounded, effect_path)
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(grounded)
    bridge = RelationalCausalCircuit()
    return GroundingInBothCircuitTypes(
        bridge.from_grounded_circuit(grounded, [cause], [effect]),
        bridge.from_grounded_circuit(layered, [cause], [effect]),
        cause,
        effect,
    )


def test_a_grounding_is_support_deterministic_in_both_circuit_types(
    relational_probabilistic_circuit, room_query_4
):
    """
    A grounding passes the support-determinism check whichever circuit type it is read
    in.
    """
    both = _wrapped_in_both_circuit_types(
        relational_probabilistic_circuit,
        room_query_4,
        "chair_count()",
        "objects[0].type",
    )
    assert both.layers.verify_support_determinism().passed is True
    assert both.units.verify_support_determinism().passed is True


def test_the_interventional_circuit_of_a_layered_grounding_is_layered(
    relational_probabilistic_circuit, room_query_4
):
    """
    Intervening on a layered grounding gives back a layered circuit.
    """
    both = _wrapped_in_both_circuit_types(
        relational_probabilistic_circuit,
        room_query_4,
        "chair_count()",
        "objects[0].type",
    )
    assert isinstance(
        both.layers.backdoor_adjustment(both.cause, both.effect),
        LayeredProbabilisticCircuit,
    )


def test_both_circuit_types_intervene_on_a_grounding_the_same(
    relational_probabilistic_circuit, room_query_4
):
    """
    Both circuit types give the interventional distribution of one grounding the same
    probabilities.
    """
    both = _wrapped_in_both_circuit_types(
        relational_probabilistic_circuit,
        room_query_4,
        "chair_count()",
        "objects[0].type",
    )
    events = events_over([both.cause, both.effect], 20)
    assert probabilities_of(
        both.layers.backdoor_adjustment(both.cause, both.effect), events
    ) == pytest.approx(
        probabilities_of(
            both.units.backdoor_adjustment(both.cause, both.effect), events
        )
    )


def test_trimming_a_layered_grounding_keeps_only_the_registered_variables(
    relational_probabilistic_circuit, room_query_4
):
    """
    Trimming a layered grounding leaves exactly the cause and the effect.
    """
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, GroundingMode.SAMPLED
    )
    cause = RelationalCausalCircuit.resolve_variable(grounded, "chair_count()")
    effect = RelationalCausalCircuit.resolve_variable(grounded, "objects[0].type")
    trimmed = RelationalCausalCircuit().from_grounded_circuit(
        RustworkxCircuitToLayeredCircuitConverter.convert(grounded),
        [cause],
        [effect],
        trim_to_registered_variables=True,
    )
    assert list(trimmed.probabilistic_circuit.variables) == sorted(
        [cause, effect], key=lambda circuit_variable: circuit_variable.name
    )


def test_the_adjustment_warning_counts_the_regions_of_a_layered_grounding(
    relational_probabilistic_circuit, room_query_4, caplog
):
    """
    The warning about an expensive adjustment set also fires for a layered grounding.
    """
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, GroundingMode.EXACT
    )
    layered = RustworkxCircuitToLayeredCircuitConverter.convert(grounded)
    resolve = RelationalCausalCircuit.resolve_variable
    with caplog.at_level("WARNING"):
        RelationalCausalCircuit(
            adjustment_region_count_warning_threshold=0
        ).from_grounded_circuit(
            layered,
            [resolve(layered, "objects[0].type")],
            [resolve(layered, "objects[1].type")],
            adjustment_variables=[
                resolve(layered, "chair_count()"),
                resolve(layered, "table_count()"),
            ],
        )
    assert any("Cartesian product" in message for message in caplog.messages)
