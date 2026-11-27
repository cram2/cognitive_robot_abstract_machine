"""
Tests that a causal query over a structure learned with fast causal inference, which
allows hidden confounders, adjusts for a recorded confounder, refuses an effect a hidden
one may confound, and keeps each relational query to the graph of its scope.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import pytest

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.exceptions import EffectNotIdentified
from krrood.entity_query_language.factories import a, cause, confounder
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.causal_relation import (
    AttributeEndpoint,
    CausalRelation,
    CausalRelationsOfClass,
)
from krrood.parametrization.model_registries import CausalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from krrood.parametrization.probabilistic_backend import ProbabilisticBackend
from probabilistic_model.causal_structure.ancestral_graph import AncestralGraph
from probabilistic_model.causal_structure.domain_causal_knowledge import (
    DomainCausalKnowledge,
)
from probabilistic_model.causal_structure.exceptions import (
    InvalidAdjustmentSet,
    QuerySpansSeveralParts,
    UnsupportedKnowledgeUnderLatentConfounding,
)
from probabilistic_model.causal_structure.graph import (
    GraphNode,
    NodeLevel,
)
from probabilistic_model.causal_structure.identification import (
    InvalidAdjustmentReason,
    InvalidDeclaredAdjustmentRepair,
    UnidentifiedReason,
)
from probabilistic_model.causal_structure.learning import (
    FastCausalInferenceStructureLearner,
    KnowledgeKind,
    LearnedCausalStructure,
    RelationalCausalStructureLearner,
)
from probabilistic_model.causal_structure.relational_checks import ExampleToPartCheck
from probabilistic_model.learning.structure.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.structure.jpt.variables import (
    infer_variables_from_dataframe,
)
from probabilistic_model.learning.learning_method import StratifiedLearning
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    PartTrainingTable,
    RelationalProbabilisticCircuit,
)
from random_events.variable import Integer

from ...dataset.example_classes import ConfoundedPour, PouredDrop

# %% pours whose tremor drives the speed and the spill


@dataclass(kw_only=True)
class PourWithRecordedTremor:
    """
    A pour whose tremor, recorded, drives both its speed and its spill.
    """

    nozzle: int
    """
    Which nozzle is fitted, which drives the speed only.
    """

    tremor: int
    """
    Whether the hand trembles.
    """

    speed: int
    """
    Whether the pour is fast.
    """

    spill: int
    """
    Whether the pour spills.
    """


@dataclass(kw_only=True)
class PourWithHiddenTremor:
    """
    A pour whose tremor drives both its speed and its spill, but goes unrecorded.
    """

    nozzle: int
    """
    Which nozzle is fitted, which drives the speed only.
    """

    speed: int
    """
    Whether the pour is fast.
    """

    spill: int
    """
    Whether the pour spills.
    """


def flipped(
    values: np.ndarray, random_state: np.random.Generator, flip_rate: float = 0.1
) -> np.ndarray:
    """
    :param values: Binary values.
    :param random_state: The source of randomness.
    :param flip_rate: How often a value is flipped.
    :return: The values with that share flipped.
    """
    return values ^ (random_state.random(len(values)) < flip_rate)


def recorded_tremor_rows(row_count: int = 20000, seed: int = 0) -> pd.DataFrame:
    """
    :return: Rows where the nozzle and the tremor drive the speed, and the speed and
        the tremor drive the spill: P(spill | do(speed = 1)) = 0.9 and
        P(spill | do(speed = 0)) = 0.5.
    """
    random_state = np.random.default_rng(seed)
    nozzle = random_state.integers(0, 2, row_count)
    tremor = random_state.integers(0, 2, row_count)
    speed = flipped(nozzle | tremor, random_state)
    return pd.DataFrame(
        {
            "PourWithRecordedTremor.nozzle": nozzle,
            "PourWithRecordedTremor.tremor": tremor,
            "PourWithRecordedTremor.speed": speed,
            "PourWithRecordedTremor.spill": flipped(speed | tremor, random_state),
        }
    )


def hidden_tremor_rows(
    speed_spills: bool, row_count: int = 20000, seed: int = 0
) -> pd.DataFrame:
    """
    :param speed_spills: Whether the speed drives the spill, beside the hidden tremor.
    :return: Rows where the nozzle and the hidden tremor drive the speed, and the
        tremor drives the spill.
    """
    random_state = np.random.default_rng(seed)
    nozzle = random_state.integers(0, 2, row_count)
    tremor = random_state.integers(0, 2, row_count)
    speed = flipped(nozzle | tremor, random_state)
    spill_cause = (speed | tremor) if speed_spills else tremor
    return pd.DataFrame(
        {
            "PourWithHiddenTremor.nozzle": nozzle,
            "PourWithHiddenTremor.speed": speed,
            "PourWithHiddenTremor.spill": flipped(spill_cause, random_state),
        }
    )


def learned_structure(
    clazz: type, rows: pd.DataFrame
) -> dict[type, LearnedCausalStructure]:
    """
    :param clazz: A flat class whose fields are the rows' columns.
    :param rows: One column per field, named ``Class.field``.
    :return: The structure fast causal inference learns from the rows, kept as causal
        relations of the class in a class diagram.
    """
    nodes = [
        GraphNode(variable=Integer(name=column), level=NodeLevel.EXAMPLE_ATTRIBUTE)
        for column in rows.columns
    ]
    graph = FastCausalInferenceStructureLearner().learn(
        rows, nodes, DomainCausalKnowledge()
    )
    class_diagram = ClassDiagram(classes=[clazz])
    owner = class_diagram.get_wrapped_class(clazz=clazz)
    relations = CausalRelationsOfClass(class_diagram=class_diagram, owner=owner)
    endpoint_of = {
        f"{clazz.__name__}.{clazz_field.name}": AttributeEndpoint(
            fields=(owner.get_field(name=clazz_field.name),)
        )
        for clazz_field in fields(clazz)
    }
    relations.replace_relations(
        relations=[
            CausalRelation(
                source=owner,
                target=owner,
                first=endpoint_of[edge.first.name],
                second=endpoint_of[edge.second.name],
                mark_at_first=edge.mark_at_first,
                mark_at_second=edge.mark_at_second,
            )
            for edge in graph.edges()
        ]
    )
    return {
        clazz: LearnedCausalStructure(
            relations, nodes, rows, graph_class=AncestralGraph
        )
    }


def recorded_tremor_registry() -> CausalCircuitRegistry:
    """
    :return: A causal circuit over the recorded tremor's rows, stratified on the speed
        and the tremor, so that it can intervene on the speed and adjust for the tremor.
    """
    rows = recorded_tremor_rows()
    variables = infer_variables_from_dataframe(rows)
    by_name = {annotated.variable.name: annotated.variable for annotated in variables}
    causes = [
        by_name["PourWithRecordedTremor.speed"],
        by_name["PourWithRecordedTremor.tremor"],
    ]
    circuit = StratifiedLearning(
        variables=[variable.name for variable in causes],
        method=JointProbabilityTree(),
    ).fit(rows, variables)
    return CausalCircuitRegistry(
        circuits={
            PourWithRecordedTremor: CausalCircuit.from_probabilistic_circuit(
                circuit,
                MarginalDeterminismTreeNode.from_variable_order(causes),
                causes,
                [by_name["PourWithRecordedTremor.spill"]],
            )
        }
    )


def recorded_tremor_query(
    declares_the_nozzle: bool = False,
) -> Match[PourWithRecordedTremor]:
    """
    :param declares_the_nozzle: Whether the nozzle is marked ``confounder``.
    :return: Does the speed cause the spill?
    """
    query = a(PourWithRecordedTremor)(
        nozzle=confounder if declares_the_nozzle else ...,
        tremor=...,
        speed=cause,
        spill=...,
    )
    query.causes_effect(query.spill == 1)
    return query


def hidden_tremor_query() -> Match[PourWithHiddenTremor]:
    """
    :return: Does the speed cause the spill, the tremor unrecorded?
    """
    query = a(PourWithHiddenTremor)(nozzle=..., speed=cause, spill=...)
    query.causes_effect(query.spill == 1)
    return query


def adjustment_names(
    backend: ProbabilisticBackend, query: Match
) -> dict[str, list[str]]:
    """
    :param backend: A backend with a learned structure for the query's class.
    :param query: A causal query.
    :return: The names of the variables each cause adjusts for.
    """
    parameters = UnderspecifiedParameters(statement=query)
    backend.choose_adjustment_sets(parameters=parameters, expression=query)
    return {
        cause_variable.name: [variable.name for variable in variables]
        for cause_variable, variables in parameters.derived_adjustment_variables.items()
    }


# %% a recorded confounder


def test_the_recorded_confounder_is_adjusted_for():
    """
    The recorded confounder is adjusted for.
    """
    backend = ProbabilisticBackend(
        causal_structures=learned_structure(
            PourWithRecordedTremor, recorded_tremor_rows()
        )
    )

    assert adjustment_names(backend, recorded_tremor_query()) == {
        "PourWithRecordedTremor.speed": ["PourWithRecordedTremor.tremor"]
    }


def test_adjusting_for_the_recorded_confounder_gives_the_interventional_answer():
    """
    Adjusting for the recorded confounder gives the interventional answer.
    """
    backend = ProbabilisticBackend(
        model_registry=recorded_tremor_registry(),
        causal_structures=learned_structure(
            PourWithRecordedTremor, recorded_tremor_rows()
        ),
    )

    [answer] = backend.interventional_range(expression=recorded_tremor_query())

    assert sorted(
        (region.lowest_effect_probability, region.highest_effect_probability)
        for region in answer.regions
    ) == [
        (pytest.approx(0.5, abs=0.03), pytest.approx(0.5, abs=0.03)),
        (pytest.approx(0.9, abs=0.03), pytest.approx(0.9, abs=0.03)),
    ]


def test_a_declared_set_leaving_the_confounder_out_is_refused():
    """
    A declared set leaving the confounder out is refused.
    """
    backend = ProbabilisticBackend(
        causal_structures=learned_structure(
            PourWithRecordedTremor, recorded_tremor_rows()
        )
    )

    with pytest.raises(InvalidAdjustmentSet) as refusal:
        adjustment_names(backend, recorded_tremor_query(declares_the_nozzle=True))

    assert refusal.value.reason is InvalidAdjustmentReason.LEAVES_NON_CAUSAL_PATH_OPEN


def test_a_declared_set_leaving_the_confounder_out_is_repaired_by_adding_it():
    """
    A declared set leaving the confounder out is repaired by adding it.
    """
    backend = ProbabilisticBackend(
        causal_structures=learned_structure(
            PourWithRecordedTremor, recorded_tremor_rows()
        ),
        declared_adjustment_policy=InvalidDeclaredAdjustmentRepair(),
    )

    assert adjustment_names(
        backend, recorded_tremor_query(declares_the_nozzle=True)
    ) == {
        "PourWithRecordedTremor.speed": [
            "PourWithRecordedTremor.nozzle",
            "PourWithRecordedTremor.tremor",
        ]
    }


def test_every_possible_ancestor_of_the_effect_is_a_possible_cause():
    """
    Every possible ancestor of the effect is a possible cause.
    """
    query = a(PourWithRecordedTremor)(nozzle=..., tremor=..., speed=..., spill=...)
    query.causes_effect(query.spill == 1)
    backend = ProbabilisticBackend(
        causal_structures=learned_structure(
            PourWithRecordedTremor, recorded_tremor_rows()
        )
    )

    assert [
        variable.name
        for variable in backend.possible_causes(
            parameters=UnderspecifiedParameters(statement=query), expression=query
        )
    ] == [
        "PourWithRecordedTremor.nozzle",
        "PourWithRecordedTremor.tremor",
        "PourWithRecordedTremor.speed",
    ]


# %% a hidden confounder


def test_an_effect_a_hidden_confounder_may_confound_is_refused():
    """
    An effect a hidden confounder may confound is refused.
    """
    backend = ProbabilisticBackend(
        causal_structures=learned_structure(
            PourWithHiddenTremor, hidden_tremor_rows(speed_spills=True)
        )
    )

    with pytest.raises(EffectNotIdentified) as refusal:
        adjustment_names(backend, hidden_tremor_query())

    assert (
        refusal.value.identification.reason,
        refusal.value.suggest_correction(),
    ) == (UnidentifiedReason.NOT_AMENABLE, refusal.value.identification.correction)


def test_the_range_of_an_effect_a_hidden_confounder_may_confound_is_refused():
    """
    The range of an effect a hidden confounder may confound is refused.
    """
    backend = ProbabilisticBackend(
        causal_structures=learned_structure(
            PourWithHiddenTremor, hidden_tremor_rows(speed_spills=True)
        )
    )

    with pytest.raises(EffectNotIdentified) as refusal:
        backend.interventional_range(expression=hidden_tremor_query())

    assert refusal.value.identification.reason is UnidentifiedReason.NOT_AMENABLE


def test_a_cause_linked_to_the_effect_only_by_a_hidden_confounder_is_no_possible_cause():
    """
    A cause linked to the effect only by a hidden confounder is no possible cause.
    """
    query = a(PourWithHiddenTremor)(nozzle=..., speed=..., spill=...)
    query.causes_effect(query.spill == 1)
    backend = ProbabilisticBackend(
        causal_structures=learned_structure(
            PourWithHiddenTremor, hidden_tremor_rows(speed_spills=False)
        )
    )

    assert (
        backend.possible_causes(
            parameters=UnderspecifiedParameters(statement=query), expression=query
        )
        == []
    )


# %% one graph per scope of a relational circuit


def confounded_pours(
    pour_count: int, hidden_tremor: bool = False, seed: int = 0
) -> list[ConfoundedPour]:
    """
    :param hidden_tremor: Whether the tremor that drives the speed and every drop is
        left unrecorded, the recorded one drawn independently.
    :param pour_count: How many pours to draw.
    :param seed: The seed of the draw.
    :return: Pours whose nozzle and tremor drive the speed, and whose speed and tremor
        drive whether each of their one to three drops spills.
    """
    random_state = np.random.default_rng(seed)
    pours = []
    for _ in range(pour_count):
        nozzle, tremor, hidden = random_state.integers(0, 2, 3)
        driving = hidden if hidden_tremor else tremor
        speed = flipped(np.array([nozzle | driving]), random_state)[0]
        pours.append(
            ConfoundedPour(
                nozzle=int(nozzle),
                tremor=int(tremor),
                speed=int(speed),
                drops=[
                    PouredDrop(
                        spilled=int(
                            flipped(np.array([speed | driving]), random_state)[0]
                        )
                    )
                    for _ in range(int(random_state.integers(1, 4)))
                ],
            )
        )
    return pours


def structure_of_pours(
    pours: list[ConfoundedPour],
) -> tuple[RelationalProbabilisticCircuit, LearnedCausalStructure]:
    """
    :param pours: Pours to learn from.
    :return: The relational circuit fitted on them, and the structure fast causal
        inference learns from its tables.
    """
    circuit = RelationalProbabilisticCircuit().fit(pours)
    structure = RelationalCausalStructureLearner(
        class_diagram=ClassDiagram(classes=[ConfoundedPour, PouredDrop]),
        learner=FastCausalInferenceStructureLearner(),
    ).learn(circuit, pours)
    return circuit, structure


@pytest.fixture(scope="module")
def recorded_tremor_structure() -> LearnedCausalStructure:
    """
    :return: The structure learned from the relational circuit of pours with a
        recorded tremor.
    """
    _, structure = structure_of_pours(confounded_pours(pour_count=20000))
    return structure


def spill_query() -> Match[ConfoundedPour]:
    """
    :return: Does the speed cause the first drop to spill?
    """
    query = a(ConfoundedPour)(
        nozzle=..., tremor=..., speed=cause, drops=[a(PouredDrop)(spilled=...)]
    )
    query.causes_effect(query.drops[0].spilled == 1)
    return query


def test_the_collection_table_holds_one_drawn_part_per_instance():
    """
    The collection table holds one drawn part per instance.
    """
    table = PartTrainingTable(
        part_instances=["a", "b", "c", "d", "e"],
        rows=pd.DataFrame({"spilled": [0, 1, 2, 3, 4]}),
        aggregation_names=[],
        example_positions=[0, 0, 0, 2, 2],
    )

    drawn = table.one_part_per_example(seed=0)

    assert (
        drawn.example_positions,
        ["abcde"[row] for row in drawn.rows["spilled"]] == drawn.part_instances,
        drawn.part_instances[0] in "abc" and drawn.part_instances[1] in "de",
        table.one_part_per_example(seed=0).part_instances == drawn.part_instances,
    ) == ([0, 2], True, True, True)


def test_different_seeds_draw_different_parts():
    """
    Different seeds draw different parts.
    """
    table = PartTrainingTable(
        part_instances=list(range(40)),
        rows=pd.DataFrame({"spilled": list(range(40))}),
        aggregation_names=[],
        example_positions=[position // 4 for position in range(40)],
    )

    assert (
        table.one_part_per_example(seed=0).part_instances
        != table.one_part_per_example(seed=1).part_instances
    )


def test_every_relation_is_learned_in_the_class_scope_or_the_collection_scope(
    recorded_tremor_structure,
):
    """
    Every relation is learned in the class scope or the collection scope.
    """
    assert {
        relation.scope for relation in recorded_tremor_structure.relations.relations
    } == {
        None,
        "drops",
    }


def test_the_class_scope_graph_holds_no_part_attribute(recorded_tremor_structure):
    """
    The class scope graph holds no part attribute.
    """
    assert [
        node.level
        for node in recorded_tremor_structure.selector.graph.nodes()
        if node.level is NodeLevel.PART_ATTRIBUTE
    ] == []


def test_the_collection_scope_graph_is_the_measured_confounders_graph(
    recorded_tremor_structure,
):
    """
    The collection scope graph is the measured confounders graph.
    """
    graph = recorded_tremor_structure.selector_covering(
        variable_names=["ConfoundedPour.drops[0].spilled"]
    ).graph

    assert sorted(str(edge) for edge in graph.edges()) == [
        "ConfoundedPour.nozzle o-> ConfoundedPour.speed",
        "ConfoundedPour.speed --> ConfoundedPour.drops.spilled",
        "ConfoundedPour.speed <-o ConfoundedPour.tremor",
        "ConfoundedPour.tremor --> ConfoundedPour.drops.spilled",
    ]


def test_a_query_about_a_part_adjusts_in_its_collections_graph(
    recorded_tremor_structure,
):
    """
    A query about a part adjusts in its collections graph.
    """
    backend = ProbabilisticBackend(
        causal_structures={ConfoundedPour: recorded_tremor_structure}
    )

    assert adjustment_names(backend, spill_query()) == {
        "ConfoundedPour.speed": ["ConfoundedPour.tremor"]
    }


def test_a_query_about_two_parts_is_refused(recorded_tremor_structure):
    """
    A query about two parts is refused.
    """
    with pytest.raises(QuerySpansSeveralParts):
        recorded_tremor_structure.selector_covering(
            variable_names=[
                "ConfoundedPour.drops[0].spilled",
                "ConfoundedPour.drops[1].spilled",
            ]
        )


def test_an_instance_level_hidden_confounder_of_a_part_is_refused():
    """
    An instance level hidden confounder of a part is refused.
    """
    _, structure = structure_of_pours(
        confounded_pours(hidden_tremor=True, pour_count=4000)
    )
    backend = ProbabilisticBackend(causal_structures={ConfoundedPour: structure})

    with pytest.raises(EffectNotIdentified) as refusal:
        adjustment_names(backend, spill_query())

    assert refusal.value.identification.reason is UnidentifiedReason.NOT_AMENABLE


def test_an_example_to_part_check_is_refused_with_a_learner_per_scope():
    """
    An example to part check is refused with a learner per scope.
    """
    pours = confounded_pours(pour_count=200)
    circuit = RelationalProbabilisticCircuit().fit(pours)

    with pytest.raises(UnsupportedKnowledgeUnderLatentConfounding) as refusal:
        RelationalCausalStructureLearner(
            class_diagram=ClassDiagram(classes=[ConfoundedPour, PouredDrop]),
            learner=FastCausalInferenceStructureLearner(),
            example_to_part_check=ExampleToPartCheck(),
        ).learn(circuit, pours)

    assert refusal.value.kind is KnowledgeKind.EXAMPLE_TO_PART_CHECK
