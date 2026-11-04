import pytest

from probabilistic_model.causal_structure.graph import NodeLevel
from probabilistic_model.causal_structure.learning import (
    RelationalCausalStructureLearner,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
)

from .test_relational_probabilistic_circuit import (  # noqa: F401  # fixtures
    layered_relational_probabilistic_circuit,
    relational_probabilistic_circuit,
    room_query_4,
    scenario,
)

# %% the tables a relational circuit is fitted on


def test_training_tables_match_the_fitted_tables(
    relational_probabilistic_circuit, scenario
):
    tables = relational_probabilistic_circuit.training_tables(list(scenario))
    template = relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]

    assert list(tables.example_rows.columns) == [
        variable.name
        for variable in relational_probabilistic_circuit.class_probabilistic_circuit.variables
    ]
    assert set(tables.part_rows["objects"].columns) == {
        variable.name
        for variable in template.template_distribution.class_probabilistic_circuit.variables
    }


# %% the graph learned from them


@pytest.mark.parametrize(
    "fitted_circuit",
    ["relational_probabilistic_circuit", "layered_relational_probabilistic_circuit"],
)
def test_learned_graph_variables_match_query_variables(
    fitted_circuit, scenario, room_query_4, request
):
    circuit: RelationalProbabilisticCircuit = request.getfixturevalue(fitted_circuit)
    graph = (
        RelationalCausalStructureLearner().learn(circuit, list(scenario)).selector.graph
    )
    grounded_names = {
        variable.name for variable in circuit.ground(room_query_4).variables
    }

    part_names = {
        node.variable_of_part(0).name
        for node in graph.nodes()
        if node.level is NodeLevel.PART_ATTRIBUTE
    }
    example_names = {
        node.name
        for node in graph.nodes()
        if node.level is not NodeLevel.PART_ATTRIBUTE
    }

    assert part_names <= grounded_names
    assert example_names <= grounded_names | {
        variable.name for variable in circuit.class_probabilistic_circuit.variables
    }


def test_statistics_are_aggregation_nodes(relational_probabilistic_circuit, scenario):
    graph = (
        RelationalCausalStructureLearner()
        .learn(relational_probabilistic_circuit, list(scenario))
        .selector.graph
    )
    template = relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]

    assert {
        node.name
        for node in graph.nodes()
        if node.level is NodeLevel.AGGREGATION_STATISTIC
    } == {variable.name for variable in template.latent_variables}
