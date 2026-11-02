import pytest
from random_events.interval import closed

from experiments.probabilistic_model_experiments.layered_circuit_speed import (
    BenchmarkStage,
    CircuitsDisagreeError,
    CorrelatedNormalTreeFactory,
    StaircaseTruncation,
    measure_conditioning,
    measure_query_durations,
    measure_truncation_scaling,
    raise_if_answers_disagree,
    staircase_of_boxes,
)
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)


@pytest.fixture(scope="module")
def tree_factory() -> CorrelatedNormalTreeFactory:
    return CorrelatedNormalTreeFactory(number_of_samples=500, min_samples_per_leaf=0.1)


@pytest.fixture(scope="module")
def rustworkx_circuit(tree_factory):
    return tree_factory.learn_circuit()


@pytest.fixture(scope="module")
def layered(rustworkx_circuit):
    return RustworkxCircuitToLayeredCircuitConverter.convert(rustworkx_circuit)


def test_the_staircase_has_one_simple_set_per_box(layered):
    first, second = layered.variables[0], layered.variables[1]
    number_of_boxes = 4
    staircase = staircase_of_boxes(
        first,
        second,
        closed(0.0, 1.0).simple_sets[0],
        closed(0.0, 1.0).simple_sets[0],
        number_of_boxes,
    )
    assert len(staircase.simple_sets) == number_of_boxes


def test_query_durations_measure_every_query_of_the_stage(rustworkx_circuit, layered):
    results = measure_query_durations(
        BenchmarkStage.BEFORE_TRUNCATION, rustworkx_circuit, layered
    )
    assert len(results) == 6
    assert {result.stage for result in results} == {BenchmarkStage.BEFORE_TRUNCATION}


def test_the_number_of_layers_does_not_grow_with_the_number_of_simple_sets(
    rustworkx_circuit, layered
):
    truncations = StaircaseTruncation.variants()
    scaling = measure_truncation_scaling(rustworkx_circuit, layered, truncations)
    assert [result.number_of_simple_sets for result in scaling.results] == [
        truncation.number_of_simple_sets for truncation in truncations
    ]
    numbers_of_layers = {result.result_number_of_layers for result in scaling.results}
    assert numbers_of_layers == {len(scaling.layered_truncated.layers)}


def test_conditioning_is_measured_for_every_number_of_variables(
    rustworkx_circuit, layered
):
    results = measure_conditioning(rustworkx_circuit, layered)
    assert [result.number_of_conditioned_variables for result in results] == list(
        range(1, len(layered.variables) + 1)
    )


def test_the_largest_number_of_simple_sets_is_the_most_any_truncation_has(
    rustworkx_circuit, layered
):
    truncations = [
        StaircaseTruncation(3),
        StaircaseTruncation(7),
        StaircaseTruncation(2),
    ]
    scaling = measure_truncation_scaling(rustworkx_circuit, layered, truncations)
    assert scaling.largest_number_of_simple_sets == max(
        truncation.number_of_simple_sets for truncation in truncations
    )


def test_agreeing_answers_do_not_raise():
    raise_if_answers_disagree("log_likelihood", [0.5, 1.0], [0.5, 1.0 + 1e-10])


def test_disagreeing_answers_raise_with_the_largest_difference():
    with pytest.raises(CircuitsDisagreeError) as raised:
        raise_if_answers_disagree(
            "log_likelihood", [0.5, 1.0], [0.5, 1.5], absolute_tolerance=1e-3
        )
    assert raised.value.query == "log_likelihood"
    assert raised.value.largest_difference == pytest.approx(0.5)
    assert raised.value.absolute_tolerance == 1e-3
