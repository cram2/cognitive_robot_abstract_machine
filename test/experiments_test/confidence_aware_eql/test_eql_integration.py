import pytest
from krrood.entity_query_language.evaluation_context import (
    EvaluationContext,
    set_evaluation_context,
)
from krrood.entity_query_language.factories import an, entity, variable

from experiments.confidence_aware_eql.domains.kitchen import KitchenObject, Material
from experiments.confidence_aware_eql.eql_integration import (
    ConfidenceAwareEvaluationObserver,
)
from experiments.confidence_aware_eql.exceptions import UnfamiliarSampleException
from experiments.confidence_aware_eql.engine.pipeline import ConfidenceModelBuilder
from test.experiments_test.confidence_aware_eql.test_kitchen import kitchen_clusters

NORMAL_PITCHER = KitchenObject(2.50, 0.25, Material.GLASS)
NORMAL_POT = KitchenObject(3.00, 0.30, Material.METAL)
IMPOSSIBLE_CUP = KitchenObject(50.0, 0.10, Material.GLASS)


@pytest.fixture
def evaluator():
    """
    An evaluator learned from the familiar kitchen clusters.
    """
    return ConfidenceModelBuilder(KitchenObject, kitchen_clusters()).build()


def evaluate_heavy_object_rule(world, observer):
    """
    Evaluate the rule "an object heavier than two kilograms" over a world.
    """
    queried_object = variable(KitchenObject, domain=world)
    query = an(entity(queried_object).where(queried_object.weight > 2.0))
    set_evaluation_context(EvaluationContext(observers=[observer]))
    try:
        return list(query.evaluate())
    finally:
        set_evaluation_context(None)


def test_familiar_objects_are_evaluated_without_raising(evaluator):
    """
    Familiar objects pass the rule conclusion without raising.
    """
    observer = ConfidenceAwareEvaluationObserver(evaluator, KitchenObject)
    results = evaluate_heavy_object_rule([NORMAL_PITCHER, NORMAL_POT], observer)
    assert len(results) == 2


def test_impossible_object_raises_during_evaluation(evaluator):
    """
    An impossible object raises an unfamiliar-sample exception when its rule fires.
    """
    observer = ConfidenceAwareEvaluationObserver(evaluator, KitchenObject)
    with pytest.raises(UnfamiliarSampleException):
        evaluate_heavy_object_rule([IMPOSSIBLE_CUP], observer)


def test_raised_exception_names_the_node_that_flagged_the_instance(evaluator):
    """
    The raised exception names the rule node that rejected the instance.
    """
    observer = ConfidenceAwareEvaluationObserver(evaluator, KitchenObject)
    with pytest.raises(UnfamiliarSampleException) as raised:
        evaluate_heavy_object_rule([IMPOSSIBLE_CUP], observer)
    assert raised.value.node_name
    assert raised.value.log_likelihood < raised.value.threshold