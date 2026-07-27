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
from experiments.confidence_aware_eql.engine.pipeline import ConfidenceModelBuilder
from experiments.confidence_aware_eql.engine.training import TrainingDataGenerator
from experiments.confidence_aware_eql.tests.test_kitchen import kitchen_prototypes

NORMAL_PITCHER = KitchenObject(2.50, 0.25, Material.GLASS)
NORMAL_POT = KitchenObject(3.00, 0.30, Material.METAL)
IMPOSSIBLE_CUP = KitchenObject(50.0, 0.10, Material.GLASS)


@pytest.fixture
def evaluator():
    """
    An evaluator trained on the familiar kitchen object prototypes.
    """
    generator = TrainingDataGenerator(kitchen_prototypes())
    return ConfidenceModelBuilder(KitchenObject, generator).build()


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


def test_deterministic_result_is_unchanged_by_the_observer(evaluator):
    """
    Observing the evaluation does not alter the solutions of the rule.
    """
    observer = ConfidenceAwareEvaluationObserver(evaluator, KitchenObject)
    world = [NORMAL_PITCHER, NORMAL_POT, IMPOSSIBLE_CUP]
    assert len(evaluate_heavy_object_rule(world, observer)) == len(world)


def test_familiar_objects_raise_no_warning(evaluator):
    """
    Familiar objects pass every node without raising a warning.
    """
    observer = ConfidenceAwareEvaluationObserver(evaluator, KitchenObject)
    evaluate_heavy_object_rule([NORMAL_PITCHER, NORMAL_POT], observer)
    assert observer.warnings == []


def test_impossible_object_is_flagged_during_evaluation(evaluator):
    """
    An impossible object is flagged while the rule tree is evaluated.
    """
    observer = ConfidenceAwareEvaluationObserver(evaluator, KitchenObject)
    evaluate_heavy_object_rule([IMPOSSIBLE_CUP], observer)
    assert observer.warnings
    assert all(
        warning.log_likelihood < warning.threshold for warning in observer.warnings
    )


def test_warning_names_the_node_that_flagged_the_instance(evaluator):
    """
    The variable node of the queried class is named as a rejecting node.
    """
    observer = ConfidenceAwareEvaluationObserver(evaluator, KitchenObject)
    evaluate_heavy_object_rule([IMPOSSIBLE_CUP], observer)
    assert KitchenObject.__name__ in {
        warning.node_name for warning in observer.warnings
    }
