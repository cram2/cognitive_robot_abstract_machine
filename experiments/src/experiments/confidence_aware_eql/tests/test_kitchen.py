import pytest
from krrood.entity_query_language.factories import a
from probabilistic_model.learning.jpt.variables import AnnotatedVariable
from random_events.variable import Continuous

from experiments.confidence_aware_eql.domains.kitchen import KitchenObject, Material
from experiments.confidence_aware_eql.engine.pipeline import ConfidenceModelBuilder
from experiments.confidence_aware_eql.engine.training import FamiliarCluster

WEIGHT = Continuous("weight")
SIZE = Continuous("size")


def kitchen_clusters():
    """
    The familiar kitchen clusters: a ceramic cup, a glass pitcher and a metal pot.
    """
    return [
        FamiliarCluster(
            a(KitchenObject)(weight=..., size=..., material=Material.CERAMIC),
            [
                AnnotatedVariable(WEIGHT, mean=0.25, standard_deviation=0.05),
                AnnotatedVariable(SIZE, mean=0.10, standard_deviation=0.02),
            ],
        ),
        FamiliarCluster(
            a(KitchenObject)(weight=..., size=..., material=Material.GLASS),
            [
                AnnotatedVariable(WEIGHT, mean=2.50, standard_deviation=0.30),
                AnnotatedVariable(SIZE, mean=0.25, standard_deviation=0.03),
            ],
        ),
        FamiliarCluster(
            a(KitchenObject)(weight=..., size=..., material=Material.METAL),
            [
                AnnotatedVariable(WEIGHT, mean=3.00, standard_deviation=0.40),
                AnnotatedVariable(SIZE, mean=0.30, standard_deviation=0.03),
            ],
        ),
    ]


@pytest.fixture
def evaluator():
    """
    An evaluator learned from the familiar kitchen clusters.
    """
    return ConfidenceModelBuilder(KitchenObject, kitchen_clusters()).build()


def test_normal_object_is_familiar(evaluator):
    """
    A typical ceramic cup is accepted as familiar.
    """
    result = evaluator.check(
        KitchenObject(0.25, 0.10, Material.CERAMIC), node_name="is_graspable"
    )
    assert result.is_familiar


def test_impossible_cup_is_flagged(evaluator):
    """
    A fifty kilogram glass cup is reported as unfamiliar.
    """
    result = evaluator.check(
        KitchenObject(50.0, 0.10, Material.GLASS), node_name="is_heavy"
    )
    assert not result.is_familiar
    assert result.warning.node_name == "is_heavy"
    assert result.warning.log_likelihood < result.warning.threshold


def test_missing_material_is_marginalised_and_still_scored(evaluator):
    """
    An object without a material is scored on its remaining features.
    """
    result = evaluator.check(KitchenObject(0.25, 0.10, None), node_name="is_glass")
    assert result.is_familiar
    assert result.warning is None


def test_missing_material_with_impossible_weight_is_still_flagged(evaluator):
    """
    Marginalising the material does not hide an impossible weight.
    """
    result = evaluator.check(KitchenObject(50.0, 0.10, None), node_name="is_heavy")
    assert not result.is_familiar
