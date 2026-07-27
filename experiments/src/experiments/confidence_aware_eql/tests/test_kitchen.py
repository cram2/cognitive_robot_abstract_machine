from probabilistic_model.learning.jpt.variables import AnnotatedVariable
from random_events.variable import Continuous

from experiments.confidence_aware_eql.domains.kitchen import KitchenObject, Material
from experiments.confidence_aware_eql.engine.pipeline import ConfidenceModelBuilder
from experiments.confidence_aware_eql.engine.training import (
    ClusterPrototype,
    TrainingDataGenerator,
)

WEIGHT = Continuous("weight")
SIZE = Continuous("size")
MATERIAL = Continuous("material")


def kitchen_prototypes():
    """
    The clusters of familiar kitchen objects: a cup, a pitcher and a pot.
    """
    return [
        ClusterPrototype(
            [
                AnnotatedVariable(WEIGHT, mean=0.25, standard_deviation=0.05),
                AnnotatedVariable(SIZE, mean=0.10, standard_deviation=0.02),
                AnnotatedVariable(
                    MATERIAL, mean=Material.CERAMIC.value, standard_deviation=0.01
                ),
            ]
        ),
        ClusterPrototype(
            [
                AnnotatedVariable(WEIGHT, mean=2.50, standard_deviation=0.30),
                AnnotatedVariable(SIZE, mean=0.25, standard_deviation=0.03),
                AnnotatedVariable(
                    MATERIAL, mean=Material.GLASS.value, standard_deviation=0.01
                ),
            ]
        ),
        ClusterPrototype(
            [
                AnnotatedVariable(WEIGHT, mean=3.00, standard_deviation=0.40),
                AnnotatedVariable(SIZE, mean=0.30, standard_deviation=0.03),
                AnnotatedVariable(
                    MATERIAL, mean=Material.METAL.value, standard_deviation=0.01
                ),
            ]
        ),
    ]


def build_kitchen_evaluator():
    """
    Build an evaluator trained on the familiar kitchen object prototypes.
    """
    generator = TrainingDataGenerator(kitchen_prototypes())
    return ConfidenceModelBuilder(KitchenObject, generator).build()


def test_normal_object_is_familiar():
    """
    A typical ceramic cup is accepted as familiar.
    """
    evaluator = build_kitchen_evaluator()
    result = evaluator.check(
        KitchenObject(0.25, 0.10, Material.CERAMIC), node_name="is_graspable"
    )
    assert result.is_familiar


def test_impossible_cup_is_flagged():
    """
    A fifty kilogram glass cup is reported as unfamiliar.
    """
    evaluator = build_kitchen_evaluator()
    result = evaluator.check(
        KitchenObject(50.0, 0.10, Material.GLASS), node_name="is_heavy"
    )
    assert not result.is_familiar
    assert result.warning.node_name == "is_heavy"
    assert result.warning.log_likelihood < result.warning.threshold


def test_missing_material_is_marginalised_and_still_scored():
    """
    An object without a material is scored on its remaining features.
    """
    evaluator = build_kitchen_evaluator()
    result = evaluator.check(KitchenObject(0.25, 0.10, None), node_name="is_glass")
    assert result.is_familiar
    assert result.warning is None


def test_missing_material_with_impossible_weight_is_still_flagged():
    """
    Marginalising the material does not hide an impossible weight.
    """
    evaluator = build_kitchen_evaluator()
    result = evaluator.check(KitchenObject(50.0, 0.10, None), node_name="is_heavy")
    assert not result.is_familiar
