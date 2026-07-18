from experiments.confidence_aware_eql.domains.kitchen import KitchenObject, Material
from experiments.confidence_aware_eql.engine.pipeline import ConfidenceModelBuilder
from experiments.confidence_aware_eql.engine.training import (
    InstancePrototype,
    TrainingSampler,
)


def build_kitchen_evaluator():
    """Build an evaluator trained on the familiar kitchen object prototypes."""
    prototypes = [
        InstancePrototype(
            KitchenObject(0.25, 0.10, Material.CERAMIC),
            {"weight": 0.05, "size": 0.02},
        ),
        InstancePrototype(
            KitchenObject(2.50, 0.25, Material.GLASS),
            {"weight": 0.30, "size": 0.03},
        ),
        InstancePrototype(
            KitchenObject(3.00, 0.30, Material.METAL),
            {"weight": 0.40, "size": 0.03},
        ),
    ]
    sampler = TrainingSampler(KitchenObject, prototypes)
    return ConfidenceModelBuilder(KitchenObject, sampler).build()


def test_normal_object_is_familiar():
    """A typical ceramic cup is accepted as familiar."""
    evaluator = build_kitchen_evaluator()
    result = evaluator.check(
        KitchenObject(0.25, 0.10, Material.CERAMIC), node_name="is_graspable"
    )
    assert result.is_familiar


def test_impossible_cup_is_flagged():
    """A fifty kilogram glass cup is reported as unfamiliar."""
    evaluator = build_kitchen_evaluator()
    result = evaluator.check(
        KitchenObject(50.0, 0.10, Material.GLASS), node_name="is_heavy"
    )
    assert not result.is_familiar
    assert result.warning.node_name == "is_heavy"


def test_missing_material_is_flagged():
    """An object without a material is reported as incomplete."""
    evaluator = build_kitchen_evaluator()
    result = evaluator.check(
        KitchenObject(0.25, 0.10, None), node_name="is_glass"
    )
    assert not result.is_familiar
    assert "material" in result.warning.reason
