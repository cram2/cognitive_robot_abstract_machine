"""
kitchen_eql_demo.py — the demo wired into the ACTIVE KRROOD EQL.

Builds a kitchen grasping rule tree with the new EQL factories, then evaluates
it while the confidence observer flags anomalous objects per node. This mirrors
the repo's own rule-tree test style (test/krrood_test/test_eql/test_core/
test_rules.py) and the monitoring observer style.

NOTE (validate after install): the EQL factory calls below match the repo's
current API as read from its tests, but have not been executed here. Run this
after `uv sync --active`; if a name/signature differs, align it to the example
in test_rules.py — the structure (variable / entity().where() / refinement /
Add(inference(...))) is the same.
"""

from dataclasses import dataclass, field
from typing_extensions import List, Optional

                                                                            
from krrood.entity_query_language.predicate import Symbol
from krrood.entity_query_language.factories import (
    variable, deduced_variable, entity, an, refinement, inference,
)
from krrood.entity_query_language.rules.conclusion import Add

                                                                            
from .engine import build_evaluator
from .domains import kitchen
from .eql_integration import run_with_confidence


                                                                             
                                                                             
                                                                             
@dataclass(unsafe_hash=True)
class KitchenObject(Symbol):
    name: str
    weight: float = 0.0
    size: float = 0.0
    material: str = "unknown"


@dataclass(unsafe_hash=True)
class GraspChoice(Symbol):
    obj: Optional[KitchenObject] = None
    style: str = "one_handed"


@dataclass
class World(Symbol):
    id: int = 0
    objects: List[KitchenObject] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)


def object_extractor(expression, sources):
    """Pull the KitchenObject being evaluated out of the EQL sources.

    Placeholder logic: scan the binding values for a KitchenObject and return it
    as a feature dict. Finalise this once you can print a real `sources`.
    """
    if sources is None:
        return None
    candidates = []
    for attr in ("bindings", "values", "result"):
        v = getattr(sources, attr, None)
        if isinstance(v, dict):
            candidates.extend(v.values())
        elif v is not None:
            candidates.append(v)
    for c in candidates:
        if isinstance(c, KitchenObject):
            return {"weight": c.weight, "size": c.size, "material": c.material}
    return None


def main():
                                                    
    world = World(1, [
        KitchenObject("cup",            0.20, 0.10, "ceramic"),
        KitchenObject("pitcher",        2.50, 0.25, "glass"),
        KitchenObject("pot",            3.00, 0.30, "metal"),
        KitchenObject("impossible_cup", 50.0, 0.10, "glass"),
    ])

    obj = variable(KitchenObject, domain=world.objects)
    grasp = deduced_variable(GraspChoice)

                                                                             
                          
    query = an(entity(grasp).where(obj.weight >= 0.0))
    with query:
        Add(grasp, inference(GraspChoice)(obj=obj, style="one_handed"))
        with refinement(obj.weight > 2.0, obj.material == "glass"):
            Add(grasp, inference(GraspChoice)(obj=obj, style="two_handed"))

                                                                    
    evaluator, model, strategy, data = build_evaluator(kitchen.DOMAIN, kitchen.SPEC, seed=0)

                                                             
    results, warnings = run_with_confidence(query, evaluator, object_extractor)

    print("Grasp decisions:")
    for r in results:
        print(" ", r)

    print("\nConfidence warnings:")
    if not warnings:
        print("  (none)")
    for w in warnings:
        print(" ", w)


if __name__ == "__main__":
    main()
