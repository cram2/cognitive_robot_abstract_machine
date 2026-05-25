"""
Phase 3 tests: live rule-tree growth via insert_at / insert_refinement / insert_alternative.

Growing a tree dynamically (outside any ``with`` block) must produce the same
classifications as building the equivalent tree statically with ``with`` blocks.
"""

import unittest

from krrood.entity_query_language.factories import (
    add,
    alternative,
    entity,
    variable,
)
from krrood.entity_query_language.rdr.observer import classify_case
from krrood.entity_query_language.rdr.rule_tree import (
    insert_alternative,
    insert_refinement,
)
from krrood.entity_query_language.rdr.utils import UNSET

from .animal import Animal, Species
from .zoo_loader import load_zoo_animals

animals, targets = load_zoo_animals()


def first(sp: Species) -> Animal:
    return next(a for a, t in zip(animals, targets) if t is sp)


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestRuleTreeGrowth(unittest.TestCase):
    def test_dynamic_alternative_insertion(self):
        animal = variable(Animal, domain=[])
        query = entity(animal).where(animal.milk == True)
        with query:
            add(animal.species, Species.mammal)
        query.build()

        # Bird does not fire yet.
        self.assertTrue(
            classify_case(query, animal, animal.species, first(Species.bird)).conclusion
            is UNSET
        )

        insert_alternative(
            query._conditions_root_,
            animal.feathers == True,
            animal.species,
            Species.bird,
        )

        self.assertEqual(
            classify_case(
                query, animal, animal.species, first(Species.bird)
            ).conclusion,
            Species.bird,
        )
        # Existing rule unaffected.
        self.assertEqual(
            classify_case(
                query, animal, animal.species, first(Species.mammal)
            ).conclusion,
            Species.mammal,
        )

    def test_dynamic_refinement_override(self):
        animal = variable(Animal, domain=[])
        query = entity(animal).where(animal.backbone == True)
        with query:
            add(animal.species, Species.fish)  # default guess for vertebrates
        query.build()

        # Before refinement, a mammal is misclassified as fish.
        self.assertEqual(
            classify_case(
                query, animal, animal.species, first(Species.mammal)
            ).conclusion,
            Species.fish,
        )

        insert_refinement(
            query._conditions_root_,
            animal.milk == True,
            animal.species,
            Species.mammal,
        )

        # Refinement overrides for mammals; fish still classified as fish.
        self.assertEqual(
            classify_case(
                query, animal, animal.species, first(Species.mammal)
            ).conclusion,
            Species.mammal,
        )
        self.assertEqual(
            classify_case(
                query, animal, animal.species, first(Species.fish)
            ).conclusion,
            Species.fish,
        )

    def test_dynamic_growth_matches_static_tree(self):
        # Static tree: milk->mammal ; alt feathers->bird ; alt fins->fish
        s_animal = variable(Animal, domain=[])
        s_query = entity(s_animal).where(s_animal.milk == True)
        with s_query:
            add(s_animal.species, Species.mammal)
            with alternative(s_animal.feathers == True):
                add(s_animal.species, Species.bird)
            with alternative(s_animal.fins == True):
                add(s_animal.species, Species.fish)
        s_query.build()

        # Dynamic tree: same rules grown one at a time.
        d_animal = variable(Animal, domain=[])
        d_query = entity(d_animal).where(d_animal.milk == True)
        with d_query:
            add(d_animal.species, Species.mammal)
        d_query.build()
        insert_alternative(
            d_query._conditions_root_,
            d_animal.feathers == True,
            d_animal.species,
            Species.bird,
        )
        insert_alternative(
            d_query._conditions_root_,
            d_animal.fins == True,
            d_animal.species,
            Species.fish,
        )

        for case, target in zip(animals, targets):
            static = classify_case(s_query, s_animal, s_animal.species, case).conclusion
            dynamic = classify_case(
                d_query, d_animal, d_animal.species, case
            ).conclusion
            self.assertEqual(
                static,
                dynamic,
                f"{case.name}: static={static} dynamic={dynamic}",
            )

    def test_refinement_then_alternative_nested(self):
        # Grow: backbone->fish ; refine milk->mammal ; alt(of refinement) aquatic->(stay mammal? no)
        animal = variable(Animal, domain=[])
        query = entity(animal).where(animal.backbone == True)
        with query:
            add(animal.species, Species.fish)
        query.build()

        ref = insert_refinement(
            query._conditions_root_,
            animal.milk == True,
            animal.species,
            Species.mammal,
        )
        # Alternative to the refinement: backbone & feathers -> bird
        insert_alternative(ref, animal.feathers == True, animal.species, Species.bird)

        self.assertEqual(
            classify_case(
                query, animal, animal.species, first(Species.mammal)
            ).conclusion,
            Species.mammal,
        )
        self.assertEqual(
            classify_case(
                query, animal, animal.species, first(Species.bird)
            ).conclusion,
            Species.bird,
        )
        self.assertEqual(
            classify_case(
                query, animal, animal.species, first(Species.fish)
            ).conclusion,
            Species.fish,
        )


if __name__ == "__main__":
    unittest.main()
