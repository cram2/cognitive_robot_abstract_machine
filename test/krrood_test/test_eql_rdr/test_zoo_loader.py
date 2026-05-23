"""
Phase 1 tests: Animal dataclass + zoo loader.
"""

import unittest

from krrood.entity_query_language.factories import entity, variable

from .animal import Animal, Species
from .zoo_loader import load_zoo_animals

animals, targets = load_zoo_animals()


@unittest.skipIf(len(animals) == 0, "Failed to load zoo dataset")
class TestZooLoader(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(animals), 101)
        self.assertEqual(len(targets), 101)

    def test_species_is_underspecified_on_cases(self):
        # Cases carry no species; the target list holds ground truth separately.
        self.assertTrue(all(a.species is None for a in animals))

    def test_field_types(self):
        a = animals[0]
        self.assertIsInstance(a.hair, bool)
        self.assertIsInstance(a.legs, int)
        # legs is a count, not a boolean.
        self.assertTrue(any(x.legs > 1 for x in animals))

    def test_first_case_is_aardvark_mammal(self):
        aardvark = animals[0]
        self.assertEqual(aardvark.name, "aardvark")
        self.assertTrue(aardvark.hair)
        self.assertTrue(aardvark.milk)
        self.assertFalse(aardvark.feathers)
        self.assertEqual(aardvark.legs, 4)
        self.assertEqual(targets[0], Species.mammal)

    def test_all_targets_are_species(self):
        self.assertTrue(all(isinstance(t, Species) for t in targets))
        self.assertEqual({t for t in targets}, set(Species))

    def test_plain_animal_works_as_eql_domain(self):
        # The plain dataclass needs no special treatment to be queried by EQL.
        animal = variable(Animal, domain=animals)
        hairy = list(entity(animal).where(animal.hair == True).evaluate())
        expected = sum(1 for a in animals if a.hair)
        self.assertEqual(len(hairy), expected)
        self.assertTrue(all(a.hair for a in hairy))

    def test_animal_is_plain_dataclass(self):
        # No EQL/ORM base classes — just a dataclass.
        a = Animal(
            name="x",
            hair=True,
            feathers=False,
            eggs=False,
            milk=True,
            airborne=False,
            aquatic=False,
            predator=False,
            toothed=True,
            backbone=True,
            breathes=True,
            venomous=False,
            fins=False,
            legs=4,
            tail=True,
            domestic=False,
            catsize=True,
        )
        self.assertIsNone(a.species)
        a.species = Species.mammal
        self.assertEqual(a.species, Species.mammal)


if __name__ == "__main__":
    unittest.main()
