import unittest

from krrood.entity_query_language.entitymatch import entity, let, and_, contains
from krrood.entity_query_language.entity_result_processors import an, the

from pycram.designator import EQLObjectDesignator, NamedObject
from pycram.designators.object_designator import *
from pycram.language import SequentialPlan
from pycram.testing import ApartmentWorldTestCase
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


class TestObjectDesignator(ApartmentWorldTestCase):

    def test_eql_designator(self):
        milk_desig = EQLObjectDesignator(
            an(
                entity(Body)(name=entity(PrefixedName)(name=contains("milk"))).from_(
                    self.world.bodies
                )
            )
        )
        found_milks = list(milk_desig)
        self.assertEqual(1, len(found_milks))
        self.assertEqual("milk.stl", found_milks[0].name.name)
        self.assertEqual(self.world.get_body_by_name("milk.stl"), found_milks[0])

    def test_named_object(self):
        named_desig = NamedObject("milk.stl")
        plan = SequentialPlan(self.context, named_desig)
        found_milks = list(named_desig)
        self.assertEqual(1, len(found_milks))
        self.assertEqual("milk.stl", found_milks[0].name.name)
        self.assertEqual(self.world.get_body_by_name("milk.stl"), found_milks[0])


if __name__ == "__main__":
    unittest.main()
