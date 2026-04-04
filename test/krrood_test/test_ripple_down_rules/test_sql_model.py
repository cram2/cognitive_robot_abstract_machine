import unittest

from sqlalchemy import select

from .datasets import *


class TestDDL(unittest.TestCase):

    session: sqlalchemy.orm.Session

    def setUp(self):
        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        self.session = sqlalchemy.orm.Session(engine)

    def test_animal(self):
        a1 = MappedAnimal(name="animal", hair=True, feathers=False, eggs=True, milk=False, airborne=False, aquatic=False, predator=False,
                          toothed=True, backbone=True, breathes=True, venomous=False, fins=False, legs=4, tail=True,
                          domestic=False, catsize=True, species=Species.mammal)
        # print(a1.habitats)
        # print(isinstance(a1.habitats, set))
        self.session.add(a1)
        self.session.commit()

        animals = self.session.scalars(select(MappedAnimal)).all()

        a1.habitats.add(HabitatTable(Habitat.water))
        # print(a1.habitats)
        self.session.commit()

        queried_a1 = self.session.scalars(select(HabitatTable)).first()
        # print(queried_a1)

