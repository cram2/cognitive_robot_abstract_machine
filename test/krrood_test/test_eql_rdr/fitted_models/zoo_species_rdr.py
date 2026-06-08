"""Auto-generated EQL-RDR rule tree. Do not edit by hand."""
from krrood.entity_query_language.factories import (
    variable,
    entity,
    add,
    refinement,
    alternative,
    next_rule,
    and_,
    or_,
    not_,
)
from test.krrood_test.test_eql_rdr.animal import Animal, Species

animal = variable(Animal, domain=[])
query = entity(animal).where(animal.milk)
with query:
    add(animal.species, Species.mammal)
    with alternative(animal.aquatic):
        add(animal.species, Species.fish)
        with refinement(animal.backbone == False):
            add(animal.species, Species.molusc)
        with refinement(animal.feathers):
            add(animal.species, Species.bird)
        with refinement(animal.fins == False):
            add(animal.species, Species.amphibian)
            with refinement(animal.tail):
                add(animal.species, Species.reptile)
                with refinement(animal.legs > 0):
                    add(animal.species, Species.amphibian)
    with alternative(animal.feathers):
        add(animal.species, Species.bird)
    with alternative(animal.backbone == False):
        add(animal.species, Species.molusc)
        with refinement(animal.legs > 0):
            add(animal.species, Species.insect)
            with refinement(animal.eggs == False):
                add(animal.species, Species.molusc)
    with alternative(animal.tail):
        add(animal.species, Species.reptile)
query.build()

# Stable handles for loading.
RDR_CASE_TYPE = Animal
RDR_CONCLUSION_ATTRIBUTE = "species"
RDR_CASE_VARIABLE = animal
RDR_QUERY = query
RDR_CORNER_CASES = {0: Animal(name='aardvark', hair=True, feathers=False, eggs=False, milk=True, airborne=False, aquatic=False, predator=True, toothed=True, backbone=True, breathes=True, venomous=False, fins=False, legs=4, tail=False, domestic=False, catsize=True, species=None), 1: Animal(name='bass', hair=False, feathers=False, eggs=True, milk=False, airborne=False, aquatic=True, predator=True, toothed=True, backbone=True, breathes=False, venomous=False, fins=True, legs=0, tail=True, domestic=False, catsize=False, species=None), 2: Animal(name='crab', hair=False, feathers=False, eggs=True, milk=False, airborne=False, aquatic=True, predator=True, toothed=False, backbone=False, breathes=False, venomous=False, fins=False, legs=4, tail=False, domestic=False, catsize=False, species=None), 3: Animal(name='duck', hair=False, feathers=True, eggs=True, milk=False, airborne=True, aquatic=True, predator=False, toothed=False, backbone=True, breathes=True, venomous=False, fins=False, legs=2, tail=True, domestic=False, catsize=False, species=None), 4: Animal(name='frog', hair=False, feathers=False, eggs=True, milk=False, airborne=False, aquatic=True, predator=True, toothed=True, backbone=True, breathes=True, venomous=False, fins=False, legs=4, tail=False, domestic=False, catsize=False, species=None), 5: Animal(name='seasnake', hair=False, feathers=False, eggs=False, milk=False, airborne=False, aquatic=True, predator=True, toothed=True, backbone=True, breathes=False, venomous=True, fins=False, legs=0, tail=True, domestic=False, catsize=False, species=None), 6: Animal(name='newt', hair=False, feathers=False, eggs=True, milk=False, airborne=False, aquatic=True, predator=True, toothed=True, backbone=True, breathes=True, venomous=False, fins=False, legs=4, tail=True, domestic=False, catsize=False, species=None), 7: Animal(name='duck', hair=False, feathers=True, eggs=True, milk=False, airborne=True, aquatic=True, predator=False, toothed=False, backbone=True, breathes=True, venomous=False, fins=False, legs=2, tail=True, domestic=False, catsize=False, species=None), 8: Animal(name='clam', hair=False, feathers=False, eggs=True, milk=False, airborne=False, aquatic=False, predator=True, toothed=False, backbone=False, breathes=False, venomous=False, fins=False, legs=0, tail=False, domestic=False, catsize=False, species=None), 9: Animal(name='flea', hair=False, feathers=False, eggs=True, milk=False, airborne=False, aquatic=False, predator=False, toothed=False, backbone=False, breathes=True, venomous=False, fins=False, legs=6, tail=False, domestic=False, catsize=False, species=None), 10: Animal(name='scorpion', hair=False, feathers=False, eggs=False, milk=False, airborne=False, aquatic=False, predator=True, toothed=False, backbone=False, breathes=True, venomous=True, fins=False, legs=8, tail=True, domestic=False, catsize=False, species=None), 11: Animal(name='seasnake', hair=False, feathers=False, eggs=False, milk=False, airborne=False, aquatic=True, predator=True, toothed=True, backbone=True, breathes=False, venomous=True, fins=False, legs=0, tail=True, domestic=False, catsize=False, species=None)}