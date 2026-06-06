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
query = entity(animal).where(animal.milk == True)
with query:
    add(animal.species, Species.mammal)
    with alternative(animal.aquatic == True):
        add(animal.species, Species.fish)
        with refinement(animal.fins == False):
            add(animal.species, Species.molusc)
            with refinement(animal.feathers == True):
                add(animal.species, Species.bird)
            with refinement(animal.legs > 0):
                add(animal.species, Species.amphibian)
                with refinement(animal.backbone == False):
                    add(animal.species, Species.molusc)
            with refinement(animal.tail == True):
                add(animal.species, Species.reptile)
    with alternative(animal.feathers == True):
        add(animal.species, Species.bird)
    with alternative(animal.backbone == False):
        add(animal.species, Species.molusc)
        with refinement(animal.eggs == True):
            add(animal.species, Species.insect)
            with refinement(animal.legs == 0):
                add(animal.species, Species.molusc)
    with alternative(animal.tail == True):
        add(animal.species, Species.reptile)
query.build()

# Stable handles for loading.
RDR_CASE_TYPE = Animal
RDR_CONCLUSION_ATTRIBUTE = "species"
RDR_CASE_VARIABLE = animal
RDR_QUERY = query