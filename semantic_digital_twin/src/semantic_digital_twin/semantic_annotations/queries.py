from semantic_digital_twin.semantic_annotations.semantic_annotations import Cola, Fanta, Water, Beer
from semantic_digital_twin.world_description.world_entity import Human
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


# Ralf = Human(name=PrefixedName("Ralf"), favourite_drink=Fanta)
# Rolf = Human(name=PrefixedName("Rolf"), favourite_drink=Cola)
# Rody = Human(name=PrefixedName("Ralf"), favourite_drink=Beer)


def query_favourite_drink(Human): # -> Optional[type]:
    """
    Resolve `person` to a Human in `world` and return their favourite drink class (subclass of Drink),
    or None if not found / not set.
    """
    return Human.favourite_drink