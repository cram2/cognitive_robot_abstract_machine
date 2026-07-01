from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import FrozenSet


class Countability(StrEnum):
    """Whether a noun denotes discrete instances (countable) or an undifferentiated mass."""

    COUNTABLE = "countable"
    """A noun that pluralises and takes an article — *"a battery"* / *"the battery"*."""
    UNCOUNTABLE = "uncountable"
    """A mass noun that resists the indefinite article and, in non-anaphoric use, the definite one —
    *"money"*, not *"a money"* / *"the money"* when first mentioned generically."""


_UNCOUNTABLE_NOUNS: FrozenSet[str] = frozenset(
    {
        # Materials and substances
        "water", "milk", "blood", "oxygen", "air", "gasoline", "gas", "petrol", "fuel",
        "oil", "electricity", "energy", "sand", "salt", "sugar", "rice", "flour",
        "wood", "metal", "steel", "plastic", "leather", "wool", "cotton", "silk",
        "nylon", "rubber", "concrete", "cement", "clay", "marble", "chalk", "coal",
        "smoke", "steam", "ice", "snow", "rain", "dust", "dirt", "mud", "mist", "fog",
        "foam", "ash", "wax", "glue", "soap", "shampoo", "toothpaste", "perfume",
        "paint", "ink", "plaster", "gravel",
        # Food and drink
        "coffee", "tea", "juice", "wine", "beer", "bread", "butter", "cheese", "meat",
        "pasta", "cereal", "soup", "honey", "jam", "cream", "yogurt", "chocolate",
        "candy", "ketchup", "mustard", "mayonnaise", "vinegar", "pepper", "corn",
        "wheat", "dough", "bacon",
        # Abstractions and states
        "money", "cash", "wealth", "poverty", "love", "hate", "anger", "fear", "joy",
        "happiness", "sadness", "grief", "courage", "patience", "honesty", "kindness",
        "generosity", "freedom", "justice", "beauty", "truth", "wisdom", "intelligence",
        "confidence", "pride", "luck", "fun", "fame", "strength", "health", "peace",
        "violence", "guilt", "innocence", "anxiety", "stress", "comfort", "satisfaction",
        "motivation", "creativity", "curiosity", "importance", "nonsense", "progress",
        # Information and study
        "information", "knowledge", "evidence", "advice", "news", "research", "homework",
        "housework", "training", "education", "advertising", "marketing",
        # Activities, services, and collective masses
        "music", "travel", "transport", "transportation", "accommodation", "parking",
        "leisure", "entertainment", "traffic", "weather", "employment", "unemployment",
        "behaviour", "behavior", "conduct",
        "furniture", "equipment", "machinery", "hardware", "software", "gear",
        "luggage", "baggage", "clothing", "footwear", "underwear", "jewellery",
        "jewelry", "makeup", "cutlery", "crockery", "stationery", "mail", "postage",
        "garbage", "rubbish", "trash", "litter", "waste", "poultry", "wildlife",
        "vegetation", "foliage", "scenery",
    }
)
"""Common English mass nouns, lower-cased. Curated rather than inferred: countability is a stable
lexical property, so a small audited set is more reliable than a runtime heuristic over schema-derived
attribute names. Nouns with a dominant countable sense (*a glass*, *a paper*, *a work*) are
deliberately excluded so they keep their article."""


@dataclass(frozen=True)
class NounCountability:
    """Classifies a noun as countable or uncountable against a curated mass-noun lexicon."""

    uncountable_nouns: FrozenSet[str] = _UNCOUNTABLE_NOUNS
    """The mass-noun lexicon consulted; overridable so a domain may extend it."""

    def classify(self, noun: str) -> Countability:
        """:return: the countability of *noun* (case-insensitive lookup).

        >>> NounCountability().classify("money")
        <Countability.UNCOUNTABLE: 'uncountable'>
        >>> NounCountability().classify("battery")
        <Countability.COUNTABLE: 'countable'>
        """
        if noun.lower() in self.uncountable_nouns:
            return Countability.UNCOUNTABLE
        return Countability.COUNTABLE

    def is_uncountable(self, noun: str) -> bool:
        """:return: whether *noun* is a mass noun.

        >>> NounCountability().is_uncountable("water")
        True
        >>> NounCountability().is_uncountable("department")
        False
        """
        return self.classify(noun) is Countability.UNCOUNTABLE
