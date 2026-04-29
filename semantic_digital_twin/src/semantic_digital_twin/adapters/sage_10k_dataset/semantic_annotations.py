from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import takewhile
from typing import Optional

import enchant

from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)

logger = logging.getLogger(__name__)

from nltk.stem import WordNetLemmatizer


@dataclass(eq=False)
class NaturalLanguageDescriptionWithTypeDescription(NaturalLanguageDescription):
    """
    A natural language description of a Sage10k object including the type information of the object.
    """

    type_description: Optional[str] = field(default=None)
    """
    The cleaned description of the type of the object.
    """


@dataclass
class Sage10kTypeNameCleaner:
    """
    Clean type names from the Sage10k dataset.
    """

    word_dictionary: enchant.Dict = field(default_factory=lambda: enchant.Dict("en_US"))
    """
    The word dictionary used to check if a type name is valid.
    """

    lemmatizer: WordNetLemmatizer = field(default_factory=lambda: WordNetLemmatizer())
    """
    The lemmatizer used to convert type names to singular form.
    """

    def clean(self, type_name: str) -> Optional[str]:
        """
        Clean a type name from the sage 10k dataset.
        Types are cleaned by removing non-alphabetic characters.
        Valid type names are types that are in:
            - Singular form
            - The word dictionary

        :param type_name: The type name to clean.
        :return: The cleaned type name or None if the type name is invalid.
        """

        cleaned_type = " ".join(
            "".join(takewhile(str.isalpha, word)) for word in type_name.split("_")
        ).title()

        if not cleaned_type:
            return None
        if not self.word_dictionary.check(cleaned_type):
            return None

        singular = self.lemmatizer.lemmatize(cleaned_type, pos="n")

        # skip plural terms
        if singular != cleaned_type:
            return None

        return cleaned_type
