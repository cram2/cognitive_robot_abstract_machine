from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from itertools import takewhile
from types import NoneType
from typing import List, Set, Optional

import enchant
import rustworkx
import tqdm
from graphql.pyutils import cached_property

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


@dataclass(eq=False)
class Sage10kLabel(HasRootBody):
    """
    Represents a label in the Sage10k dataset annotation hierarchy.
    """


@dataclass
class Sage10kSemanticAnnotationCreator:
    """
    Creates semantic annotations for Sage10k dataset scenes.
    """

    raw_type_names: List[str]
    """
    The raw type names that should be converted to semantic annotation classes.
    """

    word_dictionary: enchant.Dict = field(default_factory=lambda: enchant.Dict("en_US"))
    """
    The word dictionary used to check if a type name is valid.
    """

    lemmatizer: WordNetLemmatizer = field(default_factory=lambda: WordNetLemmatizer())
    """
    The lemmatizer used to convert type names to singular form.
    """

    word_hierarchy: rustworkx.PyDiGraph[str, NoneType] = field(init=False)

    @cached_property
    def cleaned_type_names(self) -> Set[str]:

        cleaned_type_names = set()
        for type_name in self.raw_type_names:
            cleaned_type_name = self.clean_type_name(type_name)
            if cleaned_type_name is not None:
                cleaned_type_names.add(cleaned_type_name)

        return cleaned_type_names

    def clean_type_name(self, type_name: str) -> Optional[str]:
        cleaned_type = "".join(takewhile(str.isalpha, type_name))

        if not cleaned_type:
            return None
        if not self.word_dictionary.check(cleaned_type):
            return None

        singular = self.lemmatizer.lemmatize(cleaned_type, pos="n")

        # skip plural terms
        if singular != cleaned_type:
            return None

        return cleaned_type

    def _create_word_hierarchy(self):
        self.word_hierarchy = rustworkx.PyDiGraph(multigraph=False)
        for type_name in self.cleaned_type_names:
            self.word_hierarchy.add_node(type_name)

        for parent_index, child_index in tqdm.tqdm(
            itertools.product(
                self.word_hierarchy.node_indices(), self.word_hierarchy.node_indices()
            ),
            total=len(self.word_hierarchy.node_indices()) ** 2,
        ):
            if parent_index == child_index:
                continue

            parent_type = self.word_hierarchy.nodes()[parent_index]
            child_type = self.word_hierarchy.nodes()[child_index]

            if self.is_specialization_of(
                child_word=child_type, parent_word=parent_type
            ):
                self.word_hierarchy.add_edge(parent_index, child_index, None)

    @staticmethod
    def is_specialization_of(child_word: str, parent_word: str) -> bool:
        child_synsets = wordnet.synsets(child_word, pos=wordnet.NOUN)
        parent_synsets = wordnet.synsets(parent_word, pos=wordnet.NOUN)

        for child_synset in child_synsets:
            hypernyms = {
                hypernym for path in child_synset.hypernym_paths() for hypernym in path
            }

            for parent_synset in parent_synsets:
                if parent_synset in hypernyms:
                    return True

        return False
