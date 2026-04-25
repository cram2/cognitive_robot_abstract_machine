from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from itertools import takewhile
from types import NoneType
from typing import List, Set, Optional, Dict, Type, Tuple

import enchant
import rustworkx
import tqdm
from enum import Enum
from graphql.pyutils import cached_property

from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasSupportingSurface,
    HasStorageSpace,
    IsPerceivable,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Furniture,
    Table,
    Chair,
    DrinkingContainer,
    CookingContainer,
    Food,
)
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
        """
        Creates the word hierarchy graph by identifying IS_A relationships.
        Applies transitive reduction to the IS_A relationships to simplify the hierarchy.
        """
        word_hierarchy = rustworkx.PyDiGraph(multigraph=False)
        for type_name in self.cleaned_type_names:
            word_hierarchy.add_node(type_name)

        for parent_index, child_index in tqdm.tqdm(
            itertools.product(
                word_hierarchy.node_indices(), word_hierarchy.node_indices()
            ),
            total=len(word_hierarchy.node_indices()) ** 2,
        ):
            if parent_index == child_index:
                continue

            parent_type = word_hierarchy.nodes()[parent_index]
            child_type = word_hierarchy.nodes()[child_index]

            if self.is_specialization_of(
                child_word=child_type, parent_word=parent_type
            ):
                word_hierarchy.add_edge(parent_index, child_index, None)

        cycles = list(rustworkx.simple_cycles(word_hierarchy))
        for cycle in cycles:
            print("Cycle detected:", [word_hierarchy.nodes()[node] for node in cycle])

        # Apply Transitive Reduction for IS_A relationships
        reduced_is_a, _ = rustworkx.transitive_reduction(word_hierarchy)

        # Reconstruct word_hierarchy with reduced IS_A edges
        cleaned_hierarchy = rustworkx.PyDiGraph(multigraph=False)
        for node in self.word_hierarchy.nodes():
            cleaned_hierarchy.add_node(node)

        for p, c in reduced_is_a.edge_list():
            cleaned_hierarchy.add_edge(p, c, None)

        self.word_hierarchy = cleaned_hierarchy

    def is_specialization_of(self, child_word: str, parent_word: str) -> bool:
        """
        Determines if a word is a specialization (hyponym) of another word.
        """
        child_synsets = self._get_relevant_synsets(child_word)
        parent_synsets = self._get_relevant_synsets(parent_word)

        for child_synset in child_synsets:
            # Get all hypernyms excluding the synset itself to avoid cycles
            # between words that share the same synsets (e.g., bench and workbench)
            hypernyms = {
                hypernym
                for path in child_synset.hypernym_paths()
                for hypernym in path
                if hypernym != child_synset
            }

            for parent_synset in parent_synsets:
                if parent_synset in hypernyms:
                    return True

        return False

    def _get_relevant_synsets(self, word: str) -> List[wordnet.Synset]:
        """
        Returns synsets for a word that are relevant to the domain of physical entities.
        """
        synsets = wordnet.synsets(word, pos=wordnet.NOUN)
        if not synsets:
            return []

        relevant_synsets = []
        root_synset = wordnet.synset("physical_entity.n.01")

        for synset in synsets:

            # Check if it's a descendant of any root synsets
            all_hypernyms = {h for path in synset.hypernym_paths() for h in path}
            if root_synset in all_hypernyms:
                relevant_synsets.append(synset)

        return relevant_synsets
