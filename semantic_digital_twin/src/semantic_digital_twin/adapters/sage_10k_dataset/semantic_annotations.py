from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field, make_dataclass
from itertools import takewhile
from pathlib import Path
from typing import List, Set, Optional

import enchant
import rustworkx
import tqdm
from graphql.pyutils import cached_property

from krrood.class_diagrams.module_generation import ModuleRenderer

logger = logging.getLogger(__name__)

from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


@dataclass(eq=False)
class Sage10kLabel(HasRootBody):
    """
    Represents a label in the Sage10k dataset annotation hierarchy.
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


@dataclass
class Sage10kSemanticAnnotationCreator:
    """
    Creates semantic annotations for Sage10k dataset scenes.
    """

    raw_type_names: List[str]
    """
    The raw type names that should be converted to semantic annotation classes.
    """

    type_name_cleaner: Sage10kTypeNameCleaner = field(
        default_factory=Sage10kTypeNameCleaner
    )
    """
    The type cleaner used to clean the type names.
    """

    annotation_prefix: str = "Sage10k"
    """
    The prefix for the generated semantic annotations.
    """

    @cached_property
    def cleaned_type_names(self) -> Set[str]:

        cleaned_type_names = set()
        for type_name in self.raw_type_names:
            cleaned_type_name = self.type_name_cleaner.clean(type_name)
            if cleaned_type_name is not None:
                cleaned_type_names.add(cleaned_type_name)

        return cleaned_type_names

    @cached_property
    def word_hierarchy(self) -> rustworkx.PyDiGraph:
        """
        Creates the word hierarchy graph by identifying IS_A relationships.
        Applies transitive reduction to the IS_A relationships to simplify the hierarchy.
        """
        word_hierarchy = rustworkx.PyDiGraph(multigraph=False, check_cycle=True)
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
                try:
                    word_hierarchy.add_edge(parent_index, child_index, None)
                except rustworkx.DAGWouldCycle:
                    logger.warning(
                        f"Adding edge between {parent_type} and {child_type} would create a cycle. Skipping."
                    )
                    continue

        # Apply Transitive Reduction for IS_A relationships
        reduced_is_a, _ = rustworkx.transitive_reduction(word_hierarchy)

        # Reconstruct word_hierarchy with reduced IS_A edges
        cleaned_hierarchy = rustworkx.PyDiGraph(multigraph=False)
        for node in word_hierarchy.nodes():
            cleaned_hierarchy.add_node(node)

        for p, c in reduced_is_a.edge_list():
            cleaned_hierarchy.add_edge(p, c, None)

        return cleaned_hierarchy

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

    def write_annotations_to_file(self, output_file: Optional[Path] = None):
        """
        Write the annotations to a file.

        :param output_file: The path to write the annotations to.
        """

        if output_file is None:
            output_file = Path(__file__).parent / "generated_semantic_annotations.py"

        name_to_dataclass = {}

        for type_name_index in rustworkx.topological_sort(self.word_hierarchy):
            type_name = self.word_hierarchy[type_name_index]

            bases = self.word_hierarchy.predecessors(type_name_index)

            if not bases:
                bases = (Sage10kLabel,)
            else:
                bases = tuple(name_to_dataclass[node] for node in bases)

            dataclass_ = make_dataclass(
                cls_name=f"{self.annotation_prefix}{type_name}",
                bases=bases,
                fields=[],
            )
            name_to_dataclass[type_name] = dataclass_

        renderer = ModuleRenderer.from_dataclasses(name_to_dataclass.values())
        renderer.write_to_file(output_file)
