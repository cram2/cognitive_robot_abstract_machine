from dataclasses import dataclass, field

from krrood.entity_query_language.predicate import symbolic_function
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from nltk.corpus import wordnet
from nltk.corpus.reader import Synset
from typing import Tuple, Callable


@dataclass(eq=False)
class NaturalLanguageDescription(HasRootBody):
    """
    Annotation for descriptions in natural language, e. g.
    `WordNet <https://www.nltk.org/howto/wordnet.html>`_ concepts.
    """

    description: str = field(kw_only=True)
    """
    The natural language description of root entity.
    """


@symbolic_function
def most_similar_synonym(
    word: str, target: Synset, similarity_measure: Callable = Synset.wup_similarity
) -> Tuple[float, Synset]:
    """
    Get the most similar synonym for a given word with respect to the target word.

    :param word: The word to find the best synonym for.
    :param target: The target word to compare against.
    :param similarity_measure: The similarity measure to use for comparison.
    :return: The best similarity score and the corresponding synonym.
    """
    best_similarity, best_synset = (0, None)
    for possible_synonym in wordnet.synsets(word, pos=wordnet.NOUN):
        similarity = similarity_measure(target, possible_synonym)
        if similarity > best_similarity:
            best_similarity = similarity
            best_synset = possible_synonym

    return best_similarity, best_synset
