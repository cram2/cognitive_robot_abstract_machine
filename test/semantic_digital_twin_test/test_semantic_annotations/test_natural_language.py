from nltk.corpus.reader import Synset

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import *
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from nltk.corpus import wordnet


def test_natural_language_comparison():

    @symbolic_function
    def max_similarity(target: Synset, sample: str) -> Tuple[float, Synset]:
        best_similarity, best_synset = (0, None)
        for synset in wordnet.synsets(sample):
            similarity = target.wup_similarity(synset)
            if similarity > best_similarity:
                best_similarity = similarity
                best_synset = synset
        return best_similarity, best_synset

    matching_annotation = NaturalLanguageDescription(root=None, description="Notebook")
    non_matching_annotation = NaturalLanguageDescription(root=None, description="Bank")
    annotations = [matching_annotation, non_matching_annotation]

    book = wordnet.synsets("Book")[1]

    annotation = variable_from(annotations)

    query = max(annotation, key=lambda a: max_similarity(book, a.description)[0])
    result = query.tolist()[0]

    assert result is matching_annotation
