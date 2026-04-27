from nltk.corpus.reader import Synset

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import *
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
)
from nltk.corpus import wordnet


def test_natural_language_comparison():

    @symbolic_function
    def similarity_key(target: Synset, sample: str):
        sample_synset = wordnet.synsets(sample)[0]
        return target.path_similarity(sample_synset)

    book_annotation = NaturalLanguageDescription(root=None, description="Notebook")
    bank_annotation = NaturalLanguageDescription(root=None, description="Bank")
    annotations = [book_annotation, bank_annotation]

    book = wordnet.synsets("Book")[0]

    annotation = variable_from(annotations)

    query = max(annotation, key=lambda a: similarity_key(book, a.description))
    result = query.tolist()[0]
    assert result is book_annotation
