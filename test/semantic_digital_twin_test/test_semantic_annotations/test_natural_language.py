from nltk.corpus import wordnet

from krrood.entity_query_language.factories import *
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageDescription,
    most_similar_synonym,
)


def test_natural_language_comparison():

    matching_annotation = NaturalLanguageDescription(root=None, description="Notebook")
    non_matching_annotation = NaturalLanguageDescription(root=None, description="Bank")
    annotations = [matching_annotation, non_matching_annotation]

    book = wordnet.synsets("Book")[1]

    annotation = variable_from(annotations)

    query = max(annotation, key=lambda a: most_similar_synonym(a.description, book)[0])
    result = query.tolist()[0]

    assert result is matching_annotation
