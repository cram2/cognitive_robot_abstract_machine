import krrood.entity_query_language
from krrood.ormatic.utils import classes_of_package


def test_classes_of_package():
    classes = classes_of_package(krrood.entity_query_language)
    assert len(classes) > 50
