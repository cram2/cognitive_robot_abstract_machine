import krrood.entity_query_language
from krrood.ormatic.utils import classes_of_module, classes_of_package


def test_classes_of_package():
    classes = classes_of_package(krrood.entity_query_language)
    assert len(classes) > 50


def test_classes_of_package_with_plain_module():
    """classes_of_package should handle a plain module (no __path__) gracefully."""
    import krrood.exceptions

    assert not hasattr(krrood.exceptions, "__path__")
    result = classes_of_package(krrood.exceptions)
    expected = classes_of_module(krrood.exceptions)
    assert result == expected
