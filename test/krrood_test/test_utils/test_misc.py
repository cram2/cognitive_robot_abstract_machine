from krrood.utils import is_dynamic_class


def test_is_dynamic_class():
    assert not is_dynamic_class(type)
    dynamic_class = type("DynamicClassForTesting", (), {})
    assert is_dynamic_class(dynamic_class)
