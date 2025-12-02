import os

from random_events.utils import get_full_class_name

import semantic_digital_twin.semantic_annotations.semantic_annotations
from semantic_digital_twin.utils import (
    get_semantic_digital_twin_directory_root,
    type_string_to_type,
)


def test_get_semantic_digital_twin_directory_root():
    path = os.path.abspath(__file__)
    root = get_semantic_digital_twin_directory_root(path)
    assert root == os.path.abspath(os.path.join(os.path.dirname(path), "..", ".."))


def test_type_string_to_string():
    original_class = (
        semantic_digital_twin.semantic_annotations.semantic_annotations.Handle
    )
    original_class_name = get_full_class_name(original_class)

    converted_class = type_string_to_type(original_class_name)

    assert converted_class == original_class
