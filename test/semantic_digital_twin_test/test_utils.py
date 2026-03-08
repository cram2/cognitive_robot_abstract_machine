from pathlib import Path

from random_events.utils import get_full_class_name

from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.utils import (
    type_string_to_type,
    get_path_to_project_root,
)


def test_type_string_to_string():
    original_class = Handle
    original_class_name = get_full_class_name(original_class)

    converted_class = type_string_to_type(original_class_name)

    assert converted_class == original_class


def test_get_project_root():
    expected_project_root = Path(__file__).resolve().parent.parent.parent
    project_root = get_path_to_project_root(Path(__file__).resolve())
    assert project_root == expected_project_root
