from typing_extensions import get_type_hints

from krrood.class_diagrams.class_diagram import resolve_type
from ..dataset.classes_with_generic import FirstGeneric


def test_resolve_generic_type():
    resolved_hints = get_type_hints(FirstGeneric, include_extras=True)
    pass
    # Use the resolved hint if available, else fallback to the raw field type
    # raw_type = resolved_hints.get(f.name, f.type)
    # new_type = resolve_type(raw_type, {}, {})