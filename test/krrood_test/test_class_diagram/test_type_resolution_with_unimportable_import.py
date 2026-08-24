from typing_extensions import Optional

from krrood.class_diagrams.utils import get_type_hints_of_object

from ..dataset.latebound_annotation_type import LateBoundAnnotationType
from ..dataset.subclass_importing_missing_module import SubclassImportingMissingModule


def test_annotation_resolves_from_a_base_when_the_subclass_module_import_is_missing():
    """
    An annotation inherited from a base must still resolve when the subclass's own
    module imports a module that does not exist.

    Resolving a late-bound annotation falls back to searching the hierarchy, building
    each class's module scope in turn. The subclass is reached first, so a module it
    cannot import used to end the search before the base that defines the name was ever
    consulted.
    """
    get_type_hints_of_object.cache_clear()

    type_hints = get_type_hints_of_object(SubclassImportingMissingModule)

    assert type_hints["value"] == Optional[LateBoundAnnotationType]
