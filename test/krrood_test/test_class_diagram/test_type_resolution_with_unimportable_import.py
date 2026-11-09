"""
Tests that type hints resolve when the module defining them names an import that cannot
be imported.
"""

from typing_extensions import Optional

from krrood.class_diagrams.utils import get_type_hints_of_object

from ..dataset.latebound_annotation_type import LateBoundAnnotationType
from ..dataset.subclass_importing_missing_module import SubclassImportingMissingModule
from ..dataset.type_checking_import_of_uninstalled_krrood_part import (
    OwnerOfAnnotationNextToUninstalledKrroodPart,
)
from ..dataset.type_checking_import_of_module_with_uninstalled_dependency import (
    OwnerOfAnnotationNextToUnimportableModule,
)
from ..dataset.type_checking_plain_import_of_missing_module import (
    OwnerOfAnnotationNextToUninstalledImport,
)

# %% subclass module importing a missing module


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


# %% uninstalled package named in a TYPE_CHECKING block


def test_annotation_resolves_when_its_module_names_an_uninstalled_package():
    """
    An annotation must resolve from its module's ``TYPE_CHECKING`` imports even when
    that block also imports a package that is not installed.
    """
    get_type_hints_of_object.cache_clear()

    type_hints = get_type_hints_of_object(OwnerOfAnnotationNextToUninstalledImport)

    assert type_hints["value"] == Optional[LateBoundAnnotationType]


def test_annotation_resolves_when_its_module_names_an_uninstalled_part_of_an_installed_package():
    """
    An annotation must resolve from its module's ``TYPE_CHECKING`` imports even when
    that block also imports from a subpackage that is not installed, inside a package
    that is.
    """
    get_type_hints_of_object.cache_clear()

    type_hints = get_type_hints_of_object(OwnerOfAnnotationNextToUninstalledKrroodPart)

    assert type_hints["value"] == Optional[LateBoundAnnotationType]


def test_annotation_resolves_when_its_module_names_a_module_whose_dependencies_are_missing():
    """
    An annotation must resolve from its module's ``TYPE_CHECKING`` imports even when
    that block also plainly imports an installed module whose own dependencies are
    missing.
    """
    get_type_hints_of_object.cache_clear()

    type_hints = get_type_hints_of_object(OwnerOfAnnotationNextToUnimportableModule)

    assert type_hints["value"] == Optional[LateBoundAnnotationType]
