"""
A module whose ``TYPE_CHECKING`` block imports from a submodule that its package finds
only through the path it adds to itself.

The import is absolute because a package's own submodules are named absolutely from
other distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from test.krrood_test.dataset.package_extending_its_path.type_found_through_an_extended_path import (
        TypeFoundThroughAnExtendedPath,
    )


@dataclass
class OwnerOfAnnotationFromAnExtendedPackagePath:
    """
    Owner of a field whose annotation names a type found through an extended package
    path.
    """

    value: Optional[TypeFoundThroughAnExtendedPath] = None
    """
    The annotation whose type resolves from this module's own ``TYPE_CHECKING`` imports.
    """
