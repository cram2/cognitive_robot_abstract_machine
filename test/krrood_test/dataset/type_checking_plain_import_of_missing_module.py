"""
A module whose ``TYPE_CHECKING`` block imports, with a plain ``import`` statement, a
package that is not installed.

Mirrors a core module that names a library from an optional extra only for type
checking: the module itself imports and runs fine on an install without that extra.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import uninstalled_optional_dependency
    from .latebound_annotation_type import LateBoundAnnotationType


@dataclass
class OwnerOfAnnotationNextToUninstalledImport:
    """
    Owner of a field whose annotation is importable, in a module that also names an
    uninstalled package.
    """

    value: Optional[LateBoundAnnotationType] = None
    """
    The annotation whose type resolves from this module's own ``TYPE_CHECKING`` imports.
    """
