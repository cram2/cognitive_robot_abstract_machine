"""
A module whose ``TYPE_CHECKING`` block imports, with a plain ``import`` statement, a
module that is installed but whose own dependencies are not.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import test.krrood_test.dataset.installed_module_with_uninstalled_dependency
    from .latebound_annotation_type import LateBoundAnnotationType


@dataclass
class OwnerOfAnnotationNextToUnimportableModule:
    """
    Owner of a field whose annotation is importable, in a module that also names a
    module that cannot be imported.
    """

    value: Optional[LateBoundAnnotationType] = None
    """
    The annotation whose type resolves from this module's own ``TYPE_CHECKING`` imports.
    """
