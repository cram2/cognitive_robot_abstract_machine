"""
A module whose ``TYPE_CHECKING`` block imports from a part of ``krrood`` that is not
installed, while ``krrood`` itself is.

Mirrors a core module that names a module of another ``krrood.<part>`` distribution only
for type checking: on an install of krrood-core alone, that part is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import krrood.uninstalled_part.plain_imported_module
    from krrood.uninstalled_part.module import UninstalledPartType
    from .latebound_annotation_type import LateBoundAnnotationType


@dataclass
class OwnerOfAnnotationNextToUninstalledKrroodPart:
    """
    Owner of a field whose annotation is importable, in a module that also names a part
    of ``krrood`` that is not installed.
    """

    value: Optional[LateBoundAnnotationType] = None
    """
    The annotation whose type resolves from this module's own ``TYPE_CHECKING`` imports.
    """
