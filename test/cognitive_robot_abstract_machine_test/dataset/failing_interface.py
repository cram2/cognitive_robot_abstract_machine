"""
Stand-in for an ORM interface that fails to import for a reason other than no longer
matching the classes it maps.

Raises the way a mapping the installed packages cannot support would, rather than the
:class:`ImportError` or :class:`AttributeError` a stale interface raises.
"""

from __future__ import annotations

DIAGNOSTIC = "the mapped columns of this interface do not fit their table"
"""
What this interface raises with.
"""

raise ValueError(DIAGNOSTIC)
