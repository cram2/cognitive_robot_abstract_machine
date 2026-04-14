"""
Internal utilities shared across llmr modules.
"""
from __future__ import annotations


def field_short_name(name: str) -> str:
    """Strip the 'ClassName.' prefix from an EQL access-path name.

    ``name_from_variable_access_path`` on a Match attribute returns the last
    element of the EQL path, which may be prefixed with the action class name
    (e.g. ``'PickUpAction.arm'``).  Downstream code — introspector, slot filler,
    and backend — all work with bare field names (``'arm'``), so strip the prefix.

    Examples::

        field_short_name('PickUpAction.arm')          # -> 'arm'
        field_short_name('grasp_description.grasp_type')  # -> 'grasp_type'
        field_short_name('arm')                       # -> 'arm'
    """
    return name.rsplit(".", 1)[-1] if "." in name else name
