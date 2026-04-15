"""
Internal utilities shared across llmr modules.
"""
from __future__ import annotations


def field_short_name(name: str) -> str:
    """Return the leaf component of a dotted field path.

    Strips everything up to and including the last ``.``, leaving only the
    final field name.  Used to normalise dotted slot names returned by the LLM
    (e.g. ``'grasp_description.grasp_type'``) to their leaf for dict lookups.

    Examples::

        field_short_name('grasp_description.grasp_type')  # -> 'grasp_type'
        field_short_name('arm')                            # -> 'arm'
    """
    return name.rsplit(".", 1)[-1] if "." in name else name


def slot_prompt_name(name: str, action_cls: type) -> str:
    """Strip only the root action-class prefix, preserving nested slot paths.

    ``name_from_variable_access_path`` on a Match attribute may include the
    action class name as a leading component (e.g. ``'PickUpAction.arm'`` or
    ``'PickUpAction.grasp_description.grasp_type'``).  This strips only that
    root prefix so nested paths like ``'grasp_description.grasp_type'`` are
    kept intact for the slot-filler prompt and LLM response matching.

    Examples::

        slot_prompt_name('PickUpAction.arm', PickUpAction)
            # -> 'arm'
        slot_prompt_name('PickUpAction.grasp_description.grasp_type', PickUpAction)
            # -> 'grasp_description.grasp_type'
        slot_prompt_name('arm', PickUpAction)
            # -> 'arm'
    """
    prefix = f"{action_cls.__name__}."
    return name[len(prefix):] if name.startswith(prefix) else name
