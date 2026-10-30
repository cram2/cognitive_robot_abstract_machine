"""
The rules on imports.
"""

from __future__ import annotations


from krrood.development_tools.agents.rule import Section


class ImportsRules(Section):
    """
    The rules on imports.
    """

    ABSOLUTE_IMPORTS = (
        "Imports are absolute. Exception: a test imports another test module (a "
        "shared mimic or fixture from the test datasets) relatively"
    )

    IMPORTS_AT_THE_TOP = (
        "Imports go at the top of the module; fix an import cycle by restructuring "
        "the modules, never with a local import"
    )

    GUARD_TYPE_ONLY_IMPORTS = "Guard type-only imports with `TYPE_CHECKING`"
