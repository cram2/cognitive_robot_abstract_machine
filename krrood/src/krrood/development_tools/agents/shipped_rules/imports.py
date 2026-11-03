"""
The rules on imports.
"""

from __future__ import annotations


from krrood.development_tools.agents.core_boundary import CoreBoundaryChecker
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

    CORE_LOADS_ONLY_CORE = (
        "A distribution that declares its core in `[tool.krrood.core-boundary]` of its "
        "`pyproject.toml` keeps it light: a core module imports a module outside core, "
        "or a library outside core's `dependencies` and what they require, only "
        "inside the functions that need it or under `TYPE_CHECKING`, never at module "
        "level. This is the one exception to importing at the top of the module",
        (),
        CoreBoundaryChecker(),
    )

    CHECK_THROUGH_PRE_COMMIT = (
        "Run `krrood check` over the distribution's source through the repository's "
        "pre-commit hook, so every rule a checker enforces holds for the code a commit "
        "changes"
    )
