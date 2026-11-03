"""
The rules on imports.
"""

from __future__ import annotations

from krrood.development_tools.agents.rule import Section
from krrood.development_tools.agents.optional_imports import (
    OptionalImportChecker,
)


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

    IMPORT_OPTIONAL_LIBRARIES_WHERE_USED = (
        "A library whose distribution only an extra requires is optional. A core "
        "module - one of the source modules of the import-linter contract that "
        "`core-contract` in `[tool.krrood.core-boundary]` of the distribution's "
        "`pyproject.toml` names - imports it only inside the functions that need it, "
        "or under `TYPE_CHECKING`, never at module level. This is the one exception "
        "to importing at the top of the module",
        (),
        OptionalImportChecker(),
    )

    CHECK_OPTIONAL_IMPORTS = (
        "Run `krrood check` over the distribution's source through the repository's "
        "pre-commit hook, so no core module imports an optional library at module "
        "level"
    )
