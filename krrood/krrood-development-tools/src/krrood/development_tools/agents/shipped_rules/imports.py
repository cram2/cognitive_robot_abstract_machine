"""
The rules on imports.
"""

from __future__ import annotations


from krrood.development_tools.agents.core_boundary import CoreBoundaryChecker
from krrood.development_tools.agents.rule import Section
from krrood.development_tools.agents.ruff_rules import (
    CheckedModules,
    RuffRule,
    RuffRuleChecker,
    RuffSetting,
)


class ImportsRules(Section):
    """
    The rules on imports.
    """

    ABSOLUTE_IMPORTS = (
        "Imports are absolute. Exception: a test imports another test module (a "
        "shared mimic or fixture from the test datasets) relatively",
        (),
        RuffRuleChecker(
            lint_rules=(RuffRule.RELATIVE_IMPORTS,),
            settings=(RuffSetting.RELATIVE_IMPORTS_BANNED,),
            modules=CheckedModules.OUTSIDE_TESTS,
        ),
    )

    IMPORTS_AT_THE_TOP = (
        "Imports go at the top of the module, before any other code",
        (),
        RuffRuleChecker(lint_rules=(RuffRule.MODULE_IMPORT_NOT_AT_TOP_OF_FILE,)),
    )

    NO_LOCAL_IMPORTS = (
        "Fix an import cycle by restructuring the modules, never with a local import"
    )

    GUARD_TYPE_ONLY_IMPORTS = "Guard type-only imports with `TYPE_CHECKING`"

    NO_RUNTIME_IMPORTS_UNDER_TYPE_CHECKING = (
        "Never import under `TYPE_CHECKING` what the module uses when it runs",
        (),
        RuffRuleChecker(lint_rules=(RuffRule.RUNTIME_IMPORT_IN_TYPE_CHECKING_BLOCK,)),
    )

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
