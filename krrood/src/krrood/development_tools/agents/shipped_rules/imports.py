"""
The rules on imports.
"""

from __future__ import annotations

from krrood.development_tools.agents.rule import CodeReference, Section
from krrood.patterns.exceptions import OptionalModuleNotInstalled
from krrood.patterns.optional_module import Extra, OptionalModule


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

    IMPORT_OPTIONAL_MODULES_WHERE_USED = (
        "A library that only one extra of the distribution installs is an optional "
        "module; declare it as a member of the distribution's subclass of "
        f"{CodeReference(OptionalModule).path}, naming its "
        f"{CodeReference(Extra).path} member. Import it only inside the functions "
        f"that need it, each decorated with the member's "
        f"`{OptionalModule.require.__name__}`, which raises "
        f"{CodeReference(OptionalModuleNotInstalled).name} naming the extra. This is "
        "the one exception to importing at the top of the module; a type-only import "
        "of an optional module goes under `TYPE_CHECKING`"
    )

    CHECK_OPTIONAL_IMPORTS = (
        "Run `krrood optional-imports` over the distribution's source, through the "
        "repository's pre-commit hook, so every optional import is guarded"
    )
