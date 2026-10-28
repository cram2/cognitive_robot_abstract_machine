"""
The rules for this repository only, which ``krrood agents-md`` writes into ``AGENTS.md``
after the rules krrood ships.
"""

from __future__ import annotations

from krrood.development_tools.agents.rule import Section


class GeneratedORMInterfacesRules(Section):
    """
    The rules on the generated ORM interfaces.
    """

    NEVER_EDIT = (
        "`ormatic_interface.py` files are generated, never written, and "
        "git-ignored (see `.gitignore`). Never edit them, and never read them to "
        "debug an ORM problem; regenerate them with "
        "`scripts/regenerate_all_orm.py` instead"
    )

    BUILD_LOCALLY = (
        "The test suite builds them for its runs; a local checkout builds them "
        "with `scripts/regenerate_all_orm.py`"
    )

    NEVER_TRACK = (
        "Never track one again: git refuses to overwrite a tracked path a checkout "
        "has generated its own copy of, which makes every branch switch fail"
    )

    NEVER_BUILD_IN_TEST_RUNS = (
        "Pass `--orm-build never` to the test runs you start, and regenerate them "
        "with `scripts/regenerate_all_orm.py` when a change needs it"
    )

    ASK_ABOUT_PERSISTENT_ORM_PROBLEMS = (
        "When regenerating the ORM interfaces does not fix an ORM problem, ask the "
        "developer"
    )

    ASK_ABOUT_ORM_IMPORT_CYCLES = (
        "If an ORM import creates an import cycle, ask the user if a local import "
        "is fine"
    )


class KrroodIsolationRules(Section):
    """
    The rules keeping krrood independent of the other workspace packages.
    """

    KRROOD_IMPORTS_NO_WORKSPACE_PACKAGE = (
        "`krrood`, including its tests under `test/krrood_test`, never imports "
        "another workspace package - in particular not `coraplex`, "
        "`semantic_digital_twin`, `giskardpy`, `physics_simulators`, `robokudo` or "
        "`experiments`. The only exceptions are `random_events` and "
        "`probabilistic_model`, which `krrood`'s source already depends on"
    )

    MIMIC_OTHER_PACKAGES_IN_KRROOD_TESTS = (
        "When a `krrood` test needs behaviour another package triggers, mimic the "
        "relevant classes and patterns in `test/krrood_test/dataset` and test "
        "against the mimics. Keep the test in `krrood` and do not depend on "
        "another package to reproduce the scenario"
    )

    MIMICS_IMPORT_ONLY_KRROOD_DEPENDENCIES = (
        "Those mimic classes import only `krrood`, `random_events` and "
        "`probabilistic_model`"
    )


class DomainSpecificConventionsRules(Section):
    """
    The rules pointing to the conventions of one domain.
    """

    FOLLOW_THE_SPATIAL_STYLE_GUIDE = (
        "For spatial types and connections, follow "
        "`semantic_digital_twin/doc/style_guide.md`"
    )
