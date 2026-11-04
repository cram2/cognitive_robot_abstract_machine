"""
The rules for this repository only, which ``krrood agents-md`` writes into ``AGENTS.md``
after the rules krrood ships.
"""

from __future__ import annotations

from cognitive_robot_abstract_machine.pull_request_labels import ChangeKindLabel
from cognitive_robot_abstract_machine.repository_files import RepositoryFile
from krrood.development_tools.agents.rule import Section

# %% generated ORM interfaces


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


# %% krrood isolation


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


# %% domain specific conventions


class DomainSpecificConventionsRules(Section):
    """
    The rules pointing to the conventions of one domain.
    """

    FOLLOW_THE_SPATIAL_STYLE_GUIDE = (
        "For spatial types and connections, follow "
        "`semantic_digital_twin/doc/style_guide.md`"
    )


class CramphIsolationRules(Section):
    """
    The rules keeping cramph independent of the motion and plan packages built on it.
    """

    CRAMPH_IMPORTS_ONLY_KRROOD_AND_SEMANTIC_DIGITAL_TWIN = (
        "`cramph`, including its tests under `test/cramph_test`, imports only "
        "`krrood` and `semantic_digital_twin` among the workspace packages - in "
        "particular not `giskardpy`, `coraplex`, `segmind`, `physics_simulators`, "
        "`robokudo` or `experiments`"
    )

    ROS_IS_A_SOFT_DEPENDENCY_OF_CRAMPH = (
        "ROS is a soft dependency of `cramph`: only a tool that needs it, such as "
        "the statechart inspector, imports ROS packages, and nothing else in "
        "`cramph` imports that tool"
    )

    GENERIC_STATECHART_BEHAVIOUR_BELONGS_IN_CRAMPH = (
        "Generic statechart behaviour (life cycles, transition conditions, ticking, "
        "composite nodes, generic monitors, plotting) belongs in `cramph`; "
        "motion-specific behaviour (QP constraints, tasks, world and robot monitors) "
        "belongs in `giskardpy`, built on top of `cramph`"
    )


# %% pull request labels


class PullRequestLabelsRules(Section):
    """
    The rules on labelling pull requests by the kinds of change they make.
    """

    LABEL_EVERY_PULL_REQUEST = (
        "Label every pull request you open with its kinds of change. Where you cannot "
        "set a label, because the repository lacks it or the pull request is opened "
        "through a link, name the labels to the user instead"
    )

    SUGGEST_LABELS_FOR_OTHER_PULL_REQUESTS = (
        "On a pull request you do not open yourself - one you prepare for the "
        "developer to open, review, or are asked about - set no labels; suggest to its "
        f"developer which labels to add and, for an `{ChangeKindLabel.API_BREAK}`, "
        "what the rules below ask of it"
    )

    CHANGE_KIND_LABELS = (
        "Apply every kind-of-change label that describes a part of the change:",
        (
            (
                f"`{ChangeKindLabel.API_BREAK}`: the change can break code that "
                "already uses it - it removes or renames a public module, class, "
                "method, field or parameter, changes a signature incompatibly, changes "
                "the documented behaviour or return value of a public method, or "
                "changes an ORM schema or a serialized format"
            ),
            (
                f"`{ChangeKindLabel.NEW_FEATURE}`: the change adds public API or "
                "user-visible behaviour without breaking existing users"
            ),
            (
                f"`{ChangeKindLabel.BUG_FIX}`: the change makes existing behaviour "
                "match its documented or intended contract"
            ),
            (
                f"`{ChangeKindLabel.MAINTENANCE}`: the change modifies code without "
                "changing its behaviour or API - refactoring, cleanup, dependency "
                "updates"
            ),
        ),
    )

    API_BREAK_HOWEVER_SMALL = (
        f"Apply `{ChangeKindLabel.API_BREAK}` whenever any part of the change breaks "
        "existing users, however small that part is"
    )

    FIND_THE_API_BREAKS_THE_CHECK_MISSES = (
        f"The workflow `{RepositoryFile.API_BREAK_CHECK_WORKFLOW}` adds "
        f"`{ChangeKindLabel.API_BREAK}` when a public signature breaks; look for the "
        "breaks it cannot see: changed documented behaviour or return values, ORM "
        "schemas and serialized formats"
    )

    LEAVE_PATH_LABELS_TO_THE_LABELER = (
        f"Never set or suggest framework labels, `{ChangeKindLabel.DOCUMENTATION}` or "
        f"`{ChangeKindLabel.DEV_TOOLS}`: the labeler sets them on every pull request "
        f"from the files it changes, as `{RepositoryFile.LABELER_CONFIGURATION}` maps "
        "them"
    )

    DISCUSS_API_BREAKS_BEFORE_REVIEW = (
        f"Before review is requested on an `{ChangeKindLabel.API_BREAK}` pull request:",
        (
            "its description states what breaks and how users migrate",
            (
                "tell the user to discuss the break with the code owners of the "
                f"affected frameworks, listed in `{RepositoryFile.CODE_OWNERS}`, or in "
                "the Cramgineers channel"
            ),
        ),
    )
