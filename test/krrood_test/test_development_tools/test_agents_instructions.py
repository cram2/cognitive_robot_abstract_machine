from __future__ import annotations

from pathlib import Path

import pytest
from typing_extensions import Callable

from krrood.development_tools.agent_instructions import AgentsInstructions
from krrood.development_tools.agent_rule import (
    CodeReference,
    KrroodRules,
    ProjectRules,
    PythonRules,
    RuleSet,
    Section,
)
from krrood.development_tools.command_line import (
    CommandLine,
    ExitStatus,
    Option,
    Subcommand,
)
from krrood.development_tools.exceptions import SectionWithoutParent
from krrood.development_tools.repository_files import RepositoryInstructionsFile
from krrood.exceptions import DataclassException
from krrood.patterns import path_enumeration

from .dataset.rule_sections import (
    DatasetRules,
    Deployment,
    DesignPrinciples,
    Errors,
    ReleasesThroughCI,
)
from .dataset_files import DatasetFile

# %% sections


def test_a_rule_set_lists_its_sections_and_their_rules(repository_root: Path) -> None:
    instructions = AgentsInstructions(repository_root, (DatasetRules,))

    assert instructions.text == DatasetFile.RULE_SECTIONS_TEXT.read_text().format(
        source_path=DatasetRules.source_path
    )


def test_the_subsections_of_a_section_name_it_as_their_parent() -> None:
    assert DesignPrinciples.subsections == [Errors]


def test_a_section_naming_no_parent_is_rejected() -> None:
    with pytest.raises(SectionWithoutParent):
        Section.parent


def test_a_section_lists_only_its_own_rules() -> None:
    assert Errors.rules == [Errors.RAISE_ERRORS, Errors.NAME_ERRORS]


def test_a_section_without_rules_has_none_to_list() -> None:
    assert not Deployment.has_rules


def test_a_title_keeps_acronyms_in_capitals() -> None:
    assert ReleasesThroughCI.title == "Releases through CI"


def test_the_rule_sets_are_the_subclasses_of_rule_set() -> None:
    assert RuleSet.available[:3] == [PythonRules, KrroodRules, ProjectRules]


def test_python_rules_name_no_krrood_api(repository_root: Path) -> None:
    text = AgentsInstructions(repository_root, (PythonRules,)).text
    rules = "\n".join(line for line in text.splitlines() if not line.startswith("<!--"))

    assert "krrood" not in rules.lower()


def test_shipped_rules_name_no_cram_package(repository_root: Path) -> None:
    text = AgentsInstructions(repository_root, (PythonRules, KrroodRules)).text

    assert "cram" not in text.lower()


# %% code named by rules


def test_a_class_is_named_by_its_import_path() -> None:
    assert CodeReference(DataclassException).path == (
        f"`{DataclassException.__module__}.{DataclassException.__qualname__}`"
    )


def test_a_module_is_named_by_its_import_path() -> None:
    assert CodeReference(path_enumeration).path == f"`{path_enumeration.__name__}`"


def test_an_object_is_named_by_the_module_it_is_imported_from() -> None:
    assert CodeReference(pytest.raises, pytest).path == (
        f"`{pytest.__name__}.{pytest.raises.__qualname__}`"
    )


def test_a_method_is_named_by_its_own_name() -> None:
    assert CodeReference(DataclassException.error_message).name == (
        f"`{DataclassException.error_message.__name__}`"
    )


# %% project rules


@pytest.fixture
def project_rules_repository(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
) -> Path:
    """
    :return: :func:`repository_root`, holding the repository's own rules module.
    """
    copy_into_temporary_directory(
        DatasetFile.PROJECT_RULES_MODULE, RepositoryInstructionsFile.PROJECT_RULES
    )
    return repository_root


def test_the_project_rules_module_declares_project_sections(
    project_rules_repository: Path,
) -> None:
    module = ProjectRules.import_from(project_rules_repository)

    assert module.Deployment in ProjectRules.subsections


def test_the_project_rules_module_is_imported_once(
    project_rules_repository: Path,
) -> None:
    assert ProjectRules.import_from(
        project_rules_repository
    ) is ProjectRules.import_from(project_rules_repository)


def test_a_repository_without_a_project_rules_module_has_none(
    repository_root: Path,
) -> None:
    assert ProjectRules.import_from(repository_root) is None


# %% command line


def test_agents_md_command_writes_a_new_file(repository_root: Path) -> None:
    exit_status = CommandLine.main([Subcommand.AGENTS_MD])

    assert exit_status == ExitStatus.FAILURE
    assert (repository_root / RepositoryInstructionsFile.AGENTS).read_text() == (
        AgentsInstructions(repository_root, tuple(RuleSet.available)).text
    )


def test_agents_md_command_replaces_a_hand_written_file(
    repository_root: Path,
) -> None:
    path = repository_root / RepositoryInstructionsFile.AGENTS
    path.write_text(DatasetFile.HAND_WRITTEN_RULES.read_text())

    CommandLine.main([Subcommand.AGENTS_MD])

    assert path.read_text() == (
        AgentsInstructions(repository_root, tuple(RuleSet.available)).text
    )


def test_agents_md_command_writes_only_the_chosen_rule_sets(
    repository_root: Path,
) -> None:
    CommandLine.main(
        [Subcommand.AGENTS_MD, Option.RULE_SETS, PythonRules.command_line_name]
    )

    assert (repository_root / RepositoryInstructionsFile.AGENTS).read_text() == (
        AgentsInstructions(repository_root, (PythonRules,)).text
    )


def test_agents_md_command_keeps_rule_set_order_whatever_the_order_given(
    repository_root: Path,
) -> None:
    CommandLine.main(
        [
            Subcommand.AGENTS_MD,
            Option.RULE_SETS,
            KrroodRules.command_line_name,
            PythonRules.command_line_name,
        ]
    )

    assert (repository_root / RepositoryInstructionsFile.AGENTS).read_text() == (
        AgentsInstructions(repository_root, (PythonRules, KrroodRules)).text
    )


def test_agents_md_command_succeeds_when_the_file_is_current(
    repository_root: Path,
) -> None:
    CommandLine.main([Subcommand.AGENTS_MD])

    assert CommandLine.main([Subcommand.AGENTS_MD]) == ExitStatus.SUCCESS


def test_check_fails_on_an_edited_file_without_rewriting_it(
    repository_root: Path,
) -> None:
    path = repository_root / RepositoryInstructionsFile.AGENTS
    path.write_text(DatasetFile.HAND_WRITTEN_RULES.read_text())

    exit_status = CommandLine.main([Subcommand.AGENTS_MD, Option.CHECK])

    assert exit_status == ExitStatus.FAILURE
    assert path.read_text() == DatasetFile.HAND_WRITTEN_RULES.read_text()


def test_agents_md_command_adds_the_project_rules(
    project_rules_repository: Path,
) -> None:
    CommandLine.main([Subcommand.AGENTS_MD])

    text = (project_rules_repository / RepositoryInstructionsFile.AGENTS).read_text()
    assert ProjectRules.import_from(project_rules_repository).Deployment.heading in (
        text.splitlines()
    )


def test_check_fails_when_the_file_is_missing(repository_root: Path) -> None:
    assert CommandLine.main([Subcommand.AGENTS_MD, Option.CHECK]) == ExitStatus.FAILURE
