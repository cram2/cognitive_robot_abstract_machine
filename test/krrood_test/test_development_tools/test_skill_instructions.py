"""
Tests of declaring skills and generating their ``SKILL.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typing_extensions import Callable, Iterator

from krrood.development_tools.agents.instructions import (
    RepositorySkills,
    SkillInstructions,
)
from krrood.development_tools.agents.repository_files import (
    RepositoryInstructionsFile,
)
from krrood.development_tools.agents.rule_module import RuleModule
from krrood.development_tools.command_line import (
    CommandLine,
    ExitStatus,
    Option,
)
from krrood.development_tools.subcommand import Subcommand

from .dataset import project_skills_module
from .dataset.project_skills_module import AuditingSkill, DeployingTheServiceSkill
from .dataset_files import DatasetFile

# %% skills


def test_a_skill_is_named_by_the_words_of_its_class_name_in_kebab_case() -> None:
    assert DeployingTheServiceSkill.name == "deploying-the-service"


def test_a_skill_is_titled_by_the_words_of_its_class_name_in_sentence_case() -> None:
    assert DeployingTheServiceSkill.title == "Deploying the service"


def test_a_module_lists_the_skills_it_defines_alphabetically() -> None:
    assert RuleModule(module=project_skills_module).skills == [
        AuditingSkill,
        DeployingTheServiceSkill,
    ]


def test_a_module_lists_no_skill_it_only_imports() -> None:
    """
    A skill imported into a module is listed only by the module defining it.
    """
    assert RuleModule(module=sys.modules[__name__]).skills == []


# %% the generated file


def test_the_file_of_a_skill_lies_where_the_skill_says(
    repository_root: Path,
) -> None:
    instructions = SkillInstructions(
        repository_root=repository_root, skill=DeployingTheServiceSkill
    )

    assert instructions.path == repository_root / DeployingTheServiceSkill.path


def test_the_file_lists_the_introduction_then_the_sections_in_the_order_given(
    repository_root: Path,
) -> None:
    """
    The front matter comes first, then the title, the introduction and every section one
    heading level below the title.
    """
    instructions = SkillInstructions(
        repository_root=repository_root, skill=DeployingTheServiceSkill
    )

    assert instructions.text == DatasetFile.SKILL_TEXT.read_text().format(
        generate_command=Subcommand.SKILLS.invocation,
        source_path=DeployingTheServiceSkill.source_path,
    )


# %% project skills


@pytest.fixture
def forget_project_skills() -> Iterator[None]:
    """
    Remove the repository's skills module from the imported modules after the test, so
    that the next test imports its own.
    """
    yield
    for module_name in list(sys.modules):
        if module_name.split(".")[0] == RepositoryInstructionsFile.PROJECT_SKILLS:
            del sys.modules[module_name]


@pytest.fixture
def project_skills_repository(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
    forget_project_skills: None,
) -> Path:
    """
    :return: :func:`repository_root`, holding the repository's skills module.
    """
    copy_into_temporary_directory(
        DatasetFile.PROJECT_SKILLS_MODULE,
        f"{RepositoryInstructionsFile.PROJECT_SKILLS}.py",
    )
    return repository_root


def test_a_repository_without_a_skills_module_has_no_skill_files(
    repository_root: Path,
) -> None:
    assert RepositorySkills(repository_root=repository_root).instructions == []


def test_a_repository_has_a_file_for_every_skill_it_declares(
    project_skills_repository: Path,
) -> None:
    skills = [
        instructions.skill
        for instructions in RepositorySkills(
            repository_root=project_skills_repository
        ).instructions
    ]

    assert [skill.name for skill in skills] == [
        AuditingSkill.name,
        DeployingTheServiceSkill.name,
    ]


# %% command line


def test_skills_command_writes_every_skill_file(
    project_skills_repository: Path,
) -> None:
    """
    ``krrood skills`` writes the missing files and exits with a failure, as pre-commit
    expects of a hook that changed a file.
    """
    exit_status = CommandLine.main(command_line=[Subcommand.SKILLS])

    instructions = RepositorySkills(
        repository_root=project_skills_repository
    ).instructions
    assert exit_status == ExitStatus.FAILURE
    assert [file.path.read_text() for file in instructions] == [
        file.text for file in instructions
    ]


def test_skills_command_succeeds_when_every_file_is_current(
    project_skills_repository: Path,
) -> None:
    CommandLine.main(command_line=[Subcommand.SKILLS])

    assert CommandLine.main(command_line=[Subcommand.SKILLS]) == ExitStatus.SUCCESS


def test_skills_check_fails_on_an_edited_file_without_rewriting_it(
    project_skills_repository: Path,
) -> None:
    CommandLine.main(command_line=[Subcommand.SKILLS])
    path = (
        RepositorySkills(repository_root=project_skills_repository).instructions[0].path
    )
    path.write_text(DatasetFile.HAND_WRITTEN_RULES.read_text())

    exit_status = CommandLine.main(command_line=[Subcommand.SKILLS, Option.CHECK])

    assert exit_status == ExitStatus.FAILURE
    assert path.read_text() == DatasetFile.HAND_WRITTEN_RULES.read_text()


def test_skills_check_succeeds_when_every_file_is_current(
    project_skills_repository: Path,
) -> None:
    CommandLine.main(command_line=[Subcommand.SKILLS])

    assert (
        CommandLine.main(command_line=[Subcommand.SKILLS, Option.CHECK])
        == ExitStatus.SUCCESS
    )
