"""
Tests of declaring issue forms and generating their files.
"""

from __future__ import annotations

import sys
from pathlib import Path


from krrood.development_tools.agents.rule_module import RuleModule
from krrood.development_tools.command_line import CommandLine, ExitStatus, Option
from krrood.development_tools.issue_forms import bug_report
from krrood.development_tools.issue_forms.issue_form import IssueForm
from krrood.development_tools.issue_forms.issue_form_files import (
    IssueFormFile,
    RepositoryIssueForms,
)
from krrood.development_tools.subcommand import Subcommand

from .dataset import project_issue_forms_module
from .dataset.project_issue_forms_module import (
    AnnouncementIssueForm,
    CrashReportIssueForm,
)
from .dataset_files import DatasetFile

# %% issue forms


def test_an_issue_form_is_titled_by_the_words_of_its_class_name_in_sentence_case() -> (
    None
):
    """
    The title GitHub shows for the form is its class name without ``IssueForm``.
    """
    assert CrashReportIssueForm.title == "Crash report"


def test_an_issue_form_lies_in_the_issue_template_directory_in_kebab_case() -> None:
    """
    GitHub finds the form only in ``.github/ISSUE_TEMPLATE``.
    """
    assert CrashReportIssueForm.path == Path(".github/ISSUE_TEMPLATE/crash-report.yml")


def test_an_issue_form_has_every_text_area_classproperty_in_definition_order() -> None:
    """
    A classproperty returning a :class:`TextArea` is a field of the form.
    """
    assert CrashReportIssueForm.text_areas == [
        CrashReportIssueForm.what_crashed,
        CrashReportIssueForm.script,
    ]


def test_a_module_lists_the_issue_forms_it_defines_in_definition_order() -> None:
    """
    Every subclass of the base the module defines is found, in the order it defines
    them.
    """
    assert RuleModule(module=project_issue_forms_module).defined_subclasses(
        base=IssueForm
    ) == [CrashReportIssueForm, AnnouncementIssueForm]


def test_a_module_lists_no_issue_form_it_only_imports() -> None:
    """
    An issue form imported into a module is listed only by the module defining it.
    """
    assert (
        RuleModule(module=sys.modules[__name__]).defined_subclasses(base=IssueForm)
        == []
    )


# %% the generated file


def test_the_file_lists_the_text_areas_in_the_order_given(
    repository_root: Path,
) -> None:
    """
    The notice comes first, then the title, description, issue type and every text area
    with only the attributes it sets.
    """
    file = IssueFormFile(repository_root=repository_root, form=CrashReportIssueForm)

    assert file.text == DatasetFile.ISSUE_FORM_TEXT.read_text().format(
        generate_command=Subcommand.ISSUE_FORMS.invocation,
        source_path=CrashReportIssueForm.source_path,
    )


def test_the_file_of_a_form_without_an_issue_type_sets_no_type(
    repository_root: Path,
) -> None:
    """
    A form leaving :attr:`IssueForm.issue_type` unset writes no ``type`` key.
    """
    file = IssueFormFile(repository_root=repository_root, form=AnnouncementIssueForm)

    assert file.text == DatasetFile.ISSUE_FORM_WITHOUT_TYPE_TEXT.read_text().format(
        generate_command=Subcommand.ISSUE_FORMS.invocation,
        source_path=AnnouncementIssueForm.source_path,
    )


# %% project issue forms


def test_a_repository_without_an_issue_forms_module_has_the_shipped_forms(
    repository_root: Path,
) -> None:
    """
    Every repository gets the issue forms krrood ships.
    """
    files = RepositoryIssueForms(repository_root=repository_root).files

    assert [file.form for file in files] == RuleModule(
        module=bug_report
    ).defined_subclasses(base=IssueForm)


def test_a_repository_has_a_file_for_every_shipped_and_declared_form_alphabetically(
    project_issue_forms_repository: Path,
) -> None:
    """
    The files are ordered by the titles of their forms, not by definition order.
    """
    files = RepositoryIssueForms(repository_root=project_issue_forms_repository).files
    forms = [
        *RuleModule(module=bug_report).defined_subclasses(base=IssueForm),
        AnnouncementIssueForm,
        CrashReportIssueForm,
    ]

    assert [file.form.title for file in files] == sorted(form.title for form in forms)


# %% command line


def test_issue_forms_command_writes_every_form_file(
    project_issue_forms_repository: Path,
) -> None:
    """
    ``krrood issue-forms`` writes the missing files and exits with a failure, as pre-
    commit expects of a hook that changed a file.
    """
    exit_status = CommandLine.main(command_line=[Subcommand.ISSUE_FORMS])

    files = RepositoryIssueForms(repository_root=project_issue_forms_repository).files
    assert exit_status == ExitStatus.FAILURE
    assert [file.path.read_text() for file in files] == [file.text for file in files]


def test_issue_forms_check_fails_on_an_edited_file_without_rewriting_it(
    project_issue_forms_repository: Path,
) -> None:
    """
    ``--check`` reports a file edited by hand and leaves it as it is.
    """
    CommandLine.main(command_line=[Subcommand.ISSUE_FORMS])
    path = (
        RepositoryIssueForms(repository_root=project_issue_forms_repository)
        .files[0]
        .path
    )
    path.write_text(DatasetFile.HAND_WRITTEN_RULES.read_text())

    exit_status = CommandLine.main(command_line=[Subcommand.ISSUE_FORMS, Option.CHECK])

    assert exit_status == ExitStatus.FAILURE
    assert path.read_text() == DatasetFile.HAND_WRITTEN_RULES.read_text()


def test_issue_forms_check_succeeds_when_every_file_is_current(
    project_issue_forms_repository: Path,
) -> None:
    """
    ``--check`` passes once the files have been written.
    """
    CommandLine.main(command_line=[Subcommand.ISSUE_FORMS])

    assert (
        CommandLine.main(command_line=[Subcommand.ISSUE_FORMS, Option.CHECK])
        == ExitStatus.SUCCESS
    )
