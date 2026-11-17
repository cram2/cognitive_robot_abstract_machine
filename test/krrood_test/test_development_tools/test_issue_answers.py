"""
Tests of reading the answers of an issue opened with an issue form.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from krrood.development_tools.command_line import CommandLine, ExitStatus, Option
from krrood.development_tools.issue_forms.exceptions import (
    UnknownIssueForm,
    UnknownTextArea,
)
from krrood.development_tools.subcommand import Subcommand

from .dataset.project_issue_forms_module import CrashReportIssueForm
from .dataset_files import DatasetFile

# %% answers in an issue body


def test_a_code_answer_is_the_code_without_its_fence() -> None:
    """
    A line inside the code that looks like a heading does not end the answer.
    """
    answer = CrashReportIssueForm.find_answer(
        issue_body=DatasetFile.CRASH_REPORT_ISSUE_BODY.read_text(),
        text_area=CrashReportIssueForm.script,
    )

    assert answer == DatasetFile.CRASH_REPORT_SCRIPT.read_text().rstrip("\n")


def test_a_field_left_empty_has_no_answer() -> None:
    """
    GitHub writes a placeholder for an optional field left empty.
    """
    answer = CrashReportIssueForm.find_answer(
        issue_body=DatasetFile.CRASH_REPORT_UNANSWERED_ISSUE_BODY.read_text(),
        text_area=CrashReportIssueForm.script,
    )

    assert answer is None


def test_a_field_whose_heading_was_removed_has_no_answer() -> None:
    """
    An issue body edited after it was opened may no longer hold a field.
    """
    answer = CrashReportIssueForm.find_answer(
        issue_body=DatasetFile.HAND_WRITTEN_RULES.read_text(),
        text_area=CrashReportIssueForm.script,
    )

    assert answer is None


def test_a_text_area_is_found_by_its_identifier() -> None:
    """
    The identifier names the text area on the command line.
    """
    text_area = CrashReportIssueForm.find_text_area(
        identifier=CrashReportIssueForm.script.identifier
    )

    assert text_area == CrashReportIssueForm.script


def test_an_unknown_identifier_names_no_text_area() -> None:
    """
    A form refuses an identifier none of its text areas has.
    """
    with pytest.raises(UnknownTextArea):
        CrashReportIssueForm.find_text_area(identifier="no-such-field")


# %% command line


def run_issue_answer(
    *, monkeypatch: pytest.MonkeyPatch, issue_body: str, form_name: str, identifier: str
) -> ExitStatus:
    """
    Run ``krrood issue-answer`` with the issue body on its standard input.

    :return: The exit status of the command.
    """
    monkeypatch.setattr(sys, "stdin", io.StringIO(issue_body))
    return CommandLine.main(
        command_line=[
            Subcommand.ISSUE_ANSWER,
            Option.FORM,
            form_name,
            Option.FIELD,
            identifier,
        ]
    )


def test_issue_answer_command_prints_the_answer(
    project_issue_forms_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    The answer goes to standard output, so a workflow can write it into a file.
    """
    exit_status = run_issue_answer(
        monkeypatch=monkeypatch,
        issue_body=DatasetFile.CRASH_REPORT_ISSUE_BODY.read_text(),
        form_name=CrashReportIssueForm.name,
        identifier=CrashReportIssueForm.script.identifier,
    )

    assert exit_status == ExitStatus.SUCCESS
    assert capsys.readouterr().out == DatasetFile.CRASH_REPORT_SCRIPT.read_text()


def test_issue_answer_command_fails_on_an_unanswered_field(
    project_issue_forms_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Nothing is printed for a field left empty, and the command fails.
    """
    exit_status = run_issue_answer(
        monkeypatch=monkeypatch,
        issue_body=DatasetFile.CRASH_REPORT_UNANSWERED_ISSUE_BODY.read_text(),
        form_name=CrashReportIssueForm.name,
        identifier=CrashReportIssueForm.script.identifier,
    )

    assert exit_status == ExitStatus.FAILURE
    assert capsys.readouterr().out == ""


def test_issue_answer_command_refuses_an_unknown_form(
    project_issue_forms_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A form name no shipped or declared form has is an error, not an empty answer.
    """
    with pytest.raises(UnknownIssueForm):
        run_issue_answer(
            monkeypatch=monkeypatch,
            issue_body=DatasetFile.CRASH_REPORT_ISSUE_BODY.read_text(),
            form_name="no-such-form",
            identifier=CrashReportIssueForm.script.identifier,
        )
