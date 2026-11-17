"""
The issue forms krrood ships, which ``krrood issue-forms`` writes into every
repository's ``.github/ISSUE_TEMPLATE`` alongside the repository's own.
"""

from __future__ import annotations

from krrood.development_tools.issue_forms.issue_form import (
    CodeLanguage,
    IssueForm,
    TextArea,
)
from krrood.patterns.classproperty import classproperty

# %% bug report


class BugReportIssueForm(IssueForm):
    """
    The report of code that behaves differently from its documentation or intent.
    """

    @classproperty
    def description(cls) -> str:
        return "Report code that behaves differently from its documentation or intent."

    @classproperty
    def issue_type(cls) -> str:
        """
        Every bug report is a ``Bug``, as :meth:`IssueForm.issue_type` allows.
        """
        return "Bug"

    @classproperty
    def text_areas(cls) -> list[TextArea]:
        return [
            TextArea(
                identifier="describe-the-issue",
                label="Describe the issue",
                description=(
                    "What happens, in which situation, and what should happen instead."
                ),
                required=True,
            ),
            TextArea(
                identifier="reproduce-the-code-example",
                label="Reproduce the code example",
                description=(
                    "A short, self contained script that reproduces the problem and "
                    "runs as it is with `python reproduce.py`. Name the commit or "
                    "version it ran on."
                ),
                code_language=CodeLanguage.PYTHON,
                required=True,
            ),
            TextArea(
                identifier="error-message",
                label="Error message",
                description=(
                    "The full error message, if there is one, starting from "
                    "`Traceback (most recent call last):`."
                ),
                code_language=CodeLanguage.SHELL,
            ),
        ]
