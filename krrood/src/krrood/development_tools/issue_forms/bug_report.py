"""
The bug report form, which ``krrood issue-forms`` writes into every repository's
``.github/ISSUE_TEMPLATE`` alongside the repository's own forms.
"""

from __future__ import annotations

from krrood.development_tools.issue_forms.issue_form import (
    CodeLanguage,
    IssueForm,
    IssueType,
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
    def issue_type(cls) -> IssueType:
        """
        Every bug report is a :attr:`IssueType.BUG`.
        """
        return IssueType.BUG

    @classproperty
    def issue_description(cls) -> TextArea:
        """
        What goes wrong.
        """
        return TextArea(
            identifier="describe-the-issue",
            label="Describe the issue",
            description=(
                "What happens, in which situation, and what should happen instead."
            ),
            required=True,
        )

    @classproperty
    def code_example(cls) -> TextArea:
        """
        The script that makes the bug happen.
        """
        return TextArea(
            identifier="reproduce-the-code-example",
            label="Reproduce the code example",
            description=(
                "A short, self contained script that reproduces the problem and runs as "
                "it is with `python reproduce.py`. Name the commit or version it ran on "
                "in a comment at its top."
            ),
            code_language=CodeLanguage.PYTHON,
            required=True,
        )

    @classproperty
    def error_message(cls) -> TextArea:
        """
        What the script prints when the bug happens.
        """
        return TextArea(
            identifier="error-message",
            label="Error message",
            description=(
                "The full error message, if there is one, starting from "
                "`Traceback (most recent call last):`."
            ),
            code_language=CodeLanguage.SHELL,
        )
