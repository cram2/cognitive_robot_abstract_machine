"""
Generates the file of each issue form krrood ships and a repository declares.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from krrood.code_generation.generator import CodeGenerator
from krrood.development_tools.agents.rule_module import RuleModule
from krrood.development_tools.generated_file import GeneratedFile
from krrood.development_tools.issue_forms import shipped_forms
from krrood.development_tools.issue_forms.issue_form import IssueForm
from krrood.development_tools.subcommand import Subcommand
from krrood.patterns.path_enumeration import PathEnumeration

# %% files read


class IssueFormTemplateFile(PathEnumeration):
    """
    The templates the issue form files are rendered from.
    """

    ISSUE_FORM = Path(__file__).parent / "templates" / "issue_form.yml.jinja"
    """
    The content of one issue form file.
    """


class RepositoryIssueFormsModule(StrEnum):
    """
    The module ``krrood issue-forms`` imports from the repository root to find the
    repository's issue forms: the forms defined in it, or in the modules of a package of
    that name.
    """

    ISSUE_FORMS = "issue_forms"
    """
    The module or package declaring the issue forms.
    """


# %% generated files


@dataclass(kw_only=True)
class IssueFormFile(GeneratedFile):
    """
    The file of one issue form of a repository.
    """

    repository_root: Path
    """
    The directory the form's :attr:`IssueForm.path` is relative to.
    """

    form: type[IssueForm]
    """
    The form the file declares.
    """

    @property
    def path(self) -> Path:
        return self.repository_root / self.form.path

    @property
    def text(self) -> str:
        return CodeGenerator(
            template_directory=str(IssueFormTemplateFile.ISSUE_FORM.parent)
        ).render(
            template_name=IssueFormTemplateFile.ISSUE_FORM.name,
            form=self.form,
            generate_command=Subcommand.ISSUE_FORMS.invocation,
        )


@dataclass(kw_only=True)
class RepositoryIssueForms:
    """
    The issue forms of a repository: those krrood ships and those the repository
    declares in its :attr:`RepositoryIssueFormsModule.ISSUE_FORMS` module.
    """

    repository_root: Path
    """
    The directory holding the module, which the forms' paths are relative to.
    """

    @property
    def files(self) -> list[IssueFormFile]:
        """
        The file of every shipped and declared form, in alphabetical order of their
        titles.
        """
        forms = RuleModule(module=shipped_forms).defined_subclasses(base=IssueForm)
        project_forms = RuleModule.import_from(
            directory=self.repository_root,
            module_name=RepositoryIssueFormsModule.ISSUE_FORMS,
        )
        if project_forms is not None:
            forms += project_forms.defined_subclasses(base=IssueForm)
        return [
            IssueFormFile(repository_root=self.repository_root, form=form)
            for form in sorted(forms, key=lambda form: form.title)
        ]
