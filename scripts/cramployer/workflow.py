"""
Installation steps parsed from a repository's GitHub Actions workflow.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from scripts.cramployer.exceptions import InvalidWorkflowStep, MissingWorkflowStep
from scripts.cramployer.vocabulary import PayloadKey, ShellSyntax

if TYPE_CHECKING:
    from typing_extensions import Any, Self, Sequence


# %% Workflow steps
@dataclass
class WorkflowStep:
    """
    A named CI step with optional shell commands and environment settings.
    """

    title: str
    """Title identifying the step in the workflow."""
    shell_commands: str | None
    """
    Shell commands executed by the step, if present.
    """

    environment: dict[str, Any] = field(default_factory=dict)
    """
    Environment settings supplied by the external workflow.
    """

    @classmethod
    def from_json(class_type, payload: dict[str, Any]) -> Self:
        """
        Parse a named step from a GitHub Actions mapping.
        """
        return class_type(
            title=payload[PayloadKey.NAME],
            shell_commands=payload.get(PayloadKey.RUN),
            environment=payload.get(PayloadKey.STEP_ENVIRONMENT, {}),
        )

    def build_commands(self) -> str:
        """
        Validate a step and return its commands in an independent subshell.
        """
        if (
            not isinstance(self.shell_commands, str)
            or ShellSyntax.GITHUB_EXPRESSION in self.shell_commands
        ):
            raise InvalidWorkflowStep(self.title, PayloadKey.RUN)
        for variable, value in self.environment.items():
            if not isinstance(value, str) or ShellSyntax.GITHUB_EXPRESSION in value:
                raise InvalidWorkflowStep(self.title, variable)
        return "(\n" + self.shell_commands + "\n)"


@dataclass
class Workflow:
    """
    Named steps exposed by the reusable CI test job.
    """

    steps: list[WorkflowStep]
    """
    Named steps in workflow order.
    """

    @classmethod
    def from_json(class_type, payload: dict[str, Any]) -> Self:
        """
        Parse the installation source from a GitHub Actions mapping.
        """
        return class_type(
            [
                WorkflowStep.from_json(step)
                for step in payload[PayloadKey.JOBS][PayloadKey.TEST][PayloadKey.STEPS]
                if PayloadKey.NAME in step
            ]
        )

    def select_steps(self, titles: Sequence[str]) -> list[WorkflowStep]:
        """
        Resolve exactly one step for each requested title in selection order.
        """
        selected = []
        for title in titles:
            matches = [step for step in self.steps if step.title == title]
            if len(matches) != 1:
                raise MissingWorkflowStep(title, len(matches))
            selected.append(matches[0])
        return selected
