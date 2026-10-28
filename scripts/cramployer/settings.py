"""
Host deployment settings parsed from the shell configuration environment.
"""

from __future__ import annotations

# %% Imports
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from scripts.cramployer.launcher_vocabulary import (
    DockerArgument,
    EnvironmentVariable,
    Executable,
)
from scripts.cramployer.vocabulary import FileName, OptionalPackage, DependencySource


# %% Deployment configuration
@dataclass
class DeploymentSettings:
    """
    Source locations and image settings for a development environment.
    """

    deployment_directory: Path
    """Directory containing deployment configuration and scripts."""
    source_checkout: Path
    """
    Editable CRAM checkout mounted in development containers.
    """

    robot_workspace_sources: Path
    """Editable ROS package sources."""
    repository_address: str = (
        "https://github.com/cram2/cognitive_robot_abstract_machine.git"
    )
    """
    Repository cloned when the editable checkout does not exist.
    """

    repository_branch: str = "main"
    """
    Branch cloned for a new checkout.
    """

    user_identifier: int = field(default_factory=os.getuid)
    """
    Host user identifier for generated file ownership.
    """

    group_identifier: int = field(default_factory=os.getgid)
    """
    Host group identifier for generated file ownership.
    """

    dependency_source: str = DependencySource.DEFAULT
    """
    PyTorch dependency index selection.
    """

    additional_packages: str = OptionalPackage.ROBOKUDO
    """
    Comma-separated optional package-specific CI installations.
    """

    workspace_extras: str = ""
    """
    Comma-separated optional workspace features.
    """

    project_name: str = "cramployer"
    """
    Compose project identifying development containers.
    """

    compose_files: list[Path] = field(default_factory=lambda: [Path("compose.yaml")])
    """
    Compose files applied in order.
    """

    process_environment: dict[str, str] = field(
        default_factory=lambda: os.environ.copy()
    )
    """
    Environment passed to deployment processes.
    """

    @classmethod
    def from_json(
        class_type, payload: dict[str, str], deployment_directory: Path
    ) -> DeploymentSettings:
        """
        Parse exported launcher settings from an environment mapping.
        """
        return class_type(
            deployment_directory=deployment_directory,
            source_checkout=Path(payload[EnvironmentVariable.SOURCE_CHECKOUT]),
            robot_workspace_sources=Path(
                payload[EnvironmentVariable.ROBOT_WORKSPACE_SOURCES]
            ),
            repository_address=payload[EnvironmentVariable.REPOSITORY_ADDRESS],
            repository_branch=payload[EnvironmentVariable.REPOSITORY_BRANCH],
            user_identifier=int(payload[EnvironmentVariable.USER_IDENTIFIER]),
            group_identifier=int(payload[EnvironmentVariable.GROUP_IDENTIFIER]),
            dependency_source=payload[EnvironmentVariable.DEPENDENCY_SOURCE],
            additional_packages=payload.get(
                EnvironmentVariable.ADDITIONAL_PACKAGES, class_type.additional_packages
            ),
            workspace_extras=payload.get(
                EnvironmentVariable.WORKSPACE_EXTRAS, class_type.workspace_extras
            ),
            project_name=payload.get(
                EnvironmentVariable.PROJECT_NAME, class_type.project_name
            ),
            process_environment=payload.copy(),
        )

    def build_compose_command(self) -> list[str]:
        """
        Build the Compose invocation for the selected configuration files.
        """
        arguments: list[str] = [
            Executable.DOCKER,
            DockerArgument.COMPOSE,
            DockerArgument.ENVIRONMENT_FILE,
            FileName.ENVIRONMENT,
        ]
        for compose_file in self.compose_files:
            arguments.extend([DockerArgument.FILE, str(compose_file)])
        return arguments

    def run(
        self,
        arguments: list[str],
        capture_output: bool = False,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run a deployment process with the configured environment and directory.
        """
        return subprocess.run(
            arguments,
            cwd=self.deployment_directory,
            env=self.process_environment,
            text=True,
            capture_output=capture_output,
            input=input_text,
            check=True,
        )
