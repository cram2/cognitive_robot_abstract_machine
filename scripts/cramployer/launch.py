"""
Select deployment command and container feature implementations.
"""

from __future__ import annotations

# %% Imports
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from scripts.cramployer.commands import (
    DeploymentCommand,
    InterfaceGeneration,
    RobotWorkspaceBuild,
    DevelopmentShell,
    RootShell,
    EnvironmentSetup,
    ConfigurationDisplay,
    UsageDisplay,
)
from scripts.cramployer.features import ContainerFeature, GraphicalAccess, NvidiaAccess
from scripts.cramployer.launcher_vocabulary import (
    CommandName,
    EnvironmentVariable,
    ExitStatus,
    FeatureSetting,
)
from scripts.cramployer.settings import DeploymentSettings


# %% Deployment dispatch
@dataclass
class DeploymentLauncher:
    """
    Compose selected container features and execute a registered deployment command.
    """

    configuration: DeploymentSettings
    """Host source locations and container settings."""
    command_types: dict[str, type[DeploymentCommand]] = field(
        default_factory=lambda: {
            CommandName.SETUP: EnvironmentSetup,
            CommandName.SHELL: DevelopmentShell,
            CommandName.OFFLINE_SHELL: DevelopmentShell,
            CommandName.ROOT_SHELL: RootShell,
            CommandName.BUILD_ROBOT_WORKSPACE: RobotWorkspaceBuild,
            CommandName.GENERATE_INTERFACES: InterfaceGeneration,
            CommandName.CONFIGURATION: ConfigurationDisplay,
            CommandName.HELP: UsageDisplay,
            CommandName.SHORT_HELP: UsageDisplay,
            CommandName.LONG_HELP: UsageDisplay,
        }
    )
    """
    Command implementations selected by public command names.
    """

    feature_types: dict[str, type[ContainerFeature]] = field(
        default_factory=lambda: {
            EnvironmentVariable.NVIDIA_ACCESS: NvidiaAccess,
            EnvironmentVariable.GRAPHICAL_ACCESS: GraphicalAccess,
        }
    )
    """
    Feature implementations selected by configuration environment variables.
    """

    def run(self, arguments: tuple[str, ...]) -> int:
        """
        Execute a selected command after configuring enabled container features.
        """
        command_name = next(iter(arguments), CommandName.HELP)
        if command_name not in self.command_types:
            print(
                f"Unknown command: {command_name}. Run ./cramployer help for available commands.",
                file=sys.stderr,
            )
            return ExitStatus.INVALID_SELECTION
        command = self.command_types[command_name](self.configuration)
        features = [
            implementation()
            for setting, implementation in self.feature_types.items()
            if self.configuration.process_environment.get(setting)
            == FeatureSetting.ENABLED
        ]
        for feature in features:
            feature.configure(self.configuration, command.authorize_desktop)
        return command.execute(arguments[1:])


# %% Entry point
if __name__ == "__main__":
    raise SystemExit(
        DeploymentLauncher(
            DeploymentSettings.from_json(dict(os.environ), Path(sys.argv[1]))
        ).run(tuple(sys.argv[2:]))
    )
