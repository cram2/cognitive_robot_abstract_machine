"""
Locations used by deployment test fixtures.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration


# %% Test files
class FixturePath(PathEnumeration):
    """
    Repository inputs and isolated deployment filenames.
    """

    REPOSITORY = Path(__file__).resolve().parents[3]
    """
    Root of the checkout under test.
    """

    DEPLOYMENT = REPOSITORY / "scripts/cramployer"
    """
    Deployment implementation.
    """

    DATASET = Path(__file__).parent
    """
    Input configurations and command executables.
    """

    LAUNCHER = Path("cramployer")
    """
    Development environment launcher.
    """

    CONFIGURATION = Path(".env")
    """
    Local launcher settings.
    """

    EXAMPLE = Path(".env.example")
    """
    Default launcher settings.
    """

    COMMAND_LOG = Path("docker-commands.jsonl")
    """
    Recorded Docker invocations.
    """

    DOCKER = Path("docker")
    """
    Recording executable name.
    """

    COMPOSE = Path("compose.yaml")
    """
    Development service definition.
    """

    CONTAINER_ENVIRONMENT = Path("container.env")
    """
    Optional container settings.
    """

    WORKFLOW = Path(".github/workflows/ci_reusable.yml")
    """
    Installation workflow within a checkout.
    """

    INSTALLATION = Path(__file__).parent / "../installation.yaml"
    """
    Local commands replacing CI installation steps.
    """

    MISSING_STEP = Path(__file__).parent / "../missing-step.yaml"
    """
    Workflow missing required installation steps.
    """

    READ_SOURCE = Path(__file__).parent / "read-source.sh"
    """
    Shell probe for source configuration.
    """

    SOURCE_PROBE = Path(__file__).parent / "docker-source.py"
    """
    Docker mimic reporting its source checkout.
    """

    RECORDER = Path(__file__).parent / "docker.py"
    """
    Docker command recorder executable.
    """

    DEPLOYMENT_ENVIRONMENT = Path(__file__).parent / "deployment.env"
    """
    Launcher settings for isolated checkouts.
    """

    FIRST_OUTPUT = Path("first-output")
    """
    Output produced by the first local installation step.
    """

    SECOND_OUTPUT = Path("second-output")
    """
    Output produced by the second local installation step.
    """

    CHECKOUT = Path("cram")
    """
    Editable checkout in isolated fixtures.
    """

    GIT = Path(".git")
    """
    Git metadata directory.
    """

    DOCKER_DIRECTORY = Path(".github/docker")
    """
    Shared ROS image build context.
    """

    DOCKERFILE = Path("Dockerfile")
    """
    ROS base-image definition.
    """

    ROBOT_WORKSPACE_SOURCES = Path("ros/src")
    """
    ROS package sources within an isolated workspace.
    """

    EXISTING_PACKAGE = Path("existing-package")
    """
    Marker preventing ROS source seeding in command tests.
    """

    EXECUTABLES = Path("bin")
    """
    Directory containing recording executables.
    """

    ISOLATED_DEPLOYMENT = Path("scripts/cramployer")
    """
    Temporary deployment configuration directory.
    """

    DESKTOP_COOKIE = Path(__file__).parent / "desktop-cookie"
    """
    Desktop authorization records supplied by the mimic.
    """

    AUTHORIZATION_MIMIC = Path(__file__).parent / "desktop-authorization.py"
    """
    Executable supplying desktop authorization without an X server.
    """

    AUTHORIZATION_EXECUTABLE = Path("xauth")
    """
    Desktop authorization executable name.
    """
