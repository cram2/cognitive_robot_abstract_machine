"""
When a test run builds the ORM interfaces it reads, and which process does it.
"""

from __future__ import annotations

import sys

import pytest
from typing_extensions import List

from cognitive_robot_abstract_machine.exceptions import (
    MissingOrmBuildChoiceError,
    UnknownOrmBuildChoiceError,
)
from cognitive_robot_abstract_machine.orm_interfaces import (
    WORKSPACE_ORM_INTERFACES,
    WorkspaceOrmInterfaces,
)

from ..orm_interface_build import ORM_BUILD_OPTION, OrmBuild, regenerate_orm_interfaces
from ..pytest_environment import PytestEnvironmentVariable
from .test_orm_interfaces import checkout, workspace  # noqa: F401  (shared fixtures)

PYTEST_COMMAND = "pytest"
"""
The command the arguments of a run follow.
"""

UNKNOWN_CHOICE = "sometimes"
"""
A choice no run offers, for the readings of one to reject.
"""

ABBREVIATED_OPTION = ORM_BUILD_OPTION.removesuffix("uild")
"""
The option cut short, the way argparse lets a run abbreviate a long option.
"""


@pytest.fixture()
def recorded_builds(monkeypatch) -> List[str]:
    """
    Replace the real build with a recorder, so the guard can be exercised without paying
    for a whole regeneration.

    :return: The list every build appends to.
    """
    builds: List[str] = []
    monkeypatch.setattr(
        WORKSPACE_ORM_INTERFACES, "regenerate", lambda: builds.append("built")
    )
    monkeypatch.setattr(sys, "argv", [PYTEST_COMMAND])
    monkeypatch.setenv(PytestEnvironmentVariable.ORM_BUILD, OrmBuild.ALWAYS)
    regenerate_orm_interfaces.cache_clear()
    return builds


# %% one build per run, on the controller


class TestOrmInterfacesBuiltOnlyByTheXdistController:
    """
    The workspace's ORM interfaces must be built once per run, by the controller.

    Every worker imports this conftest too. Letting them build as well would have
    several processes writing the same files at once, and each would pay for the whole
    build again.
    """

    def test_worker_leaves_the_interfaces_alone(self, monkeypatch, recorded_builds):
        monkeypatch.setenv(PytestEnvironmentVariable.XDIST_WORKER, "gw0")

        assert regenerate_orm_interfaces() is False
        assert recorded_builds == []

    def test_controller_builds_them(self, monkeypatch, recorded_builds):
        monkeypatch.delenv(PytestEnvironmentVariable.XDIST_WORKER, raising=False)

        assert regenerate_orm_interfaces() is True
        assert recorded_builds == ["built"]

    def test_a_second_ask_does_not_build_again(self, monkeypatch, recorded_builds):
        monkeypatch.delenv(PytestEnvironmentVariable.XDIST_WORKER, raising=False)
        regenerate_orm_interfaces()

        assert regenerate_orm_interfaces() is True
        assert recorded_builds == ["built"]


# %% the choice a run is started with


@pytest.fixture
def current_interfaces(workspace: WorkspaceOrmInterfaces) -> WorkspaceOrmInterfaces:
    """
    The interfaces of a checkout that has been built since its sources last changed.
    """
    workspace.regenerate()
    return workspace


@pytest.fixture
def outdated_interfaces(
    current_interfaces: WorkspaceOrmInterfaces,
) -> WorkspaceOrmInterfaces:
    """
    The interfaces of a checkout that has not been built since its sources changed.
    """
    current_interfaces.interfaces[0].remove()
    return current_interfaces


class TestWhenAChoiceBuilds:
    """
    A run builds the interfaces every time, never, or only when the checkout has not
    built them since its sources changed.
    """

    def test_never_skips_a_checkout_that_has_none(
        self, outdated_interfaces: WorkspaceOrmInterfaces
    ):
        assert OrmBuild.NEVER.builds(outdated_interfaces) is False

    def test_always_builds_a_current_checkout(
        self, current_interfaces: WorkspaceOrmInterfaces
    ):
        assert OrmBuild.ALWAYS.builds(current_interfaces) is True

    def test_auto_skips_a_current_checkout(
        self, current_interfaces: WorkspaceOrmInterfaces
    ):
        assert OrmBuild.AUTO.builds(current_interfaces) is False

    def test_auto_builds_an_outdated_checkout(
        self, outdated_interfaces: WorkspaceOrmInterfaces
    ):
        assert OrmBuild.AUTO.builds(outdated_interfaces) is True


class TestTheChoiceReachesTheProcessesOfARun:
    """
    The conftests build while pytest is still importing them, before it has parsed the
    command line, so the choice is read off the arguments as they were given.
    """

    def test_a_run_stating_no_choice_builds_what_is_outdated(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [PYTEST_COMMAND])
        monkeypatch.delenv(PytestEnvironmentVariable.ORM_BUILD, raising=False)

        assert OrmBuild.requested() is OrmBuild.AUTO

    def test_a_choice_joined_to_the_option_is_read(self):
        stated = OrmBuild.stated_on(
            [PYTEST_COMMAND, f"{ORM_BUILD_OPTION}={OrmBuild.NEVER}"]
        )

        assert stated is OrmBuild.NEVER

    def test_a_choice_following_the_option_is_read(self):
        stated = OrmBuild.stated_on([PYTEST_COMMAND, ORM_BUILD_OPTION, OrmBuild.ALWAYS])

        assert stated is OrmBuild.ALWAYS

    def test_a_command_line_without_the_option_states_nothing(self):
        assert OrmBuild.stated_on([PYTEST_COMMAND, "-q", "test"]) is None

    def test_the_environment_states_a_choice_for_runs_that_state_none(
        self, monkeypatch
    ):
        monkeypatch.setattr(sys, "argv", [PYTEST_COMMAND])
        monkeypatch.setenv(PytestEnvironmentVariable.ORM_BUILD, OrmBuild.NEVER)

        assert OrmBuild.requested() is OrmBuild.NEVER

    def test_a_stated_choice_beats_the_environment(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", [PYTEST_COMMAND, f"{ORM_BUILD_OPTION}={OrmBuild.ALWAYS}"]
        )
        monkeypatch.setenv(PytestEnvironmentVariable.ORM_BUILD, OrmBuild.NEVER)

        assert OrmBuild.requested() is OrmBuild.ALWAYS

    def test_an_abbreviated_option_is_read(self):
        stated = OrmBuild.stated_on(
            [PYTEST_COMMAND, f"{ABBREVIATED_OPTION}={OrmBuild.NEVER}"]
        )

        assert stated is OrmBuild.NEVER

    def test_the_registered_option_and_the_one_the_build_read_agree(
        self, pytestconfig: pytest.Config
    ):
        """
        Pytest parses the option after the conftests have already built, so the two
        readings of this run's command line must not be able to drift apart.
        """
        registered = pytestconfig.getoption(ORM_BUILD_OPTION)

        expected = None if registered is None else OrmBuild(registered)
        assert OrmBuild.stated_on(sys.argv) is expected


class TestARunStatingAChoiceThatDoesNotExist:
    """
    The conftests read the choice before pytest parses the command line, so a choice
    pytest would reject has to be rejected here as well, rather than crashing the import
    or paying for a build the run is about to abandon.
    """

    def test_the_option_without_a_choice_is_rejected(self):
        with pytest.raises(MissingOrmBuildChoiceError):
            OrmBuild.stated_on([PYTEST_COMMAND, ORM_BUILD_OPTION])

    def test_a_choice_the_option_does_not_offer_is_rejected(self):
        with pytest.raises(UnknownOrmBuildChoiceError):
            OrmBuild.stated_on([PYTEST_COMMAND, f"{ORM_BUILD_OPTION}={UNKNOWN_CHOICE}"])

    def test_a_choice_the_environment_does_not_offer_is_rejected(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [PYTEST_COMMAND])
        monkeypatch.setenv(PytestEnvironmentVariable.ORM_BUILD, UNKNOWN_CHOICE)

        with pytest.raises(UnknownOrmBuildChoiceError):
            OrmBuild.requested()


class TestARunThatBuildsNothing:
    """
    A run told not to build reads whatever interfaces the checkout holds.
    """

    def test_it_leaves_the_interfaces_alone(self, monkeypatch, recorded_builds):
        monkeypatch.delenv(PytestEnvironmentVariable.XDIST_WORKER, raising=False)
        monkeypatch.setattr(
            sys, "argv", [PYTEST_COMMAND, f"{ORM_BUILD_OPTION}={OrmBuild.NEVER}"]
        )

        assert regenerate_orm_interfaces() is False
        assert recorded_builds == []
