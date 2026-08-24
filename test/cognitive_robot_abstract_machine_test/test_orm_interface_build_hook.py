import pytest

from cognitive_robot_abstract_machine.orm_interfaces import WORKSPACE_ORM_INTERFACES

from ..orm_interface_build import regenerate_orm_interfaces
from ..pytest_environment import PytestEnvironmentVariable


@pytest.fixture()
def recorded_builds(monkeypatch) -> list:
    """
    Replace the real build with a recorder, so the guard can be exercised without paying
    for a whole regeneration.

    :return: The list every build appends to.
    """
    builds = []
    monkeypatch.setattr(
        WORKSPACE_ORM_INTERFACES, "regenerate", lambda: builds.append("built")
    )
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
