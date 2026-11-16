import importlib
import pkgutil

import pytest

from giskardpy.middleware.ros2 import scripts

# %% script discovery


def script_module_names() -> list[str]:
    """
    :return: The dotted name of every module below the scripts package.
    """
    return [
        module.name
        for module in pkgutil.walk_packages(
            scripts.__path__, prefix=f"{scripts.__name__}."
        )
        if not module.ispkg
    ]


# %% importing scripts


@pytest.mark.parametrize("module_name", script_module_names())
def test_script_imports(module_name: str) -> None:
    """
    A script is only started on the robot, so a name it imports that no longer exists
    would otherwise go unnoticed until then.
    """
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name
