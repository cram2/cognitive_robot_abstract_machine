from __future__ import annotations

import contextlib
import logging

from typing_extensions import Iterator

logger = logging.getLogger(__name__)



@contextlib.contextmanager
def robocasa_version_assertions_relaxed(robocasa_mujoco_version: str = "3.3.1", robocasa_numpy_version: str = "2.2.5") -> Iterator[None]:
    """
    Make RoboCasa's own ``mujoco``/``numpy`` version assertions pass for the duration of
    an ``import robocasa`` (or ``from robocasa...``) statement.

    RoboCasa hard-asserts ``mujoco.__version__ == "3.3.1"`` and ``numpy.__version__ in
    ["2.2.5"]`` at the top of ``robocasa/__init__.py`` and refuses to import otherwise,
    even though its public API does not depend on those exact patch versions. This
    workspace pins other versions for other packages (see
    ``physics_simulators/pyproject.toml``), so importing RoboCasa unmodified fails here.
    This context manager spoofs the two version strings RoboCasa reads while its package
    body runs, then restores the real versions, so the rest of the process keeps seeing
    the versions that are actually installed.

    :param robocasa_mujoco_version: The exact ``mujoco`` version RoboCasa's own
        ``robocasa/__init__.py`` asserts on import.
    :param robocasa_numpy_version: The exact ``numpy`` version RoboCasa's own
        ``robocasa/__init__.py`` asserts on import.
    """
    import mujoco
    import numpy

    real_mujoco_version = mujoco.__version__
    real_numpy_version = numpy.__version__

    if real_mujoco_version != robocasa_mujoco_version:
        logger.warning(
            "Spoofing mujoco.__version__ (%s) as %s so RoboCasa's version assertion "
            "passes.",
            real_mujoco_version,
            robocasa_mujoco_version,
        )
        mujoco.__version__ = robocasa_mujoco_version

    if real_numpy_version != robocasa_numpy_version:
        logger.warning(
            "Spoofing numpy.__version__ (%s) as %s so RoboCasa's version assertion "
            "passes.",
            real_numpy_version,
            robocasa_numpy_version,
        )
        numpy.__version__ = robocasa_numpy_version

    try:
        yield
    finally:
        mujoco.__version__ = real_mujoco_version
        numpy.__version__ = real_numpy_version
