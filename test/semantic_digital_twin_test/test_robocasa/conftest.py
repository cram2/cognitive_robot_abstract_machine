from semantic_digital_twin.adapters.robocasa_dataset.mujoco_compat import (
    robocasa_version_assertions_relaxed,
)


def pytest_configure(config):
    """
    Import RoboCasa once, with its version assertions relaxed, before this package's
    test modules are collected.

    The test modules call ``pytest.importorskip("robocasa", ...)`` as their first
    statement, which only catches :class:`ImportError` -- not the
    :class:`AssertionError` RoboCasa's own ``__init__.py`` raises when the installed
    ``mujoco``/``numpy`` versions do not match what it hard-codes. Pre-importing RoboCasa
    here, with those assertions relaxed, caches the module in ``sys.modules`` so the
    later ``pytest.importorskip`` call just finds it there and never re-runs its
    ``__init__.py``. If RoboCasa is genuinely not installed, this is a no-op and
    ``pytest.importorskip`` skips the tests as usual.
    """
    try:
        with robocasa_version_assertions_relaxed():
            import robocasa  # noqa: F401
    except ImportError:
        pass
