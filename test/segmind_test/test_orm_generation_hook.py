from pathlib import Path

from .conftest import pytest_configure

# %% xdist worker guard

ORMATIC_INTERFACE = (
    Path(__file__).resolve().parents[2]
    / "segmind"
    / "src"
    / "segmind"
    / "orm"
    / "ormatic_interface.py"
)


class TestOrmGenerationSkippedOnXdistWorkers:
    """
    The ORM interface must be generated once, by the xdist controller only.
    Concurrent writers truncate the file while another process formats it.
    """

    def test_worker_leaves_interface_untouched(self, monkeypatch):
        monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
        before = ORMATIC_INTERFACE.stat().st_mtime_ns

        pytest_configure(config=None)

        assert ORMATIC_INTERFACE.stat().st_mtime_ns == before
