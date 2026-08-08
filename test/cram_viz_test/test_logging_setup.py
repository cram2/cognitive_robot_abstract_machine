"""
The entry points must actually emit their progress output.

A library that calls :func:`logging.basicConfig` at import time installs a root
handler, which turns every later ``basicConfig`` call into a silent no-op — so the
console scripts would print none of their startup output. ``cram_viz`` therefore
leaves logging to the entry points, and they configure it with ``force=True``
because other CRAM packages do configure the root logger on import.
"""

import logging
import socketserver
import subprocess
import sys

import pytest

from cram_viz import server

IMPORT_AND_REPORT_ROOT_HANDLERS = (
    "import logging, cram_viz; print(logging.getLogger().handlers)"
)


class TestImportDoesNotConfigureLogging:
    def test_import_adds_no_root_handler(self):
        """
        Importing the package must not install a handler on the root logger.
        """
        result = subprocess.run(
            [sys.executable, "-c", IMPORT_AND_REPORT_ROOT_HANDLERS],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "[]"

    def test_package_logger_propagates(self):
        """
        The package logger must propagate rather than carry its own handler.
        """
        logger = logging.getLogger("cram_viz")
        assert logger.handlers == []
        assert logger.propagate is True


class TestEntryPointEmitsStartupOutput:
    @pytest.fixture()
    def isolated_logging(self):
        """
        Restore the root logger afterwards.

        ``main`` reconfigures it with ``force=True``, which would otherwise leave a
        handler pointing at this test's captured stream for the rest of the session.
        """
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        yield
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)

    @pytest.fixture()
    def server_that_returns_immediately(self, monkeypatch):
        """
        Neutralize the serve loop so ``server.main`` runs to completion.
        """
        monkeypatch.setattr(
            socketserver.ThreadingTCPServer, "serve_forever", lambda self: None
        )

    def test_banner_survives_a_preconfigured_root_logger(
        self, isolated_logging, server_that_returns_immediately, capsys, fixture_scene
    ):
        """
        The startup banner must appear even when logging was already configured.
        """
        logging.basicConfig(format="%(levelname)s %(message)s")
        server.main(["0"])
        assert "cram_viz running at" in capsys.readouterr().err
