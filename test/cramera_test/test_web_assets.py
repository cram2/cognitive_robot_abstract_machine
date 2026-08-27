"""
Consistency checks of the packaged frontend, plus the node-based JS tests.

The asset checks keep the panel architecture honest: every asset the shell references
must exist, every panel id in config.js must be defined by an included panel script, and
panels must not reach outside their own DOM subtree.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from typing_extensions import List

from cramera.paths import WEB_ROOT

JS_DIR = Path(__file__).parent / "js"

SCRIPT_PATTERN = re.compile(r'<script src="([^"]+)"')
"""
How the shell references the assets that must ship with it.
"""
STYLESHEET_PATTERN = re.compile(r'<link rel="stylesheet" href="([^"]+)"')
IMAGE_PATTERN = re.compile(r'<img[^>]+src="([^"]+)"')
CSS_URL_PATTERN = re.compile(r"url\(['\"]?([^)'\"]+)['\"]?\)")


def read(relative_path: str) -> str:
    """
    The text of one packaged frontend file.
    """
    return (WEB_ROOT / relative_path).read_text(encoding="utf-8")


def panel_scripts() -> List[Path]:
    """
    Every script belonging to a panel, at any depth under ``panels/``.
    """
    return sorted(WEB_ROOT.glob("panels/**/*.js"))


class TestAssetConsistency:
    def test_every_included_script_exists(self):
        for source in SCRIPT_PATTERN.findall(read("index.html")):
            assert (WEB_ROOT / source).is_file(), source

    def test_stylesheet_and_slots_exist(self):
        html = read("index.html")
        for href in STYLESHEET_PATTERN.findall(html):
            assert (WEB_ROOT / href).is_file(), href
        assert 'data-slot="left"' in html and 'data-slot="right"' in html

    def test_every_referenced_image_exists(self):
        """
        A missing image is invisible at runtime, so it has to fail here.
        """
        for source in IMAGE_PATTERN.findall(read("index.html")):
            assert (WEB_ROOT / source).is_file(), source

    def test_every_css_url_resolves(self):
        """
        Backgrounds and fonts referenced by the stylesheet must ship with it.
        """
        for reference in CSS_URL_PATTERN.findall(read("app.css")):
            if reference.startswith(("data:", "http:", "https:", "//")):
                continue
            assert (WEB_ROOT / reference).is_file(), reference

    def test_no_undefined_css_variables(self):
        """
        A ``var(--x)`` with no declaration silently drops the whole property.
        """
        stylesheet = read("app.css")
        declared = set(re.findall(r"(--[\w-]+)\s*:", stylesheet))
        used = set(re.findall(r"var\((--[\w-]+)", stylesheet))
        assert used <= declared, used - declared

    def test_configured_panels_are_defined(self):
        config = read("config.js")
        configured = set(re.findall(r"'([\w-]+)'", config.split("layout", 1)[1]))
        defined = set()
        for panel_js in WEB_ROOT.glob("panels/*/panel.js"):
            defined |= set(
                re.findall(r"Panels\.define\('([\w-]+)'", panel_js.read_text())
            )
        assert configured <= defined, configured - defined

    def test_panels_only_query_their_own_root(self):
        """
        A panel owns its DOM subtree; a document-level lookup couples it to the shell
        and to whichever other panel happens to use the same id.
        """
        for panel_js in panel_scripts():
            source = panel_js.read_text()
            assert "document.getElementById" not in source, panel_js.name
            assert "document.querySelector" not in source, panel_js.name

    def test_no_stale_static_urls(self):
        # the old repo served under /static/; the packaged app is rooted at /
        for panel_js in panel_scripts():
            assert "'static/" not in panel_js.read_text(), panel_js.name


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
class TestJsUnits:
    def run_node(self, name: str) -> None:
        """
        Run one node test file, failing with its output.
        """
        result = subprocess.run(
            ["node", "--test", str(JS_DIR / name)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_bus_and_registry(self):
        self.run_node("test_bus_registry.js")

    def test_graph_status_rendering(self):
        self.run_node("test_graph_status.js")

    def test_graph_panel(self):
        self.run_node("test_graph_panel.js")
