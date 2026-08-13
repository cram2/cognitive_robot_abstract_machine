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

from cramera.knowledge.eql_session import EqlSession
from cramera.paths import LIVE_SCENE_NAME, WEB_ROOT

JS_DIR = Path(__file__).parent / "js"

SCRIPT_PATTERN = re.compile(r'<script src="([^"]+)"')
"""
How the shell references the assets that must ship with it.
"""

STYLESHEET_PATTERN = re.compile(r'<link rel="stylesheet" href="([^"]+)"')
IMAGE_PATTERN = re.compile(r'<img[^>]+src="([^"]+)"')
CSS_URL_PATTERN = re.compile(r"url\(['\"]?([^)'\"]+)['\"]?\)")

#: the EQL variables the query panel advertises: the ``vars:`` list in its placeholder,
#: and every bare identifier it marks up as ``<code>``
ADVERTISED_VARIABLES_PATTERN = re.compile(r"vars: ([a-z_, ]+)")
MARKED_UP_IDENTIFIER_PATTERN = re.compile(r"<code>([a-z_]+)</code>")

#: the ready-to-run query the panel shows in its empty input box
PLACEHOLDER_QUERY_PATTERN = re.compile(r'placeholder="(the\(entity.*?\)\))')


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


PAGES = ["index.html", "models.html"]
"""
Every page the server ships; each must reference only files that exist.
"""


class TestAssetConsistency:
    def test_every_included_script_exists(self):
        for page in PAGES:
            for source in SCRIPT_PATTERN.findall(read(page)):
                assert (WEB_ROOT / source).is_file(), source

    def test_stylesheet_and_slots_exist(self):
        for page in PAGES:
            for href in STYLESHEET_PATTERN.findall(read(page)):
                assert (WEB_ROOT / href).is_file(), href
        html = read("index.html")
        assert 'data-slot="left"' in html and 'data-slot="right"' in html

    def test_the_pages_link_each_other(self):
        """
        The Scene and Models tabs switch through topbar links; a broken link strands the
        user on one page.
        """
        assert 'href="models.html"' in read("index.html")
        assert 'href="index.html"' in read("models.html")

    def test_every_referenced_image_exists(self):
        """
        A missing image is invisible at runtime, so it has to fail here.
        """
        for page in PAGES:
            for source in IMAGE_PATTERN.findall(read(page)):
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

    def test_model_constraints(self):
        self.run_node("test_model_constraints.js")

    def test_panel_visibility(self):
        self.run_node("test_panel_visibility.js")

    def test_panel_arrangement(self):
        self.run_node("test_panel_arrangement.js")

    def test_marker_specs(self):
        self.run_node("test_marker_specs.js")

    def test_marker_settings(self):
        self.run_node("test_marker_settings.js")

    def test_response_util(self):
        self.run_node("test_response_util.js")

    def test_scene_context(self):
        self.run_node("test_scene_context.js")

    def test_scene_picker(self):
        self.run_node("test_scene_picker.js")

    def test_collada_mesh(self):
        self.run_node("test_collada_mesh.js")

    def test_environment_theme(self):
        self.run_node("test_environment_theme.js")

    def test_split_sizing(self):
        self.run_node("test_split_sizing.js")

    def test_split_resize(self):
        self.run_node("test_split_resize.js")

    def test_graph_gestures(self):
        self.run_node("test_graph_gestures.js")

    def test_joint_routing(self):
        self.run_node("test_joint_routing.js")

    def test_live_mode(self):
        self.run_node("test_live_mode.js")

    def test_shape_specs(self):
        self.run_node("test_shape_specs.js")


class TestLiveSceneName:
    """
    The reserved live scene is named on both sides of the wire: the bridge bundles the
    running demo into it, and the frontend decides from that name whether the live pose
    stream may attach at all.

    A rename on one side alone would leave the viewer treating the live scene as an
    ordinary recording.
    """

    LIVE_SCENE_NAME_PATTERN = re.compile(r"const SCENE_NAME = '([^']+)'")

    def test_the_frontend_and_the_backend_name_the_same_scene(self):
        [declared] = self.LIVE_SCENE_NAME_PATTERN.findall(read("core/live-mode.js"))
        assert declared == LIVE_SCENE_NAME


class TestQueryPanelHints:
    """
    The EQL panel hard-codes the variable names it tells users to type, so a rename in
    the EQL namespace silently leaves the panel advertising names that no longer
    resolve.
    """

    def advertised_variables(self) -> List[str]:
        """
        Every EQL variable name the query panel offers the user.
        """
        panel = read("panels/eql/panel.js")
        [listed] = ADVERTISED_VARIABLES_PATTERN.findall(panel)
        names = [name.strip() for name in listed.split(",")]
        return sorted(set(names) | set(MARKED_UP_IDENTIFIER_PATTERN.findall(panel)))

    def test_every_advertised_variable_exists_in_the_namespace(self, fixture_scene):
        namespace = EqlSession.of_active_scene().namespace()
        assert self.advertised_variables()
        for name in self.advertised_variables():
            assert name in namespace, name

    def test_the_placeholder_query_runs(self, fixture_scene):
        """
        The query shown in the empty input box must be one a user can actually run.
        """
        placeholder = PLACEHOLDER_QUERY_PATTERN.search(read("panels/eql/panel.js"))
        assert placeholder is not None
        result = EqlSession.of_active_scene().run(
            placeholder.group(1).replace("\\'", "'")
        )
        assert result.ok
