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
import ast

from typing_extensions import ClassVar, Dict, List

from cramera.knowledge.eql_session import EqlSession
from cramera.knowledge.recorded_statecharts import NO_STATECHART
from cramera.mesh_format import MeshFormat
from cramera.paths import LIVE_SCENE_NAME, RECORDING_SCENE_NAME, WEB_ROOT

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


PAGES = ["index.html"]
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


# %% graph canvas layout
class TestGraphCanvasLayout:
    """
    The graph renderer draws into an element the stylesheet has to position: the panel
    stacks it under a tab bar, so without that rule the canvas keeps its own height and
    the graph is cut off at the bottom of the panel.
    """

    ATTACHED_SELECTOR_PATTERN = re.compile(
        r"Graph\.attach\(root\.querySelector\('([^']+)'\)"
    )

    def attached_selector(self) -> str:
        """
        The selector the graph panel resolves its canvas element with.
        """
        [selector] = self.ATTACHED_SELECTOR_PATTERN.findall(
            read("panels/graph/panel.js")
        )
        return selector

    def test_the_panel_markup_declares_the_canvas_it_attaches(self):
        markup_attribute = {".": 'class="%s"', "#": 'id="%s"'}[
            self.attached_selector()[0]
        ]
        assert markup_attribute % self.attached_selector()[1:] in read(
            "panels/graph/panel.js"
        )

    def test_the_stylesheet_positions_the_attached_canvas(self):
        selector = self.attached_selector()
        rules = [
            block
            for block in re.findall(r"^([^{}]+)\{", read("app.css"), re.M)
            if re.search(r"%s(?![\w-])" % re.escape(selector), block)
        ]
        assert rules, selector


# %% mesh format loaders
class TestBinaryGltfLoading:
    """
    The viewer must be able to load the compact mesh format scene bundles are converted
    to, wherever it loads a mesh from.
    """

    SCENE_PANEL = "panels/robot_scene/panel.js"

    def test_the_shell_ships_a_loader_for_it(self):
        included = SCRIPT_PATTERN.findall(read("index.html"))
        assert "vendor/GLTFLoader.js" in included

    def test_a_urdf_mesh_reference_is_dispatched_to_that_loader(self):
        panel = read(self.SCENE_PANEL)
        suffix = MeshFormat.GLB.value.lstrip(".")
        assert r"/\.%s$/i.test(path)" % suffix in panel

    def test_a_published_shape_is_dispatched_to_that_loader(self):
        panel = read(self.SCENE_PANEL)
        suffix = MeshFormat.GLB.value.lstrip(".")
        assert "shapeSpec.format === '%s'" % suffix in panel
        assert "fmt === '%s'" % suffix in panel


# %% plan builder environment choice


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

    def test_graph_canvas_visibility(self):
        self.run_node("test_graph_canvas_visibility.js")

    def test_panel_visibility(self):
        self.run_node("test_panel_visibility.js")

    def test_panel_arrangement(self):
        self.run_node("test_panel_arrangement.js")

    def test_marker_specs(self):
        self.run_node("test_marker_specs.js")

    def test_ambient_occlusion(self):
        self.run_node("test_ambient_occlusion.js")

    def test_marker_settings(self):
        self.run_node("test_marker_settings.js")

    def test_frame_axes(self):
        self.run_node("test_frame_axes.js")

    def test_camera_follow(self):
        self.run_node("test_camera_follow.js")

    def test_timeline_events(self):
        self.run_node("test_timeline_events.js")

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

    def test_recording_mode(self):
        self.run_node("test_recording_mode.js")

    def test_shape_specs(self):
        self.run_node("test_shape_specs.js")


# %% replayed statecharts
class TestPlaybackFrameEvent:
    """
    The statechart tab of a replay follows the playhead, which reaches it only as this
    bus event: the two panels are wired to each other by nothing else.
    """

    EVENT = "scene:frame"

    def test_the_scene_panel_emits_it(self):
        assert "emit('%s'" % self.EVENT in read("panels/robot_scene/panel.js")

    def test_the_graph_panel_listens_for_it(self):
        assert "on('%s'" % self.EVENT in read("panels/graph/panel.js")

    def test_the_bus_contract_documents_it(self):
        assert self.EVENT in read("core/bus.js")


class TestNoStatechartMarker:
    """
    A recorded tick with no statechart executing is marked with the same number on both
    sides of the wire; the frontend would otherwise render that marker as a moment
    index.
    """

    DECLARED_PATTERN = re.compile(r"const NO_STATECHART = (-?\d+)")

    def test_the_frontend_and_the_backend_mark_it_the_same(self):
        [declared] = self.DECLARED_PATTERN.findall(read("panels/graph/panel.js"))
        assert int(declared) == NO_STATECHART


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


class TestRecordingSceneName:
    """
    Same wiring as :class:`TestLiveSceneName`, for the reserved scene a finalized live
    recording is bundled under.
    """

    RECORDING_SCENE_NAME_PATTERN = re.compile(r"const SCENE_NAME = '([^']+)'")

    def test_the_frontend_and_the_backend_name_the_same_scene(self):
        [declared] = self.RECORDING_SCENE_NAME_PATTERN.findall(
            read("core/recording-mode.js")
        )
        assert declared == RECORDING_SCENE_NAME


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


class TestEveryLoadedModuleHasAConsumer:
    """
    A page that loads a module nobody calls looks finished and does nothing: the
    highlight arrow shipped that way once, drawn by no panel.
    """

    CONSUMED_GLOBALS: ClassVar[Dict[str, str]] = {
        "core/highlight_arrow.js": "HighlightArrow",
        "core/answer_table.js": "AnswerTable",
        "core/question_display.js": "QuestionDisplay",
        "core/voice.js": "VoiceCapture",
        "core/completion.js": "Completion",
        "core/query_source.js": "QuerySource",
        "core/preset_groups.js": "PresetGroups",
        "core/folding.js": "Folding",
        "core/replay.js": "Replay",
        "core/camera-follow.js": "CameraFollow",
    }
    """
    The global each of these modules defines, which some other script has to call.
    """

    def callers_of(self, name: str, defined_in: str) -> List[str]:
        """
        Every script other than the module itself that names ``name``.

        :param name: The global the module defines.
        :param defined_in: Path of the module defining it, relative to the web root.
        """
        return [
            str(script.relative_to(WEB_ROOT))
            for script in sorted(WEB_ROOT.glob("**/*.js"))
            if script != WEB_ROOT / defined_in
            and script.parts[-2] != "vendor"
            and ("%s." % name) in script.read_text(encoding="utf-8")
        ]

    def test_every_loaded_module_is_called_by_something(self):
        unused = {
            module: name
            for module, name in self.CONSUMED_GLOBALS.items()
            if not self.callers_of(name, module)
        }

        assert unused == {}

    def test_the_highlight_arrow_is_drawn_by_the_scene(self):
        assert "panels/robot_scene/panel.js" in self.callers_of(
            "HighlightArrow", "core/highlight_arrow.js"
        )


class TestTheViewerServesWithoutTheFullStack:
    """
    The server and the recorded viewer are meant to run on a machine without the full
    CRAM stack -- ``cramera/requirements.txt`` says so, and krrood is imported lazily
    for exactly that reason.

    segmind is a ROS package: importing it needs an overlay
    sourced, so anything the server imports on its way up must not reach it.
    """

    SERVED_MODULES: ClassVar[List[str]] = [
        "cramera/src/cramera/knowledge/detected_events.py",
        "cramera/src/cramera/knowledge/presets.py",
        "cramera/src/cramera/knowledge/eql_session.py",
        "cramera/src/cramera/knowledge/query_runner.py",
        "cramera/src/cramera/knowledge/knowledge_base.py",
        "cramera/src/cramera/onboard/scene_index.py",
    ]
    """
    Modules the recorded server pulls in when it starts.
    """

    def imported_packages(self, source: Path) -> List[str]:
        """
        The top-level packages one module imports.

        :param source: Path of the module, relative to the repository.
        """
        tree = ast.parse(source.read_text(encoding="utf-8"))
        packages = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                packages.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                packages.append(node.module.split(".")[0])
        return packages

    def test_nothing_the_server_imports_reaches_segmind(self):
        repository = Path(__file__).resolve().parents[2]
        reaching = {
            module: sorted(
                package
                for package in self.imported_packages(repository / module)
                if package == "segmind"
            )
            for module in self.SERVED_MODULES
        }

        assert {module: found for module, found in reaching.items() if found} == {}


# %% graph tab order
class TestGraphTabOrder:
    """
    The graph panel declares its tabs twice — as buttons and as the ``TABS`` table it
    resolves urls and the opening tab from — so the two have to stay in one order.
    """

    BUTTON_PATTERN = re.compile(r'<button data-view="(\w+)"(\s+class="active")?')
    TABLE_PATTERN = re.compile(r"const TABS = \{(.*?)\n  \};", re.S)

    def buttons(self) -> List[str]:
        """
        The tab names in the order the panel renders their buttons in.
        """
        return [name for name, _ in self.BUTTON_PATTERN.findall(self.source())]

    def source(self) -> str:
        """
        The graph panel's script.
        """
        return read("panels/graph/panel.js")

    def test_the_table_lists_the_tabs_in_the_rendered_order(self):
        [table] = self.TABLE_PATTERN.findall(self.source())
        assert re.findall(r"^\s+(\w+):", table, re.M) == self.buttons()

    def test_the_panel_opens_on_the_first_tab(self):
        active = [
            name
            for name, is_active in self.BUTTON_PATTERN.findall(self.source())
            if is_active
        ]
        assert active == self.buttons()[:1]

    def test_the_tabs_read_from_what_was_planned_to_where_it_moved(self):
        assert self.buttons() == [
            "knowledge",
            "plan",
            "chart",
            "kinematics",
            "transforms",
        ]
