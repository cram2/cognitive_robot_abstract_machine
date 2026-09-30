"""
End-to-end tests of the HTTP server: static frontend, scenes and JSON API.
"""

import importlib
import json
import threading
import urllib.error
import urllib.request

import pytest

from cramera import paths
from cramera.knowledge.detected_events import SceneField
from cramera.paths import RECORDING_SCENE_NAME
from cramera.live.recording_storage import SceneDestination

from .conftest import reset_knowledge_base_cache


@pytest.fixture()
def server(fixture_scene):
    """
    The real server on an ephemeral port, bound to the fixture scene.
    """
    from cramera import server as server_module

    importlib.reload(server_module)  # rebind knowledge_module under the fixture env
    httpd = server_module.make_server(0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield "http://localhost:%d" % httpd.server_address[1]
    httpd.shutdown()


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read()


def get_json(url):
    status, body = get(url)
    assert status == 200
    return json.loads(body)


def posted_answer(url, payload):
    """
    The decoded answer to a POST, error responses included.

    :param url: The endpoint to post to.
    :param payload: The JSON-serializable request body.
    """
    return post(url, payload)[1]


def post(url, payload=None, timeout=10):
    request = urllib.request.Request(
        url, method="POST", data=json.dumps(payload or {}).encode()
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


class TestStatic:
    def test_index_is_served(self, server):
        status, body = get(server + "/")
        assert status == 200
        assert b"CRAM Visualization" in body
        assert b'data-slot="left"' in body

    def test_panel_scripts_are_served(self, server):
        for path in (
            "/core/bus.js",
            "/core/registry.js",
            "/config.js",
            "/panels/robot_scene/panel.js",
            "/panels/eql/panel.js",
            "/panels/graph/panel.js",
            "/panels/graph/graph.js",
        ):
            status, _ = get(server + path)
            assert status == 200, path

    def test_scene_bundle_is_served_from_data_dir(self, server):
        scene = get_json(server + "/scenes/fixture/scene.json")
        assert scene["name"] == "fixture"
        index = get_json(server + "/scenes/index.json")
        assert index["default"] == "fixture"

    def test_a_static_asset_may_not_be_stored_by_the_browser(self, server):
        """
        The frontend is edited while a browser holds it open, and the only cache
        validator this server offers is the file's modification time — so a stored copy
        outlives every edit that leaves that time behind the date the copy carries.

        Forbidding the store is what keeps an edited frontend from being served from an
        old page load.
        """
        with urllib.request.urlopen(server + "/config.js", timeout=10) as response:
            assert response.headers["Cache-Control"] == "no-store"

    def test_scene_path_traversal_is_blocked(self, server):
        request = urllib.request.Request(server + "/scenes/../../etc/passwd")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = response.status
        except urllib.error.HTTPError as err:
            status = err.code
        assert status in (403, 404)


class TestScenesTwoRoots:
    """
    A local recording saved under ``CRAMERA_DATA`` must serve exactly like a shared,
    onboarded scene under ``CRAMERA_SCENES`` — the two directories differ here, unlike
    the ``fixture_scene`` fixture where they coincide.
    """

    @pytest.fixture()
    def two_root_server(self, tmp_path, monkeypatch):
        shared = tmp_path / "shared"
        local = tmp_path / "data" / "scenes"
        for directory, name, robot in (
            (shared, "kitchen", "pr2"),
            (local, "my_run", "tracy"),
        ):
            bundle = directory / name
            bundle.mkdir(parents=True)
            (bundle / "scene.json").write_text(
                json.dumps({"name": name, "robot": {"name": robot}, "models": []})
            )
        (shared / "index.json").write_text(
            json.dumps({"default": "kitchen", "scenes": []})
        )
        monkeypatch.setenv("CRAMERA_SCENES", str(shared))
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
        monkeypatch.delenv("CRAMERA_SCENE", raising=False)

        from cramera import server as server_module

        importlib.reload(server_module)
        httpd = server_module.make_server(0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield "http://localhost:%d" % httpd.server_address[1]
        httpd.shutdown()

    def test_a_shared_scene_is_served(self, two_root_server):
        scene = get_json(two_root_server + "/scenes/kitchen/scene.json")
        assert scene["robot"]["name"] == "pr2"

    def test_a_local_only_scene_is_served(self, two_root_server):
        scene = get_json(two_root_server + "/scenes/my_run/scene.json")
        assert scene["robot"]["name"] == "tracy"

    def test_the_index_merges_both_roots(self, two_root_server):
        index = get_json(two_root_server + "/scenes/index.json")
        names = sorted(entry["name"] for entry in index["scenes"])
        assert names == ["kitchen", "my_run"]

    def test_traversal_across_either_root_is_blocked(self, two_root_server):
        request = urllib.request.Request(two_root_server + "/scenes/../../etc/passwd")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = response.status
        except urllib.error.HTTPError as err:
            status = err.code
        assert status in (403, 404)


class TestRecordingApi:
    """
    Saving/discarding a finalized recording must work through the always-on server (this
    fixture never starts a live bridge at all), which is exactly the situation a demo
    process that already exited leaves behind.
    """

    def finalize_on_disk(self, fixture_scene, name="__recording__"):
        bundle = fixture_scene / "scenes" / name
        bundle.mkdir(parents=True)
        (bundle / "scene.json").write_text(
            json.dumps({"name": name, "robot": {"name": "pr2"}, "models": []})
        )
        return bundle

    def test_status_is_idle_without_a_finalized_recording(self, server):
        assert get_json(server + "/api/recording/status") == {"state": "idle"}

    def test_status_is_finalized_once_a_bundle_exists_on_disk(
        self, server, fixture_scene
    ):
        self.finalize_on_disk(fixture_scene)

        assert get_json(server + "/api/recording/status") == {"state": "finalized"}

    def test_save_works_with_no_bridge_involved_at_all(self, server, fixture_scene):
        self.finalize_on_disk(fixture_scene)

        status, body = post(server + "/api/recording/save", {"name": "my_run"})

        assert status == 200
        assert body == {"ok": True, "scene": "my_run"}
        saved = fixture_scene / "scenes" / "my_run" / "scene.json"
        assert json.loads(saved.read_text())["name"] == "my_run"

    def test_discard_works_with_no_bridge_involved_at_all(self, server, fixture_scene):
        bundle = self.finalize_on_disk(fixture_scene)

        status, body = post(server + "/api/recording/discard")

        assert status == 200
        assert body == {"ok": True}
        assert not bundle.exists()

    def replayable_on_disk(self, fixture_scene, frame_count=6):
        """
        A finalized bundle carrying a real trajectory, as a stopped live run leaves one.
        """
        bundle = fixture_scene / "scenes" / paths.RECORDING_SCENE_NAME
        bundle.mkdir(parents=True)
        poses = [[0.1 * index, 0, 0, 0, 0, 0, 1] for index in range(frame_count)]
        (bundle / "scene.json").write_text(
            json.dumps(
                {
                    "name": paths.RECORDING_SCENE_NAME,
                    "robot": {"name": "pr2"},
                    "models": [],
                    "objects": [{"key": "milk.stl", "spawn": poses[0]}],
                    "segments": [
                        {
                            "step": "run",
                            "action": None,
                            "arm": None,
                            "start": 0,
                            "end": frame_count - 1,
                        }
                    ],
                }
            )
        )
        (bundle / "trajectory.json").write_text(
            json.dumps(
                {
                    "framesPerSecond": 20.0,
                    "frames": [{} for _ in poses],
                    "base": [None for _ in poses],
                    "objects": [{"milk.stl": pose} for pose in poses],
                }
            )
        )
        return poses

    def test_save_trims_the_bundle_with_no_bridge_involved_at_all(
        self, server, fixture_scene
    ):
        """
        Trimming has to work off the bundle alone: by the time a run is saved the demo
        process that captured it has usually exited.
        """
        poses = self.replayable_on_disk(fixture_scene)

        status, body = post(
            server + "/api/recording/save",
            {"name": "my_run", "firstFrame": 2, "lastFrame": 4},
        )

        assert status == 200
        assert body == {"ok": True, "scene": "my_run"}
        saved = fixture_scene / "scenes" / "my_run"
        trajectory = json.loads((saved / "trajectory.json").read_text())
        assert trajectory["objects"] == [{"milk.stl": pose} for pose in poses[2:5]]
        [entry] = json.loads((saved / "scene.json").read_text())["objects"]
        assert entry["spawn"] == poses[2]

    def test_save_without_a_trim_keeps_every_frame(self, server, fixture_scene):
        poses = self.replayable_on_disk(fixture_scene)

        post(server + "/api/recording/save", {"name": "my_run"})

        trajectory = json.loads(
            (fixture_scene / "scenes" / "my_run" / "trajectory.json").read_text()
        )
        assert len(trajectory["objects"]) == len(poses)

    def test_save_rejects_a_trim_past_the_recording(self, server, fixture_scene):
        self.replayable_on_disk(fixture_scene, frame_count=3)

        status, body = post(
            server + "/api/recording/save",
            {"name": "my_run", "firstFrame": 0, "lastFrame": 9},
        )

        assert status == 400
        assert body["ok"] is False
        assert not (fixture_scene / "scenes" / "my_run").exists()

    def test_save_can_share_into_the_shared_scenes_root(
        self, server, fixture_scene, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "shared"))
        (tmp_path / "shared").mkdir()
        self.finalize_on_disk(fixture_scene)

        status, body = post(
            server + "/api/recording/save",
            {"name": "my_run", "destination": SceneDestination.SHARED.value},
        )

        assert status == 200
        assert body == {"ok": True, "scene": "my_run"}
        assert (tmp_path / "shared" / "my_run" / "scene.json").is_file()

    def test_sharing_without_a_shared_root_is_reported(self, server, fixture_scene):
        self.finalize_on_disk(fixture_scene)

        status, body = post(
            server + "/api/recording/save",
            {"name": "my_run", "destination": SceneDestination.SHARED.value},
        )

        assert status == 400
        assert body["ok"] is False

    def test_save_without_a_finalized_recording_is_rejected(self, server):
        status, body = post(server + "/api/recording/save", {"name": "my_run"})
        assert status == 400
        assert body["ok"] is False

    def test_save_rejects_an_unsafe_name(self, server, fixture_scene):
        self.finalize_on_disk(fixture_scene)

        status, body = post(server + "/api/recording/save", {"name": "../escape"})

        assert status == 400
        assert body["ok"] is False

    def test_save_rejects_a_name_collision(self, server, fixture_scene):
        self.finalize_on_disk(fixture_scene)

        status, body = post(server + "/api/recording/save", {"name": "fixture"})

        assert status == 409
        assert body["ok"] is False


class TestApi:
    def test_knowledge_overview(self, server):
        pytest.importorskip("krrood")
        payload = get_json(server + "/api/knowledge")
        assert payload["ok"]
        assert any(n["id"] == "cram" for n in payload["nodes"])

    def test_knowledge_overview_presets_are_worded(self, server):
        """
        The overview's presets carry their questions read back as English, so the
        panel's question display has words to show before a query has run.
        """
        pytest.importorskip("krrood")
        payload = get_json(server + "/api/knowledge")
        preset = next(
            entry
            for entry in payload["presets"]
            if entry["text"] == "which robot is this?"
        )
        assert preset["verbalization"]["text"]
        assert "<span" in preset["verbalization"]["html"]

    def test_knowledge_views(self, server):
        pytest.importorskip("krrood")
        for name, expect_live in (
            ("kinematics", None),
            ("plan", None),  # the plan panel knows its own live source
            ("chart", "chart"),
            ("transforms", "transforms"),
        ):
            payload = get_json(server + "/api/knowledge/view?name=" + name)
            assert payload["ok"], name
            assert payload.get("live") == expect_live

    def test_eql_query_roundtrip(self, server):
        pytest.importorskip("krrood")
        request = urllib.request.Request(
            server + "/api/eql",
            data=json.dumps(
                {"code": "the(entity(scene_object).where(scene_object.name == 'milk'))"}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        assert payload["ok"] and payload["count"] == 1

    def test_broken_query_returns_json_error(self, server):
        pytest.importorskip("krrood")
        request = urllib.request.Request(
            server + "/api/eql",
            data=json.dumps({"code": "definitely not python ((("}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
        assert payload["ok"] is False and "error" in payload

    def test_unknown_post_endpoint_is_json_404(self, server):
        request = urllib.request.Request(server + "/api/nope", data=b"{}")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as err:
            status, body = err.code, err.read()
        assert status == 404
        assert json.loads(body)["ok"] is False


class TestVocabularyApi:
    """
    What the query box is told it may name, served from the recorded scene.
    """

    def test_the_vocabulary_offers_the_scene_variables_and_workspace_classes(
        self, server
    ):
        pytest.importorskip("krrood")
        payload = get_json(server + "/api/eql/vocabulary")

        assert payload["ok"]
        offered = {entry["name"]: entry for entry in payload["entries"]}
        assert offered["scene_object"]["kind"] == "variable"
        assert offered["scene_object"]["type"] == "BenchObject"
        assert offered["entity"]["kind"] == "factory"
        # a class of the scanned architecture, which the fixture keeps miniature
        assert offered["Plan"]["kind"] == "class"
        assert offered["Plan"]["module"] == "coraplex.plans.plan"

    def test_the_members_of_a_variable_are_served_for_its_type(self, server):
        pytest.importorskip("krrood")
        payload = get_json(server + "/api/eql/members?name=scene_object")

        assert payload["ok"] and payload["name"] == "scene_object"
        assert "name" in [member["name"] for member in payload["members"]]

    def test_the_members_of_an_unknown_name_are_refused(self, server):
        pytest.importorskip("krrood")
        payload = get_json(server + "/api/eql/members?name=NoSuchType")

        assert payload["ok"] is False
        assert "NoSuchType" in payload["error"]


class TestAskedQuestions:
    """
    A natural-language question — spoken or typed — is matched to the presets the
    recorded scene can answer, and either runs as if its button had been clicked or is
    declined with the sorry reply.
    """

    def test_a_question_is_recognized_and_its_query_runs(self, server):
        """
        The full voice flow, minus the microphone: transcript in, matched preset out,
        the preset's own code answered — exactly what clicking its button runs.
        """
        pytest.importorskip("krrood")
        match = posted_answer(server + "/api/question", {"text": "which robot is this"})

        assert match["ok"] is True
        assert match["matched"] is True
        assert match["preset"]["code"] == "the(entity(robot))"

        answer = posted_answer(server + "/api/eql", {"code": match["preset"]["code"]})
        assert answer["ok"] is True
        assert answer["count"] == 1

    def test_a_paraphrase_is_recognized_too(self, server):
        """
        A reworded question reaches the same preset as the wording on its button.
        """
        pytest.importorskip("krrood")
        asked = posted_answer(server + "/api/question", {"text": "which robot is this"})

        paraphrased = posted_answer(
            server + "/api/question",
            {"text": "can you tell me which robot this is"},
        )

        assert paraphrased["matched"] is True
        assert paraphrased["preset"] == asked["preset"]

    def test_an_unanswerable_question_gets_the_sorry_reply(self, server):
        pytest.importorskip("krrood")
        from cramera.knowledge.question_matching import UNMATCHED_QUESTION_REPLY

        match = posted_answer(
            server + "/api/question", {"text": "what's the weather like today"}
        )

        assert match["ok"] is True
        assert match["matched"] is False
        assert match["reply"] == UNMATCHED_QUESTION_REPLY

    def test_an_empty_question_is_an_error(self, server):
        pytest.importorskip("krrood")
        assert posted_answer(server + "/api/question", {"text": "   "})["ok"] is False

    def test_a_preset_needing_a_running_demo_is_not_offered(self, server):
        """
        A bundle-declared preset ranges over a demo the recording does not have;
        matching it here would hand the panel a query it cannot run.
        """
        pytest.importorskip("krrood")
        bundle_directory = paths.scenes_directory() / "fixture"
        (bundle_directory / "presets.json").write_text(
            json.dumps(
                {
                    "presets": [
                        {
                            "text": "which shapes are inserted?",
                            "code": "an(entity(shape))",
                        }
                    ]
                }
            )
        )
        reset_knowledge_base_cache()

        match = posted_answer(
            server + "/api/question", {"text": "which shapes are inserted"}
        )

        assert match["matched"] is False


# %% the command line


class TestParseArguments:
    def test_no_arguments_serve_the_default_port_and_open_the_page(self):
        from cramera import server as server_module

        options = server_module.parse_arguments([])

        assert options.port == server_module.DEFAULT_PORT
        assert options.open_browser is True

    def test_a_port_argument_is_honored(self):
        from cramera import server as server_module

        assert server_module.parse_arguments(["8123"]).port == 8123

    def test_no_browser_keeps_the_page_closed(self):
        from cramera import server as server_module

        options = server_module.parse_arguments(["--no-browser", "8123"])

        assert options.port == 8123
        assert options.open_browser is False


class TestMainOpensTheViewerPage:
    def run_main(self, monkeypatch, arguments):
        """
        Run ``main`` with the browser recorded and the serve loop cut short.

        :param monkeypatch: The active monkeypatch fixture.
        :param arguments: The command line to run with.
        :return: The URLs the browser was asked to open.
        """
        from cramera import server as server_module

        opened = []
        monkeypatch.setattr(
            server_module.webbrowser, "open", lambda url: opened.append(url)
        )

        real_make_server = server_module.make_server

        def make_short_lived_server(port=0):
            server = real_make_server(0)

            def stop_immediately():
                raise KeyboardInterrupt

            monkeypatch.setattr(server, "serve_forever", stop_immediately)
            return server

        monkeypatch.setattr(server_module, "make_server", make_short_lived_server)
        server_module.main(arguments)
        return opened

    def test_main_opens_the_viewer_page(self, monkeypatch):
        assert self.run_main(monkeypatch, ["8123"]) == ["http://localhost:8123/"]

    def test_no_browser_opts_out(self, monkeypatch):
        assert self.run_main(monkeypatch, ["--no-browser", "8123"]) == []


# %% naming a saved recording


class TestSavingARecordingWithItsNames:
    """
    A person saving a recording says what the run itself cannot: what the environment
    really is, and what the robot was doing.

    Whatever they leave out stays as recorded.
    """

    def saved_scene(self, server, tmp_path, monkeypatch, **given) -> dict:
        """
        A finalized recording saved through the API, read back off disk.

        :param server: The running server's base URL.
        :param tmp_path: The data directory the recording lives in.
        :param monkeypatch: The active monkeypatch fixture.
        :param given: What the person saving it called it.
        """
        recording = tmp_path / "scenes" / RECORDING_SCENE_NAME
        recording.mkdir(parents=True, exist_ok=True)
        (recording / "scene.json").write_text(
            json.dumps({"robot": {"name": "unitreeg1"}, "models": []})
        )

        answer = posted_answer(
            server + "/api/recording/save", dict(name="saved_run", **given)
        )

        assert answer["ok"], answer
        return json.loads(
            (tmp_path / "scenes" / "saved_run" / "scene.json").read_text()
        )

    def test_the_names_travel_into_the_saved_bundle(
        self, server, fixture_scene, monkeypatch
    ):
        scene = self.saved_scene(
            server,
            fixture_scene,
            monkeypatch,
            robot="Unitree G1",
            environment="warehouse",
            task="fetch a wrench",
        )

        assert scene[SceneField.ROBOT_NAME] == "Unitree G1"
        assert scene[SceneField.ENVIRONMENT_NAME] == "warehouse"
        assert scene[SceneField.TASK] == "fetch a wrench"

    def test_a_recording_saved_without_them_keeps_what_it_recorded(
        self, server, fixture_scene, monkeypatch
    ):
        scene = self.saved_scene(server, fixture_scene, monkeypatch)

        assert SceneField.TASK not in scene
        assert scene["robot"]["name"] == "unitreeg1"
