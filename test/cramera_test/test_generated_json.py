"""
Tests of reading and atomically writing generated JSON artifacts.
"""

from __future__ import annotations

import json

from cramera.generated_json import GeneratedJson, write_json_atomically


class TestWriteJsonAtomically:
    def test_the_payload_is_readable_back(self, tmp_path):
        path = tmp_path / "scene.json"

        write_json_atomically(path, {"name": "lab"}, indent=1)

        assert GeneratedJson(path).read() == {"name": "lab"}

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        path = tmp_path / "scene.json"

        write_json_atomically(path, {"name": "lab"})

        assert sorted(p.name for p in tmp_path.iterdir()) == ["scene.json"]

    def test_a_previous_file_is_replaced_rather_than_appended_to(self, tmp_path):
        path = tmp_path / "scene.json"
        path.write_text(json.dumps({"name": "old", "junk": "stale data"}))

        write_json_atomically(path, {"name": "new"})

        assert GeneratedJson(path).read() == {"name": "new"}
