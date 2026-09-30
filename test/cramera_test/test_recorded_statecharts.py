"""
Tests of the motion statecharts a live run records, and of reading them back for a
replay.
"""

from __future__ import annotations

from cramera.knowledge.recorded_statecharts import (
    NO_STATECHART,
    RecordedStatecharts,
    STATECHART_FILE,
)
from cramera.live.chart_structure import (
    ChartEdgeEntry,
    ChartNodeEntry,
    ChartSnapshot,
    ObservationName,
)
from cramera.generated_json import write_json_atomically


def node(
    identifier: str,
    life_cycle: str = "RUNNING",
    observation: ObservationName = ObservationName.UNKNOWN,
    parent: str = None,
) -> ChartNodeEntry:
    return ChartNodeEntry(
        id=identifier,
        name=identifier.upper(),
        class_name="CartesianPose",
        parent=parent,
        life_cycle=life_cycle,
        observation=observation,
    )


def snapshot(
    signature: str = "c1",
    title: str = "Reach",
    life_cycles=("RUNNING", "NOT_STARTED"),
) -> ChartSnapshot:
    return ChartSnapshot(
        signature=signature,
        title=title,
        nodes=[
            node("goal", life_cycles[0]),
            node("reached", life_cycles[1], parent="goal"),
        ],
        edges=[ChartEdgeEntry(source="goal", target="reached", kind="START")],
    )


class TestRecordingSnapshots:
    def test_a_tick_without_a_statechart_maps_to_none(self):
        recorded = RecordedStatecharts.of_snapshots([None, None])

        assert recorded.moment_of_frame == [NO_STATECHART, NO_STATECHART]
        assert recorded.is_empty()

    def test_every_frame_maps_to_the_moment_it_was_in(self):
        recorded = RecordedStatecharts.of_snapshots(
            [
                None,
                snapshot(life_cycles=("RUNNING", "NOT_STARTED")),
                snapshot(life_cycles=("DONE", "DONE")),
            ]
        )

        assert recorded.moment_of_frame == [NO_STATECHART, 0, 1]
        assert recorded.moments[1].life_cycles == ["DONE", "DONE"]

    def test_one_structure_is_kept_however_many_ticks_ran_on_it(self):
        recorded = RecordedStatecharts.of_snapshots([snapshot()] * 5)

        assert len(recorded.charts) == 1
        assert recorded.charts[0].signature == "c1"
        assert [entry.id for entry in recorded.charts[0].nodes] == ["goal", "reached"]

    def test_identical_ticks_share_one_moment(self):
        recorded = RecordedStatecharts.of_snapshots([snapshot()] * 3)

        assert len(recorded.moments) == 1
        assert recorded.moment_of_frame == [0, 0, 0]

    def test_a_recompiled_statechart_is_recorded_beside_the_first(self):
        recorded = RecordedStatecharts.of_snapshots(
            [snapshot(), snapshot(signature="c2", title="Place")]
        )

        assert [chart.title for chart in recorded.charts] == ["Reach", "Place"]
        assert [moment.chart for moment in recorded.moments] == [0, 1]

    def test_the_same_structure_under_a_new_action_is_its_own_chart(self):
        """
        Two actions can compile the identical motion group; the statechart shown while
        replaying each must still carry the action it belongs to.
        """
        recorded = RecordedStatecharts.of_snapshots(
            [snapshot(), snapshot(title="Place")]
        )

        assert [chart.title for chart in recorded.charts] == ["Reach", "Place"]

    def test_observations_are_recorded_per_moment(self):
        recorded = RecordedStatecharts.of_snapshots([snapshot()])

        assert recorded.moments[0].observations == [
            ObservationName.UNKNOWN.value,
            ObservationName.UNKNOWN.value,
        ]


class TestWireShape:
    def test_a_recording_survives_a_round_trip(self):
        recorded = RecordedStatecharts.of_snapshots(
            [None, snapshot(), snapshot(life_cycles=("DONE", "DONE"))]
        )

        assert RecordedStatecharts.of_payload(recorded.to_payload()) == recorded

    def test_an_edge_is_written_with_the_keys_the_viewer_reads(self):
        payload = RecordedStatecharts.of_snapshots([snapshot()]).to_payload()

        assert payload["charts"][0]["edges"] == [
            {"from": "goal", "to": "reached", "kind": "START"}
        ]

    def test_an_unreadable_payload_reads_back_as_nothing_recorded(self):
        assert RecordedStatecharts.of_payload([]).is_empty()


def bundle(scenes_root) -> None:
    """
    A minimal scene bundle named ``run``, which a scene name only resolves to once its
    ``scene.json`` is there.
    """
    (scenes_root / "run").mkdir()
    write_json_atomically(scenes_root / "run" / "scene.json", {"name": "run"})


class TestReadingABundle:
    def test_a_bundle_without_statecharts_has_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path))
        bundle(tmp_path)

        assert RecordedStatecharts.of_scene("run").is_empty()

    def test_the_statecharts_written_beside_a_scene_are_read_back(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path))
        bundle(tmp_path)
        recorded = RecordedStatecharts.of_snapshots([snapshot()])
        write_json_atomically(tmp_path / "run" / STATECHART_FILE, recorded.to_payload())

        assert RecordedStatecharts.of_scene("run") == recorded
