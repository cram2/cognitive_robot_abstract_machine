"""
Tests for asking a query what segmind detected.
"""

import json
from dataclasses import dataclass
from datetime import datetime

import pytest
from typing_extensions import List

pytest.importorskip("krrood", reason="EQL requires krrood")

from segmind.datastructures.events import (  # noqa: E402
    ContactEvent,
    DetectionEvent,
    PickUpEvent,
)
from segmind.event_logger import EventLogger  # noqa: E402
from semantic_digital_twin.datastructures.prefixed_name import (
    PrefixedName,
)  # noqa: E402
from semantic_digital_twin.world_description.geometry import Box, Scale  # noqa: E402
from semantic_digital_twin.world_description.shape_collection import (  # noqa: E402
    ShapeCollection,
)
from semantic_digital_twin.spatial_types import (  # noqa: E402
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World  # noqa: E402
from semantic_digital_twin.world_description.connections import (  # noqa: E402
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body  # noqa: E402

from cramera.knowledge.query_runner import EqlQueryRunner  # noqa: E402
from cramera.knowledge.detected_events import (  # noqa: E402
    DetectedEventRecord,
    EventField,
    SceneField,
)
from cramera.live.detections import (  # noqa: E402
    DetectedEvents,
    detectable_event_types,
    record_of,
)
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase  # noqa: E402
from cramera.knowledge.eql_session import EqlSession  # noqa: E402
from cramera.knowledge.presets import Preset  # noqa: E402
from cramera.knowledge.queryable_knowledge import (  # noqa: E402
    QueryableKnowledge,
    QueryScope,
)
from cramera.live.bridge import Bridge  # noqa: E402
from cramera.live.query import LiveQuerySource  # noqa: E402

from .conftest import reset_knowledge_base_cache  # noqa: E402
from .test_live_bridge import world_with  # noqa: E402


def collidable_body(name: str) -> Body:
    """
    A body a detector can produce an event about: one it can measure a bounding box of.

    :param name: The body's local name.
    """
    return Body(
        name=PrefixedName(name, prefix="demo"),
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
    )


# %% a logger holding real detections


@pytest.fixture()
def detections():
    """
    An event logger holding one pick-up and one contact, on bodies of a real world.
    """
    milk = collidable_body("milk")
    table = collidable_body("table")
    world_with(milk, table)
    logger = EventLogger()
    logger.timeline.extend(
        [
            PickUpEvent(
                tracked_object=milk,
                with_object=table,
                timestamp=datetime(2026, 8, 31, 12, 0, 0),
            ),
            ContactEvent(
                tracked_object=milk,
                with_object=table,
                timestamp=datetime(2026, 8, 31, 12, 0, 5),
            ),
        ]
    )
    return logger


# %% one detection as a record


class TestRecordOfAnEvent:
    def test_a_record_names_the_event_and_what_it_happened_to(self, detections):
        record = record_of(detections.timeline[0])

        assert record.event_type == PickUpEvent.__name__
        assert record.tracked_object == "milk"
        assert record.with_object == "table"
        assert record.timestamp == detections.timeline[0].timestamp

    def test_a_record_is_named_by_what_happened_to_what(self, detections):
        assert record_of(detections.timeline[0]).name == ("milk PickUpEvent")

    def test_an_event_without_a_second_object_names_only_the_first(self):
        body = collidable_body("spoon")
        world_with(body)

        record = record_of(PickUpEvent(tracked_object=body))

        assert record.tracked_object == "spoon" and record.with_object is None


class TestRecordableEventTypes:
    def test_every_leaf_event_type_can_be_asked_for(self):
        types = detectable_event_types()

        assert PickUpEvent.__name__ in types and ContactEvent.__name__ in types

    def test_a_type_that_is_only_a_base_of_others_is_not_asked_for(self):
        subclassed = {
            event_type.__name__
            for event_type in DetectionEvent.__subclasses__()
            if event_type.__subclasses__()
        }

        assert subclassed and not subclassed & set(detectable_event_types())

    def test_the_types_are_offered_in_a_stable_order(self):
        types = detectable_event_types()

        assert list(types) == sorted(types)


# %% the detections as queryable knowledge


class TestDetectedEventsKnowledge:
    def test_the_knowledge_is_of_the_detected_events_scope(self, detections):
        knowledge = DetectedEvents(logger=detections).knowledge()

        assert knowledge.scope is QueryScope.DETECTED_EVENTS

    def test_the_domain_is_the_event_variable_a_question_names(self, detections):
        [domain] = DetectedEvents(logger=detections).knowledge().domains

        assert domain.name == "event"
        assert domain.entity_type is DetectedEventRecord

    def test_the_domain_holds_a_record_per_detection(self, detections):
        [domain] = DetectedEvents(logger=detections).knowledge().domains

        assert [record.event_type for record in domain.objects] == [
            PickUpEvent.__name__,
            ContactEvent.__name__,
        ]

    def test_a_detection_made_after_the_last_question_is_answered_too(self, detections):
        events = DetectedEvents(logger=detections)
        before = len(events.knowledge().domains[0].objects)

        detections.timeline.append(PickUpEvent(tracked_object=collidable_body("cup")))

        assert len(events.knowledge().domains[0].objects) == before + 1


class TestAskingForOneKindOfEvent:
    def test_the_offered_question_answers_with_the_events_of_the_type_it_names(
        self, detections
    ):
        events = DetectedEvents(logger=detections)
        [pick_ups] = [
            preset
            for preset in events.unlisted_presets()
            if PickUpEvent.__name__ in preset.code
        ]
        runner = EqlQueryRunner(domains=events.knowledge().domains)

        answered = runner.run(pick_ups.code)

        assert [row["__entity__"] for row in answered.rows] == ["milk PickUpEvent"]
        assert [row["event_type"] for row in answered.rows] == [PickUpEvent.__name__]


# %% the questions the panel offers


class TestOfferedQuestions:
    def test_every_recordable_type_can_be_asked_for_by_name(self, detections):
        offered = DetectedEvents(logger=detections).unlisted_presets()

        assert "give me all pick up events" in [preset.text for preset in offered]

    def test_an_offered_question_asks_the_detected_events(self, detections):
        [pick_ups] = [
            preset
            for preset in DetectedEvents(logger=detections).unlisted_presets()
            if preset.text == "give me all pick up events"
        ]

        assert pick_ups.scope is QueryScope.DETECTED_EVENTS
        assert PickUpEvent.__name__ in pick_ups.code

    def test_the_panel_offers_a_button_for_what_was_detected(self, detections):
        offered = DetectedEvents(logger=detections).presets()

        assert [preset.scope for preset in offered] == [QueryScope.DETECTED_EVENTS]
        assert offered[0].code.count("event") >= 1


# %% a demo offering its detections to the bridge


@dataclass
class DetectingDemo(LiveQuerySource):
    """
    A demo that ticks segmind detectors and offers what they saw alongside nothing else.
    """

    detections: DetectedEvents
    """
    The detections this demo offers to be questioned about.
    """

    def title(self) -> str:
        return "detecting demo"

    def knowledge(self) -> List[QueryableKnowledge]:
        return [self.detections.knowledge()]

    def presets(self) -> List[Preset]:
        return self.detections.presets()

    def unlisted_presets(self) -> List[Preset]:
        return self.detections.unlisted_presets()


class TestAskingTheBridge:
    @pytest.fixture()
    def bridge(self, detections):
        """
        A bridge a detecting demo has registered itself with.
        """
        bridge = Bridge()
        bridge.register_query_source(
            DetectingDemo(detections=DetectedEvents(detections))
        )
        return bridge

    def test_the_bridge_offers_the_detected_events_scope(self, bridge):
        assert bridge.query_scopes() == [QueryScope.DETECTED_EVENTS]

    def test_a_query_of_that_scope_is_answered_from_the_detections(self, bridge):
        answered = bridge.run_query(
            "an(entity(event).where(event.event_type == 'PickUpEvent'))",
            QueryScope.DETECTED_EVENTS,
        )

        assert [row["__entity__"] for row in answered.rows] == ["milk PickUpEvent"]

    def test_asking_for_pick_up_events_out_loud_runs_that_query(self, bridge):
        matched = bridge.match_question("give me all pick up events")

        assert matched.preset is not None
        assert matched.preset.text == "give me all pick up events"
        assert matched.preset.scope is QueryScope.DETECTED_EVENTS

    def test_the_event_variable_is_offered_to_the_query_box(self, bridge):
        offered = bridge.query_variables(QueryScope.DETECTED_EVENTS)

        assert offered == ["event"]


# %% detections carried by a recorded bundle


class TestRecordAsBundlePayload:
    def test_a_record_round_trips_through_its_payload(self, detections):
        record = record_of(detections.timeline[0])

        assert DetectedEventRecord.of_payload(record.to_payload()) == record

    def test_a_payload_states_the_moment_in_a_form_json_can_hold(self, detections):
        payload = record_of(detections.timeline[0]).to_payload()

        assert payload[EventField.TIMESTAMP] == (
            detections.timeline[0].timestamp.isoformat()
        )

    def test_a_payload_without_a_second_body_reads_back_as_none(self):
        body = collidable_body("spoon")
        world_with(body)
        record = record_of(PickUpEvent(tracked_object=body))

        assert DetectedEventRecord.of_payload(record.to_payload()).with_object is None


# %% asking a recorded scene what was detected


@pytest.fixture()
def scene_with_detections(fixture_scene, monkeypatch):
    """
    The fixture scene bundle, re-recorded with two detections in it.
    """
    scene_path = fixture_scene / "scenes" / "fixture" / "scene.json"
    scene = json.loads(scene_path.read_text())
    scene[SceneField.DETECTED_EVENTS] = [
        DetectedEventRecord(
            name="milk PickUpEvent",
            event_type=PickUpEvent.__name__,
            timestamp=datetime(2026, 8, 31, 12, 0, 0),
            tracked_object="milk",
        ).to_payload(),
        DetectedEventRecord(
            name="milk PlacingEvent",
            event_type="PlacingEvent",
            timestamp=datetime(2026, 8, 31, 12, 0, 9),
            tracked_object="milk",
            with_object="table",
        ).to_payload(),
    ]
    scene_path.write_text(json.dumps(scene))
    reset_knowledge_base_cache()
    yield fixture_scene
    reset_knowledge_base_cache()


class TestRecordedDetections:
    def test_the_knowledge_base_reads_the_detections_of_the_bundle(
        self, scene_with_detections
    ):
        events = EpisodeKnowledgeBase.of_scene(None).detected_events

        assert [record.event_type for record in events] == [
            PickUpEvent.__name__,
            "PlacingEvent",
        ]

    def test_a_scene_recorded_without_detectors_has_no_detections(self, fixture_scene):
        assert EpisodeKnowledgeBase.of_scene(None).detected_events == []

    def test_a_query_of_the_recorded_scene_may_name_the_events(
        self, scene_with_detections
    ):
        answered = EqlSession.of_scene(None).run(
            "an(entity(event).where(event.event_type == 'PickUpEvent'))"
        )

        assert [row["__entity__"] for row in answered.rows] == ["milk PickUpEvent"]
