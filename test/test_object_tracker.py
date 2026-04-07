from unittest.mock import MagicMock
from semantic_digital_twin.world_description.world_entity import Body
from segmind.datastructures.object_tracker import ObjectTracker, ObjectTrackerFactory
from segmind.datastructures.events import Event, ContactEvent, PickUpEvent, PlacingEvent
import pytest
from datetime import timedelta
import time


class MockEvent(Event):
    def __init__(self, timestamp=None):
        if timestamp is not None:
            self.timestamp = timestamp
        else:
            self.timestamp = time.time()
        
    def __eq__(self, other):
        return isinstance(other, MockEvent) and self.timestamp == other.timestamp
    
    def __hash__(self):
        return hash(self.timestamp)
    
    def __str__(self):
        return f"MockEvent(timestamp={self.timestamp})"


class TestObjectTracker:
    
    @pytest.fixture
    def body(self):
        mock_body = MagicMock(spec=Body)
        mock_body.name = "test_body"
        return mock_body

    @pytest.fixture
    def tracker(self, body):
        return ObjectTracker(body=body, context=None, _event_history=[])

    def test_add_event(self, tracker):
        event = MockEvent()
        tracker.add_event(event)
        assert tracker.get_event_history() == [event]
        assert tracker.get_latest_event() == event

    def test_clear_event_history(self, tracker):
        event = MockEvent()
        tracker.add_event(event)
        tracker.clear_event_history()
        assert tracker.get_event_history() == []

    def test_get_latest_event_of_type(self, tracker, body):
        # We need to use real event classes or mocks that isinstance can check
        contact_event = MagicMock(spec=ContactEvent)
        contact_event.timestamp = 10.0
        pickup_event = MagicMock(spec=PickUpEvent)
        pickup_event.timestamp = 20.0
        
        tracker.add_event(contact_event)
        tracker.add_event(pickup_event)
        
        assert tracker.get_latest_event_of_type(ContactEvent) == contact_event
        assert tracker.get_latest_event_of_type(PickUpEvent) == pickup_event

    def test_get_events_between_timestamps(self, tracker):
        e1 = MockEvent(timestamp=10.0)
        e2 = MockEvent(timestamp=20.0)
        e3 = MockEvent(timestamp=30.0)
        
        tracker.add_event(e1)
        tracker.add_event(e2)
        tracker.add_event(e3)
        
        events = tracker.get_events_between_timestamps(15.0, 25.0)
        assert events == [e2]
        
        events = tracker.get_events_between_timestamps(10.0, 30.0)
        assert events == [e1, e2, e3]

    def test_get_nearest_event_to(self, tracker):
        e1 = MockEvent(timestamp=10.0)
        e2 = MockEvent(timestamp=20.0)
        
        tracker.add_event(e1)
        tracker.add_event(e2)
        
        assert tracker.get_nearest_event_to(11.0) == e1
        assert tracker.get_nearest_event_to(19.0) == e2
        assert tracker.get_nearest_event_to(15.0) == e1 # tie-breaking depends on implementation, usually first found

    def test_get_event_where(self, tracker):
        e1 = MockEvent(timestamp=10.0)
        e2 = MockEvent(timestamp=20.0)
        
        tracker.add_event(e1)
        tracker.add_event(e2)
        
        found = tracker.get_event_where(lambda e: e.timestamp > 15.0)
        assert found == [e2]

    def test_object_tracker_factory(self, body):
        # Clear trackers for a clean test
        ObjectTrackerFactory._trackers = {}
        
        tracker1 = ObjectTrackerFactory.get_tracker(body)
        tracker2 = ObjectTrackerFactory.get_tracker(body)
        
        assert tracker1 is tracker2
        assert body in ObjectTrackerFactory._trackers.keys()
        assert tracker1 in ObjectTrackerFactory.get_all_trackers()

    def test_get_first_event_before_after(self, tracker):
        e1 = MockEvent(timestamp=10.0)
        e2 = MockEvent(timestamp=20.0)
        e3 = MockEvent(timestamp=30.0)
        
        tracker.add_event(e1)
        tracker.add_event(e2)
        tracker.add_event(e3)
        
        assert tracker.get_first_event_before(25.0) == e2
        assert tracker.get_first_event_after(15.0) == e2
        assert tracker.get_first_event_before(10.0) == None
        assert tracker.get_first_event_after(30.0) == None

    def test_get_nearest_event_with_tolerance(self, tracker):
        e1 = MockEvent(timestamp=10.0)
        tracker.add_event(e1)
        
        # tolerance is timedelta
        assert tracker.get_nearest_event_to(15.0, tolerance=timedelta(seconds=2)) == None
        assert tracker.get_nearest_event_to(15.0, tolerance=timedelta(seconds=6)) == e1

    def test_get_events_between_two_events(self, tracker):
        e1 = MockEvent(timestamp=10.0)
        e2 = MockEvent(timestamp=20.0)
        em = MockEvent(timestamp=25.0)
        e3 = MockEvent(timestamp=30.0)
        e4 = MockEvent(timestamp=40.0)
        
        tracker.add_event(e1)
        tracker.add_event(e2)
        tracker.add_event(em)
        tracker.add_event(e3)
        tracker.add_event(e4)
        
        # Excluding boundary events (e2 and e3)
        events = tracker.get_events_between_two_events(e2, e3)
        assert events == [em]

    def test_get_nearest_event_to_event_with_conditions(self, tracker):
        e1 = MockEvent(timestamp=10.0)
        e2 = MockEvent(timestamp=20.0)
        e3 = MockEvent(timestamp=30.0)
        
        tracker.add_event(e1)
        tracker.add_event(e2)
        tracker.add_event(e3)
        
        # Condition: timestamp > 25
        found = tracker.get_nearest_event_to_event_with_conditions(e2, lambda e: e.timestamp > 25.0)
        assert found == e3

    def test_get_events_sorted_by_nearest(self, tracker):
        e1 = MockEvent(timestamp=10.0)
        e2 = MockEvent(timestamp=20.0)
        e3 = MockEvent(timestamp=22.0)
        
        tracker.add_event(e1)
        tracker.add_event(e2)
        tracker.add_event(e3)
        
        sorted_events_with_diff = tracker.get_events_sorted_by_nearest_to_timestamp(21.0)
        # Should return [(e2, 1.0), (e3, 1.0), (e1, 11.0)]
        assert [e for e, dt in sorted_events_with_diff] == [e2, e3, e1]
        assert sorted_events_with_diff[0][1] == 1.0
