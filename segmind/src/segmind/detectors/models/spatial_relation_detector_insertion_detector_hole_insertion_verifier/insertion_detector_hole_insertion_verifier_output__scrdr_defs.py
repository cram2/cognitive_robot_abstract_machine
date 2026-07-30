import itertools
import time

from random_events.product_algebra import SimpleEvent
from ripple_down_rules.datastructures.case import Case

from pycram.datastructures.dataclasses import AxisAlignedBoundingBox
from ....datastructures.events import InsertionEvent, InterferenceEvent, PickUpEvent
from typing_extensions import Dict, Optional, Union
from types import NoneType
from ....detectors.spatial_relation_detector import InsertionDetector
from pycram.datastructures.world_entity import PhysicalBody

from ....utils import is_object_supported_by_container_body


def conditions_38355037295796650033371896063976531277(case) -> bool:
    def conditions_for_insertion_detector_hole_insertion_verifier(self_: InsertionDetector, hole: PhysicalBody, event: InterferenceEvent, output_: bool) -> bool:
        """Get conditions on whether it's possible to conclude a value for InsertionDetector_hole_insertion_verifier.output_  of type ."""
        hole_bbox = hole.get_axis_aligned_bounding_box()
        hole_max_z = hole_bbox.max_z
        obj_bbox = event.tracked_object.get_axis_aligned_bounding_box()
        obj_max = obj_bbox.max_z
        if obj_max <= hole_max_z + 1e-3:
            return True
        else:
            return False
    return conditions_for_insertion_detector_hole_insertion_verifier(**case)


def conditions_21738774625860220488991060484462427733(case) -> bool:
    def conditions_for_insertion_detector_hole_insertion_verifier(self_: InsertionDetector, hole: PhysicalBody, event: InterferenceEvent, output_: bool) -> bool:
        """Get conditions on whether it's possible to conclude a value for InsertionDetector_hole_insertion_verifier.output_  of type ."""
        return True
    return conditions_for_insertion_detector_hole_insertion_verifier(**case)


def conditions_313966059252436144481394373657043070884(case) -> bool:
    def conditions_for_insertion_detector_hole_insertion_verifier(self_: InsertionDetector, hole: PhysicalBody, event: InterferenceEvent, output_: bool) -> bool:
        """Get conditions on whether it's possible to conclude a value for InsertionDetector_hole_insertion_verifier.output_  of type ."""
        latest_insertion = event.object_tracker.get_latest_event_of_type(InsertionEvent)
        if latest_insertion is None:
            return False
        latest_pickup = event.object_tracker.get_first_event_of_type_after_event(PickUpEvent, latest_insertion)
        no_pick_up = latest_pickup is None
        latest_insertion_before_this_insertion = event.object_tracker.get_first_event_of_type_before_event(InsertionEvent, latest_insertion)
        if latest_insertion_before_this_insertion is None:
            return no_pick_up
        elif abs(latest_insertion.timestamp - latest_insertion_before_this_insertion.timestamp) < 0.5:
            return True
        else:
            return no_pick_up
    return conditions_for_insertion_detector_hole_insertion_verifier(**case)


def conclusion_313966059252436144481394373657043070884(case) -> bool:
    def insertion_detector_hole_insertion_verifier(self_: InsertionDetector, hole: PhysicalBody, event: InterferenceEvent, output_: bool) -> bool:
        """Get possible value(s) for InsertionDetector_hole_insertion_verifier.output_  of type ."""
        return False
    return insertion_detector_hole_insertion_verifier(**case)


def conclusion_21738774625860220488991060484462427733(case) -> bool:
    def insertion_detector_hole_insertion_verifier(self_: InsertionDetector, hole: PhysicalBody, event: InterferenceEvent, output_: bool) -> bool:
        """Get possible value(s) for InsertionDetector_hole_insertion_verifier.output_  of type ."""
        return True
    return insertion_detector_hole_insertion_verifier(**case)


def conclusion_38355037295796650033371896063976531277(case) -> bool:
    def insertion_detector_hole_insertion_verifier(self_: InsertionDetector, hole: PhysicalBody, event: InterferenceEvent, output_: bool) -> bool:
        """Get possible value(s) for InsertionDetector_hole_insertion_verifier.output_  of type ."""
        return True
    return insertion_detector_hole_insertion_verifier(**case)


