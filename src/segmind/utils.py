from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod

import numpy as np
from pycram.designator import ObjectDesignatorDescription
from segmind import set_logger_level, LogLevel, logger
from semantic_digital_twin.reasoning.predicates import is_supported_by, contact
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox, Color
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Optional, Tuple

from pycram.datastructures.enums import Arms, Grasp, AxisIdentifier
from pycram.datastructures.grasp import GraspDescription


from pycram.tf_transformations import quaternion_inverse, quaternion_multiply

set_logger_level(LogLevel.DEBUG)
try:
    from semantic_world.views import Container
except ImportError:
    Container = None
    print(
        "Container view is not available. Some functionalities may not work as expected."
    )

from gtts import gTTS
import pygame

speech_lock = threading.RLock()


def text_to_speech(text: str):
    # The text that you want to convert to audio
    text = "Hello" if text is None else text

    # Language in which you want to convert
    language = "en"

    # Passing the text and language to the engine,
    # here we have marked slow=False. Which tells
    # the module that the converted audio should
    # have a high speed
    myobj = gTTS(text=text, lang=language, slow=False)

    with speech_lock:
        myobj.save("welcome.mp3")

        # Initialize the mixer module
        try:
            pygame.mixer.init()
            try:
                pygame.mixer.music.load("welcome.mp3")
            except pygame.error:
                pass

            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                continue
        except pygame.error:
            print("Audio not available, running in silent mode.")


class PropagatingThread(threading.Thread, ABC):
    exc: Optional[Exception] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kill_event = threading.Event()

    def run(self):
        self.exc = None
        self._run()

    @abstractmethod
    def _run(self):
        pass

    def stop(self):
        """
        Stop the event detector.
        """
        self.kill_event.set()
        self._join()

    @abstractmethod
    def _join(self, timeout=None):
        pass


class Imaginator:
    """
    A class that provides methods for imagining objects.
    """

    surfaces_created: List[Body] = []
    latest_surface_idx: int = 0

    @classmethod
    def imagine_support_from_aabb(cls, aabb: BoundingBox) -> Body:
        """
        Imagine a support with the size of the axis-aligned bounding box.

        :param aabb: The axis-aligned bounding box for which the support of same size should be imagined.
        :return: The support object.
        """
        return cls._imagine_support(aabb=aabb)

    @classmethod
    def imagine_support_for_object(
        cls, obj: Body, support_thickness: Optional[float] = 0.005
    ) -> Body:
        """
        Imagine a support that supports the object and has a specified thickness.

        :param obj: The object for which the support should be imagined.
        :param support_thickness: The thickness of the support.
        :return: The support object
        """
        return cls._imagine_support(obj=obj, support_thickness=support_thickness)

    @classmethod
    def _imagine_support(
        cls,
        obj: Optional[Body] = None,
        aabb: Optional[BoundingBox] = None,
        support_thickness: Optional[float] = None,
    ) -> Body:
        """
        Imagine a support for the object or with the size of the axis-aligned bounding box.

        :param obj: The object for which the support should be imagined.
        :param aabb: The axis-aligned bounding box for which the support of same size should be imagined.
        :param support_thickness: The thickness of the support.
        :return: The support object.
        """
        if aabb is not None:
            obj_aabb = aabb
        elif obj is not None:
            obj_aabb = obj.get_axis_aligned_bounding_box()
        else:
            raise ValueError(
                "Either object or axis-aligned bounding box should be provided."
            )
        print(f"support index: {cls.latest_surface_idx}")
        support_name = f"imagined_support_{cls.latest_surface_idx}"
        support_thickness = (
            obj_aabb.depth if support_thickness is None else support_thickness
        )
        box_vis_shape = BoundingBox(
            Color(1, 1, 0, 1),
            Vector3(0, 0, 0),
            Vector3(obj_aabb.width, obj_aabb.depth, support_thickness * 0.5),
        )
        support = ObjectDesignatorDescription(list(support_name))
        support_obj = Body(support_name, Supporter, None, support, color=support.color)
        support_position = obj_aabb.base_origin
        support_obj.set_position(support_position)

        from .detectors.atomic_event_detectors import AbstractContactDetector

        contact_points, _ = AbstractContactDetector.get_contact_points_for_body(
            support_obj, 0.05
        )
        contacted_objects = AbstractContactDetector.get_bodies_in_contact(
            contact_points
        )
        contacted_surfaces = [
            obj
            for obj in contacted_objects
            if obj in cls.surfaces_created and obj != support_obj
        ]
        for obj in contacted_surfaces:
            support_obj = support_obj.merge(obj)
            cls.surfaces_created.remove(obj)
        World.current_world.get_object_by_type(Floor)[0].attach(support_obj)
        cls.surfaces_created.append(support_obj)
        cls.latest_surface_idx += 1
        return support_obj


def get_angle_between_vectors(vector_1: List[float], vector_2: List[float]) -> float:
    """
    Get the angle between two vectors.

    :param vector_1: A list of float values that represent the first vector.
    :param vector_2: A list of float values that represent the second vector.
    :return: A float value that represents the angle between the two vectors.
    """
    angle = np.arccos(
        np.dot(vector_1, vector_2)
        / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    )
    if isinstance(angle, np.ndarray):
        angle = float(angle.squeeze())
    return angle


def calculate_transform_difference_and_check_if_small(
    transform_1: Transform,
    transform_2: Transform,
    translation_threshold: float,
    angle_threshold: float,
) -> bool:
    """
    Calculate the translation and rotation of the object with respect to the hand to check if it was picked up,
     uses the translation and rotation thresholds to determine if the object was picked up.

    :param transform_1: The transform of the object at the first time step.
    :param transform_2: The transform of the object at the second time step.
    :param translation_threshold: The threshold for the translation difference to be considered as small.
    :param angle_threshold: The threshold for the angle between the two quaternions to be considered as small.
    :return: A tuple of two boolean values that represent the conditions for the translation and rotation of the
    object to be considered as picked up.
    """
    trans_1, quat_1 = transform_1.translation_as_list(), transform_1.rotation_as_list()
    trans_2, quat_2 = transform_2.translation_as_list(), transform_2.rotation_as_list()
    trans_diff_cond = calculate_translation_difference_and_check(
        trans_1, trans_2, translation_threshold
    )
    rot_diff_cond = calculate_angle_between_quaternions_and_check(
        quat_1, quat_2, angle_threshold
    )
    return trans_diff_cond and rot_diff_cond


def calculate_translation_difference_and_check(
    trans_1: List[float], trans_2: List[float], threshold: float
) -> bool:
    """
    Calculate the translation difference and checks if it is small.

    :param trans_1: The translation of the object at the first time step.
    :param trans_2: The translation of the object at the second time step.
    :param threshold: The threshold for the translation difference to be considered as small.
    :return: A boolean value that represents the condition for the translation of the object to be considered as
    picked up.
    """
    translation_diff = calculate_abs_translation_difference(trans_1, trans_2)
    return is_translation_difference_small(translation_diff, threshold)


def is_translation_difference_small(trans_diff: List[float], threshold: float) -> bool:
    """
    Check if the translation difference is small by comparing it to the translation threshold.

    :param trans_diff: The translation difference.
    :param threshold: The threshold for the translation difference to be considered as small.
    :return: A boolean value that represents the condition for the translation difference to be considered as small.
    """
    return np.linalg.norm(trans_diff) <= threshold



def calculate_translation(position_1: List[float], position_2: List[float]) -> List:
    """
    calculate the translation between two positions.

    :param position_1: The first position.
    :param position_2: The second position.
    :return: A list of float values that represent the translation between the two positions.
    """
    return [p2 - p1 for p1, p2 in zip(position_1, position_2)]


def calculate_abs_translation_difference(
    trans_1: List[float], trans_2: List[float]
) -> List[float]:
    """
    Calculate the translation difference.

    :param trans_1: The translation of the object at the first time step.
    :param trans_2: The translation of the object at the second time step.
    :return: A list of float values that represent the translation difference.
    """
    return [abs(t1 - t2) for t1, t2 in zip(trans_1, trans_2)]


def calculate_euclidean_distance(point_1: List[float], point_2: List[float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    :param point_1: The first point.
    :param point_2: The second point.
    :return: A float value that represents the Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(point_1) - np.array(point_2))


def calculate_translation_vector(point_1: List[float], point_2: List[float]):
    """
    Calculate the translation vector between two points.

    :param point_1: The first point.
    :param point_2: The second point.
    :return: A list of float values that represent the translation vector between the two points.
    """
    return [p2 - p1 for p1, p2 in zip(point_1, point_2)]


def calculate_angle_between_quaternions_and_check(
    quat_1: List[float], quat_2: List[float], threshold: float
) -> bool:
    """
    Calculate the angle between two quaternions and checks if it is small.

    :param quat_1: The first quaternion.
    :param quat_2: The second quaternion.
    :param threshold: The threshold for the angle between the two quaternions to be considered as small.
    :return: A boolean value that represents the condition for the angle between the two quaternions
     to be considered as small.
    """
    quat_diff_angle = calculate_angle_between_quaternions(quat_1, quat_2)
    return quat_diff_angle <= threshold


def calculate_angle_between_quaternions(
    quat_1: List[float], quat_2: List[float]
) -> float:
    """
    Calculate the angle between two quaternions.

    :param quat_1: The first quaternion.
    :param quat_2: The second quaternion.
    :return: A float value that represents the angle between the two quaternions.
    """
    quat_diff = calculate_quaternion_difference(quat_1, quat_2)
    quat_diff_angle = 2 * np.arctan2(np.linalg.norm(quat_diff[0:3]), quat_diff[3])
    return quat_diff_angle


def calculate_quaternion_difference(
    quat_1: List[float], quat_2: List[float]
) -> List[float]:
    """
    Calculate the quaternion difference.

    :param quat_1: The quaternion of the object at the first time step.
    :param quat_2: The quaternion of the object at the second time step.
    :return: A list of float values that represent the quaternion difference.
    """
    quat_diff = quaternion_multiply(quaternion_inverse(quat_1), quat_2)
    return quat_diff
