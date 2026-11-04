"""
Color images and object hypotheses whose expected classification against a vocabulary of
color names is known.
"""

from __future__ import annotations

import numpy as np

from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis

# %% color vocabulary

COLOR_BY_WORD = {
    "blue": np.array([255, 0, 0], dtype=np.uint8),
    "green": np.array([0, 255, 0], dtype=np.uint8),
    "red": np.array([0, 0, 255], dtype=np.uint8),
}
"""
The color each vocabulary word names, in the channel order of the CAS color image.
"""

COLOR_VOCABULARY = list(COLOR_BY_WORD)
"""
A vocabulary in which every word names one color channel.
"""


def rgb(color: np.ndarray) -> np.ndarray:
    """
    :return: The given color of the CAS color image in RGB channel order.
    """
    return color[::-1]


# %% scenes


def solid_image(color: np.ndarray, height: int = 20, width: int = 40) -> np.ndarray:
    """
    :return: A color image filled with one color.
    """
    return np.full((height, width, 3), color, dtype=np.uint8)


def two_color_scene() -> np.ndarray:
    """
    :return: An image whose left half is red and whose right half is blue.
    """
    image = solid_image(COLOR_BY_WORD["red"])
    image[:, image.shape[1] // 2 :] = COLOR_BY_WORD["blue"]
    return image


def object_hypothesis(x: int, y: int, width: int, height: int) -> ObjectHypothesis:
    """
    :return: An object hypothesis covering the given image region.
    """
    hypothesis = ObjectHypothesis()
    hypothesis.roi.roi.pos.x = x
    hypothesis.roi.roi.pos.y = y
    hypothesis.roi.roi.width = width
    hypothesis.roi.roi.height = height
    return hypothesis


def hypothesis_classifications(
    hypothesis: ObjectHypothesis,
) -> list[Classification]:
    """
    :return: The classifications attached to the given object hypothesis.
    """
    return [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, Classification)
    ]
