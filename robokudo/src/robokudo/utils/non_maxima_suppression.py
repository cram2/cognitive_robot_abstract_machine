"""Non-Maxima-Suppression (NMS) implementation for object detection.

This module provides NMS implementations for filtering overlapping bounding boxes.

:module: non_maxima_suppression
:synopsis: Non-Maxima-Suppression for object detection
:moduleauthor: Lennart Heinbokel

:Created:
    2023-02-02
"""

from typing_extensions import Tuple, List

# pylint: disable=invalid-name


def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate the intersection over union (IoU) of two bounding boxes.

    :param box1: First bounding box coordinates (x1, y1, x2, y2) where:
        * x1, y1: coordinates of the top-left corner
        * x2, y2: coordinates of the bottom-right corner
    :param box2: Second bounding box coordinates in the same format as box1
    :return: The intersection over union value [0,1]

    :Example:

    .. code-block:: python

        box1 = (10, 10, 20, 20)
        box2 = (15, 15, 25, 25)
        iou = _iou(box1, box2)  # Returns overlap ratio
    """
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / float(box1Area + box2Area - interArea)

    return iou


def non_max_suppression_(
    predictions: List[Tuple[Tuple[int, int, int, int], float, str]],
    confidence_treshold: float = 0.5,
    iou_threshold: float = 0.4,
) -> List[Tuple[Tuple[int, int, int, int], float, str]]:
    """Perform non-maxima suppression (NMS) on object detection results.

    :param predictions: List of tuples, where each tuple contains:
        * bounding box coordinates (x1, y1, x2, y2)
        * confidence score
        * class label
    :param confidence_treshold: Minimum confidence score for a box to be considered
    :param iou_threshold: IoU threshold above which boxes are considered to overlap
    :return: Filtered list of predictions after applying NMS

    :Example:

    .. code-block:: python

        predictions = [
            ((10, 10, 20, 20), 0.9, "person"),
            ((15, 15, 25, 25), 0.8, "person")
        ]
        filtered = non_max_suppression_(predictions)
    """
    predictions = [
        prediction for prediction in predictions if prediction[1] > confidence_treshold
    ]

    if not predictions:
        return []

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    output = [predictions[0]]

    for prediction in predictions[1:]:
        for candidate in output:
            if _iou(candidate[0], prediction[0]) > iou_threshold:
                break
        else:
            output.append(prediction)

    return output


def class_based_nms(
    predictions: List[Tuple[Tuple[int, int, int, int], float, str]],
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.4,
) -> List[Tuple[Tuple[int, int, int, int], float, str]]:
    """Perform class-based non-maximum suppression (NMS) on object detection results.

    :param predictions: List of tuples, where each tuple contains:
        * bounding box coordinates (x1, y1, x2, y2)
        * confidence score
        * class label
    :param confidence_threshold: Minimum confidence score for a box to be considered
    :param iou_threshold: IoU threshold above which boxes are considered to overlap
    :return: Filtered list of predictions after applying class-based NMS

    :Example:

    .. code-block:: python

        predictions = [
            ((10, 10, 20, 20), 0.9, "person"),
            ((15, 15, 25, 25), 0.8, "car")
        ]
        filtered = class_based_nms(predictions)

    .. note::
        This function performs NMS independently for each class, which prevents
        boxes of different classes from suppressing each other.
    """
    predictions = [
        prediction for prediction in predictions if prediction[1] > confidence_threshold
    ]

    if not predictions:
        return []

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    output = []

    # group predictions by class label
    class_labels = set(prediction[2] for prediction in predictions)
    for label in class_labels:
        class_predictions = [
            prediction for prediction in predictions if prediction[2] == label
        ]

        # perform non-maximum suppression for each class separately
        while class_predictions:
            class_predictions = sorted(
                class_predictions, key=lambda x: x[1], reverse=True
            )
            candidate = class_predictions[0]
            output.append(candidate)
            class_predictions = class_predictions[1:]

            for prediction in class_predictions:
                if _iou(candidate[0], prediction[0]) > iou_threshold:
                    class_predictions.remove(prediction)

    return output
