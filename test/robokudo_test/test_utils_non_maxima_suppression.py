import pytest

from robokudo.utils.non_maxima_suppression import non_max_suppression_, class_based_nms


class TestUtilsNonMaximaSuppression(object):
    def test_non_max_suppression(self):
        predictions = [
            ((10, 10, 20, 20), 0.9, "person"),
            ((12, 12, 22, 22), 0.8, "person"),
        ]
        filtered = non_max_suppression_(predictions)

        assert len(filtered) == 1
        assert filtered[0] == predictions[0]

    def test_non_max_suppression_empty_predictions(self):
        predictions = [
            ((10, 10, 20, 20), 0.2, "person"),
            ((12, 12, 22, 22), 0.25, "person"),
        ]
        filtered = non_max_suppression_(predictions, confidence_treshold=0.4)
        assert len(filtered) == 0

    def test_non_max_suppression_class_suppression(self):
        predictions = [
            ((10, 10, 20, 20), 0.9, "jacket"),
            ((12, 12, 22, 22), 0.8, "person"),
        ]
        filtered = non_max_suppression_(predictions)

        assert len(filtered) == 1
        assert filtered[0] == predictions[0]

    @pytest.mark.parametrize(
        ["confidence_threshold", "filtered_count"], [(0.5, 1), (0.3, 2)]
    )
    def test_non_max_suppression_confidence_threshold(
        self, confidence_threshold: float, filtered_count: int
    ):
        predictions = [
            ((10, 10, 20, 20), 0.4, "person"),
            ((15, 15, 25, 25), 0.8, "person"),
        ]
        filtered = non_max_suppression_(
            predictions, confidence_treshold=confidence_threshold
        )
        assert len(filtered) == filtered_count

    @pytest.mark.parametrize(
        ["iou_threshold", "filtered_count"],
        [
            (0.1, 1),
            (0.4, 2),
        ],
    )
    def test_non_max_suppression_iou_threshold(
        self, iou_threshold: float, filtered_count: int
    ):
        predictions = [
            ((10, 10, 20, 20), 0.9, "person"),
            ((15, 15, 25, 25), 0.8, "person"),
        ]
        filtered = non_max_suppression_(predictions, iou_threshold=iou_threshold)
        assert len(filtered) == filtered_count

    def test_class_based_nms(self):
        predictions = [
            ((10, 10, 20, 20), 0.9, "person"),
            ((12, 12, 22, 22), 0.8, "person"),
        ]
        filtered = class_based_nms(predictions)

        assert len(filtered) == 1
        assert filtered[0] == predictions[0]

    def test_class_based_nms_empty_predictions(self):
        predictions = [
            ((10, 10, 20, 20), 0.2, "person"),
            ((12, 12, 22, 22), 0.25, "person"),
        ]
        filtered = class_based_nms(predictions, confidence_threshold=0.4)
        assert len(filtered) == 0

    def test_class_based_nms_class_suppression(self):
        predictions = [
            ((10, 10, 20, 20), 0.9, "jacket"),
            ((12, 12, 22, 22), 0.8, "jacket"),
            ((12, 12, 22, 22), 0.8, "person"),
        ]
        filtered = class_based_nms(predictions)

        assert len(filtered) == 2
        assert filtered.index(predictions[0]) >= 0
        assert filtered.index(predictions[2]) >= 0

    @pytest.mark.parametrize(
        ["confidence_threshold", "filtered_count"], [(0.5, 1), (0.3, 2)]
    )
    def test_class_based_nms_confidence_threshold(
        self, confidence_threshold: float, filtered_count: int
    ):
        predictions = [
            ((10, 10, 20, 20), 0.4, "person"),
            ((15, 15, 25, 25), 0.8, "person"),
        ]
        filtered = class_based_nms(
            predictions, confidence_threshold=confidence_threshold
        )
        assert len(filtered) == filtered_count

    @pytest.mark.parametrize(
        ["iou_threshold", "filtered_count"],
        [
            (0.1, 1),
            (0.4, 2),
        ],
    )
    def test_class_based_nms_iou_threshold(
        self, iou_threshold: float, filtered_count: int
    ):
        predictions = [
            ((10, 10, 20, 20), 0.9, "person"),
            ((15, 15, 25, 25), 0.8, "person"),
        ]
        filtered = class_based_nms(predictions, iou_threshold=iou_threshold)
        assert len(filtered) == filtered_count
