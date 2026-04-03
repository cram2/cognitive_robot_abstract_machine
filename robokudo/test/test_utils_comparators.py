from __future__ import annotations

import numpy as np
import pytest
from typing_extensions import TYPE_CHECKING, Callable, Any

from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Classification,
    ColorHistogram,
    SemanticColor,
)
from robokudo.types.cv import Rect
from robokudo.utils.comparators import (
    TranslationComparator,
    BboxComparator,
    SizeComparator,
    HistogramComparator,
    SemanticColorComparator,
    AdditionalDataComparator,
    RoiComparator,
    MaskComparator,
    ClassificationComparator,
)

if TYPE_CHECKING:
    import numpy.typing as npt


class TestUtilsComparators(object):
    @pytest.mark.parametrize(
        ["query_value", "obj_value", "expected_similarity"],
        [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0),  # 100% the same
            (
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
            ),  # 0% the same (max_distance == distance)
            (
                (1.5, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
            ),  # 0% the same (distance > max_distance)
            (
                (0.5, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.5,
            ),  # 50% the same (max_distance / 2 == distance)
        ],
    )
    def test_translation_comparator(
        self,
        query_value: tuple[float, float, float],
        obj_value: tuple[float, float, float],
        expected_similarity: float,
    ):
        comparator = TranslationComparator(weight=1.0, max_distance=1.0)
        assert comparator.compute_similarity(query_value, obj_value) == pytest.approx(
            expected_similarity
        )

    @pytest.fixture
    def bounding_box_from_tuple(
        self,
    ) -> Callable[[tuple[float, float, float]], BoundingBox3DAnnotation]:
        def _make(lengths: tuple[float, float, float]):
            bbox = BoundingBox3DAnnotation()
            bbox.x_length, bbox.y_length, bbox.z_length = lengths
            return bbox

        return _make

    @pytest.mark.parametrize(
        ["query_value", "obj_value", "expected_similarity"],
        [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0),
            ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 1.0),
            ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0), 0.5),
            ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), 0.25),
        ],
    )
    def test_bbox_comparator(
        self,
        query_value: tuple[float, float, float],
        obj_value: tuple[float, float, float],
        expected_similarity: float,
        bounding_box_from_tuple: Callable[
            [tuple[float, float, float]], BoundingBox3DAnnotation
        ],
    ):
        query_bbox = bounding_box_from_tuple(query_value)
        obj_bbox = bounding_box_from_tuple(obj_value)
        comparator = BboxComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_bbox, obj_bbox)
        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.mark.parametrize(
        ["query_value", "obj_value", "expected_similarity"],
        [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0),
            ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1.0),
            ([1.0, 1.0, 1.0], [2.0, 1.0, 1.0], 0.5),
            ([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], 0.25),
        ],
    )
    def test_size_comparator(
        self,
        query_value: list[float],
        obj_value: list[float],
        expected_similarity: float,
    ):
        comparator = SizeComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_value, obj_value)
        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.fixture
    def classification_annotation_from_classname(
        self,
    ) -> Callable[[str], Classification]:
        def _make(classname: str) -> Classification:
            ann = Classification()
            ann.classname = classname
            return ann

        return _make

    @pytest.mark.parametrize(
        ["query_value", "obj_value", "expected_similarity"],
        [
            (np.array([0, 0, 0]), np.array([0, 0, 0]), pytest.approx(1.0)),
            (
                np.array([255, 255, 255]),
                np.array([0, 0, 0]),
                pytest.approx(0.0, abs=4e-2),
            ),
            (
                np.array([100, 100, 100]),
                np.array([100, 0, 0]),
                pytest.approx(0.444, abs=1e-3),
            ),
        ],
    )
    def test_histogram_comparator(
        self,
        query_value: np.ndarray,
        obj_value: np.ndarray,
        expected_similarity: pytest.approx,
    ):
        comparator = HistogramComparator(weight=1.0)

        query_hist, _ = np.histogram(query_value.ravel(), bins=256, range=[0, 256])
        obj_hist, _ = np.histogram(obj_value.ravel(), bins=256, range=[0, 256])

        query_color_hist = ColorHistogram()
        query_color_hist.hist = query_hist.astype(np.float32)

        obj_color_hist = ColorHistogram()
        obj_color_hist.hist = obj_hist.astype(np.float32)

        similarity = comparator.compute_similarity(query_color_hist, obj_color_hist)

        assert isinstance(similarity, float)
        assert similarity == expected_similarity

    @pytest.mark.parametrize(
        ["query_value", "obj_value", "expected_similarity"],
        [
            (("white", 1.0), ("black", 1.0), 0.0),  # Different colors, full ratio
            (("white", 1.0), ("black", 0.0), 0.0),  # Different colors, different ratio
            (("white", 0.0), ("white", 1.0), 0.0),  # Same colors, different ratio
            (("white", 1.0), ("white", 1.0), 1.0),  # Same colors, same ratio
            (("white", 0.5), ("white", 1.0), 0.5),  # Same colors, insecure ratio
        ],
    )
    def test_semantic_color_comparator(
        self,
        query_value: tuple[str, float],
        obj_value: tuple[str, float],
        expected_similarity: float,
    ):
        comparator = SemanticColorComparator(weight=1.0)

        query_color = SemanticColor()
        query_color.color = query_value[0]
        query_color.ratio = query_value[1]

        obj_color = SemanticColor()
        obj_color.color = obj_value[0]
        obj_color.ratio = obj_value[1]

        similarity = comparator.compute_similarity(query_color, obj_color)

        assert isinstance(similarity, float)
        assert similarity == expected_similarity

    @pytest.mark.parametrize(
        ["query_value", "obj_value", "expected_similarity"],
        [
            # Test floats
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.5),
            (1.0, 0.5, 2 / 3),
            # Test ints
            (1, 1, 1.0),
            (1, 0, 0.5),
            # Test strings
            ("test1", "test1", 1.0),
            ("test1", "test2", 0.0),
        ],
    )
    def test_additional_data_comparator(
        self, query_value: Any, obj_value: Any, expected_similarity: float
    ):
        comparator = AdditionalDataComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_value, obj_value)

        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.mark.parametrize(
        ["query_value", "obj_value", "expected_similarity"],
        [
            ((0, 0, 1, 1), (0, 0, 1, 1), 1.0),  # Full overlap
            ((1, 1, 1, 1), (0, 0, 1, 1), 0.0),  # No overlap
            ((0, 0, 1, 1), (0, 0, 2, 2), 0.25),  # Partial overlap
        ],
    )
    def test_roi_comparator(
        self,
        query_value: tuple[int, int, int, int],
        obj_value: tuple[int, int, int, int],
        expected_similarity: float,
    ):
        query_rect = Rect()
        query_rect.pos.x = query_value[0]
        query_rect.pos.y = query_value[1]
        query_rect.width = query_value[2]
        query_rect.height = query_value[3]

        obj_rect = Rect()
        obj_rect.pos.x = obj_value[0]
        obj_rect.pos.y = obj_value[1]
        obj_rect.width = obj_value[2]
        obj_rect.height = obj_value[3]

        comparator = RoiComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_rect, obj_rect)

        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.mark.parametrize(
        ["query_value", "obj_value", "expected_similarity"],
        [
            (
                np.array([[0.0, 0.0], [0.0, 0.0]]),
                np.array([[0.0, 0.0], [0.0, 0.0]]),
                0.0,
            ),
            (
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                1.0,
            ),
            (
                np.array([[1.0, 1.0], [0.0, 0.0]]),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                0.5,
            ),
            (
                np.array([[1.0, 1.0], [0.0, 0.0]]),
                np.array([[1.0, 0.0], [1.0, 0.0]]),
                1.0 / 3.0,
            ),
        ],
    )
    def test_mask_comparator(
        self,
        query_value: npt.NDArray,
        obj_value: npt.NDArray,
        expected_similarity: float,
    ):
        comparator = MaskComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_value, obj_value)

        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.mark.parametrize(
        ["query_value", "obj_value", "expected_similarity"],
        [
            (
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=1.0
                ),
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=1.0
                ),
                1.0,
            ),
            (
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=1.0
                ),
                Classification(
                    classification_type="CLASS", classname="chair", confidence=1.0
                ),
                0.0,
            ),
            (
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=1.0
                ),
                Classification(
                    classification_type="INSTANCE", classname="table", confidence=1.0
                ),
                0.0,
            ),
            (
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=0.75
                ),
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=0.95
                ),
                0.8,
            ),
            (
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=0.0
                ),
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=1.0
                ),
                0.0,
            ),
        ],
    )
    def test_classification_comparator(
        self, query_value, obj_value, expected_similarity
    ):
        comparator = ClassificationComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_value, obj_value)

        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)
