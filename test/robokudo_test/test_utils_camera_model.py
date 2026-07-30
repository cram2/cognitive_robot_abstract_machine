import pytest

from robokudo.utils.camera_model import (
    pinhole_camera_parameters_from_horizontal_field_of_view,
)


class TestUtilsCameraModel:
    def test_pinhole_camera_parameters_from_horizontal_field_of_view(self) -> None:
        parameters = pinhole_camera_parameters_from_horizontal_field_of_view(
            width=640, height=480, horizontal_field_of_view_degrees=90.0
        )

        assert parameters.width == 640
        assert parameters.height == 480
        assert parameters.focal_length_x == pytest.approx(320.0)
        assert parameters.focal_length_y == pytest.approx(320.0)
        assert parameters.center_x == pytest.approx(319.5)
        assert parameters.center_y == pytest.approx(239.5)

    @pytest.mark.parametrize(
        ("width", "height", "horizontal_field_of_view_degrees"),
        [
            (0, 480, 90.0),
            (640, 0, 90.0),
            (640, 480, 0.0),
            (640, 480, 180.0),
        ],
    )
    def test_pinhole_camera_parameters_from_horizontal_field_of_view_rejects_invalid_values(
        self, width: int, height: int, horizontal_field_of_view_degrees: float
    ) -> None:
        with pytest.raises(ValueError):
            pinhole_camera_parameters_from_horizontal_field_of_view(
                width=width,
                height=height,
                horizontal_field_of_view_degrees=horizontal_field_of_view_degrees,
            )
