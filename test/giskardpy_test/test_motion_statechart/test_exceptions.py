from __future__ import annotations

from giskardpy.motion_statechart.exceptions import (
    LocalMinimumReachedError,
    MotionStatechartError,
)


class TestLocalMinimumReachedError:
    def test_belongs_to_the_motion_statechart_error_hierarchy(self) -> None:
        """
        Callers catching MotionStatechartError must also catch a local-minimum stop.
        """
        assert isinstance(LocalMinimumReachedError(), MotionStatechartError)

    def test_reports_message_and_correction(self) -> None:
        """
        The error explains what happened and suggests a way out.
        """
        error = LocalMinimumReachedError()

        assert error.error_message() == "Motion planning reached a local minimum."
        assert (
            error.suggest_correction()
            == "Try a different starting configuration or base placement."
        )
