"""OpenCV-based visualization for RoboKudo pipelines.

This module provides OpenCV-based visualization capabilities for RoboKudo pipelines.
It handles:

* 2D image visualization with annotator overlays
* Mouse interaction handling
* Keyboard controls for navigation and control
* Window management
* Pipeline state visualization
"""

import subprocess
import sys

import cv2
import numpy as np
from py_trees.blackboard import Blackboard
from typing_extensions import Any, Optional

from robokudo.annotators.core import BaseAnnotator
from robokudo.vis.visualizer import Visualizer


class CVVisualizer(Visualizer, Visualizer.Observer):
    """OpenCV-based visualizer for 2D image data.

    This class provides visualization of 2D image data from pipeline annotators using
    OpenCV windows. It supports:

    * Image display with annotator overlays
    * Mouse interaction handling
    * Keyboard navigation between annotators
    * Window management
    * Shared visualization state

    .. note::
        This Visualizer works with a shared state and needs notifications
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the OpenCV visualizer."""
        super().__init__(*args, **kwargs)
        self.shared_visualizer_state.register_observer(self)

    def tick(self) -> None:
        """Update the visualization display.

        This method:

        * Gets current annotator outputs
        * Updates display if needed
        * Renders annotator name overlay
        * Manages OpenCV window
        """
        # print("Tick method called")
        annotator_outputs = self.get_visualized_annotator_outputs_for_pipeline()
        # print(f"outputs are {annotator_outputs}")

        assert self.shared_visualizer_state.active_annotator is not None
        active_annotator_instance: BaseAnnotator = (
            self.shared_visualizer_state.active_annotator
        )
        # print(f"Active annotator: {active_annotator_instance.name}")

        self.update_output_flag_for_new_data()

        # # Print all annotators in shared visualizer state
        # print("All annotators in shared visualizer state:")
        # for annotator in annotator_outputs.outputs.keys():
        #     print(f"- {annotator}")

        # Check if we have to render something
        if self.update_output:
            # print("Updating output")
            self.update_output = False
            # 2D imshow output

            img = None
            if active_annotator_instance.name not in annotator_outputs.outputs:
                # We do not yet have visual output set up for this annotator
                # This might happen in dynamic perception pipelines, where annotators have not been set up
                # during construction of the tree AND don't generate image outputs.
                # Create an empty image to show in the visualizer
                # print(f"No visual output for annotator {active_annotator_instance.name}")
                img = np.zeros((640, 480, 3), dtype="uint8")
            else:
                img = annotator_outputs.outputs[active_annotator_instance.name].image

            img_with_annotator_text = cv2.putText(
                img,
                active_annotator_instance.name,
                (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            # print(f"Displaying image for {active_annotator_instance.name}")

            window_title = self.window_title()
            # print(f"Window title: {window_title}")
            cv2.namedWindow(window_title)
            cv2.setMouseCallback(window_title, self.mouse_callback_cv)
            cv2.imshow(window_title, img_with_annotator_text)

    def mouse_callback_cv(
        self, event: int, x: int, y: int, flags: int, param: Any
    ) -> None:
        """
        Mouse callback for the 2D Visualizer.
        Prints double click events and forwards every event to the active annotator.

        :param event: OpenCV mouse event type
        :param x: X coordinate of mouse event
        :param y: Y coordinate of mouse event
        :param flags: OpenCV event flags
        :param param: Additional parameters (unused)
        """
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print(f"2D Visualizer double-clicked at ({x},{y})")

        vis_state: Visualizer.SharedState = self.shared_visualizer_state
        active_annotator_instance: BaseAnnotator = vis_state.active_annotator
        active_annotator_instance.mouse_callback(event, x, y, flags, param)

    def window_title(self) -> str:
        """Get the window title for this visualizer."""
        return self.identifier()

    def notify(
        self,
        observable: Visualizer.Observable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle notification of state changes.

        :param observable: The object that sent the notification
        """
        self.update_output = True

    @staticmethod
    def static_post_tick() -> None:
        """Handle keyboard input after visualization update.

        This method:

        * Checks for keyboard input
        * Routes input to appropriate visualizer
        * Handles termination requests
        """
        # print("static_post_tick called")
        key = cv2.waitKey(1)
        # print(f"Key code received: {key}")

        if key == -1:  # no key pressed. return directly
            return

        cv_visualizer_for_key = CVVisualizer.get_gui_handler_for_detected_key()
        if not cv_visualizer_for_key:
            print("Error: Couldn't map input key from CV window to a Visualizer class.")
            return

        # Key Callback returns false when execution should be stopped.
        if not cv_visualizer_for_key.handle_keycallback(key):
            cv_visualizer_for_key.indicate_termination_var = True

    @staticmethod
    def get_gui_handler_for_detected_key() -> Optional["CVVisualizer"]:
        """Get the visualizer instance for the focused window.

        :returns: The visualizer instance for the focused window, or None if not found
        """
        # Place a system call to get the title of the window that is currently focussed.
        get_imshow_title = subprocess.run(
            "xprop -id $(xprop -root _NET_ACTIVE_WINDOW | cut -d ' ' -f 5) WM_NAME | awk -F '\"' '{print $2}'",
            capture_output=True,
            shell=True,
        )
        if get_imshow_title.returncode != 0:
            print(
                f"GUI Handling can't call system method to get window title: {get_imshow_title.stderr.decode()}"
            )
            sys.exit(1)

        imshow_title = get_imshow_title.stdout.strip().decode("utf-8")
        pipeline_name_of_focussed_gui = imshow_title.strip().replace("RoboKudo/", "")

        cv_visualizers = [
            cv for cv in Visualizer.instances if isinstance(cv, CVVisualizer)
        ]

        for cv_visualizer in cv_visualizers:
            if cv_visualizer.pipeline.name == pipeline_name_of_focussed_gui:
                return cv_visualizer

        return None

    def handle_keycallback(self, key: int) -> bool:
        """
        Handle a key-press that happened in the corresponding GUI of this GUIHandler.

        :param key: An ASCII char
        :return: false if GUI reports abort (right now this only happens when ESC is pressed)
        """
        # print(f"Key pressed: {key}")
        vis_state: Visualizer.SharedState = self.shared_visualizer_state
        active_annotator_instance: BaseAnnotator = vis_state.active_annotator
        annotator_list = self.pipeline.get_annotators()

        if key == 27:  # ESC
            return False

        if key == 81 or key == 112 or key == 106:
            vis_state.active_annotator_i = (
                len(annotator_list) - 1
                if vis_state.active_annotator_i == 0
                else vis_state.active_annotator_i - 1
            )

            # If the available annotators are changing dynamically, we have to ensure
            # that we still point to a valid annotator.
            # Naive approach: Just use the last one in the list
            if vis_state.active_annotator_i < len(annotator_list):
                vis_state.active_annotator = annotator_list[
                    vis_state.active_annotator_i
                ]
            else:
                vis_state.active_annotator_i = len(annotator_list) - 1
            self.shared_visualizer_state.notify_observers()
            # print(f"Left arrow key pressed. Active annotator: {vis_state.active_annotator.name}")
            return True

        if key == 83 or key == 110 or key == 107:  # right arrow
            vis_state.active_annotator_i = (
                0
                if vis_state.active_annotator_i == len(annotator_list) - 1
                else vis_state.active_annotator_i + 1
            )
            vis_state.active_annotator = annotator_list[vis_state.active_annotator_i]
            self.shared_visualizer_state.notify_observers()
            # print(f"Right arrow key pressed. Active annotator: {vis_state.active_annotator.name}")
            return True

        if key == 32:  # space
            blackboard = Blackboard()
            blackboard.set("pipeline_trigger", True)

        # all other keys can be handled by the annotator
        active_annotator_instance.key_callback(key)
        return True
