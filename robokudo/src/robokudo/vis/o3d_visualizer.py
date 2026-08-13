"""Open3D-based visualization for RoboKudo pipelines.

This module provides 3D visualization capabilities for RoboKudo pipelines using Open3D.
It handles:

* 3D geometry visualization
* Point cloud rendering
* Camera control
* Coordinate frame display
* Window management
"""

import logging

import open3d as o3d  # this import creates a SIGINT during unit test execution....
from robokudo.annotators.core import BaseAnnotator
from robokudo.defs import PACKAGE_NAME
from robokudo.vis.visualizer import Visualizer
from typing_extensions import Any, List, Dict, Union, Optional


class O3DVisualizer(Visualizer, Visualizer.Observer):
    """Open3D-based visualizer for 3D geometry data.

    This class provides visualization of 3D geometry data from pipeline annotators using
    Open3D windows. It supports:

    * 3D geometry visualization
    * Point cloud rendering
    * Camera control
    * Coordinate frame display
    * Shared visualization state

    .. note::
        This Visualizer works with a shared state and needs notifications
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Open3D visualizer."""
        super().__init__(*args, **kwargs)

        self.viewer3d: Optional[Viewer3D] = None
        """Open3D viewer instance"""

        self.shared_visualizer_state.register_observer(self)

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

    def tick(self) -> None:
        """Update the visualization display.

        This method:

        * Initializes viewer if needed
        * Gets current annotator outputs
        * Updates display if needed
        * Handles viewer lifecycle

        :returns: False if visualization should terminate, True otherwise
        """
        if self.viewer3d is None:
            self.viewer3d = Viewer3D(self.window_title() + "_3D")

        annotator_outputs = self.get_visualized_annotator_outputs_for_pipeline()

        active_annotator_instance: BaseAnnotator = (
            self.shared_visualizer_state.active_annotator
        )

        self.update_output_flag_for_new_data()

        if self.update_output:
            self.update_output = False

            geometries = None
            # We might not yet have visual output set up for this annotator
            # This might happen in dynamic perception pipelines, where annotators have not been set up
            # during construction of the tree AND don't generate cloud outputs.
            # => Fetch geometry if present
            if active_annotator_instance.name in annotator_outputs.outputs:
                geometries = annotator_outputs.outputs[
                    active_annotator_instance.name
                ].geometries

            self.viewer3d.update_cloud(geometries)

        tick_result = (
            self.viewer3d.tick()
        )  # right now, this is the last update call. if that's true, the GUI is happy.

        if not tick_result:
            self.indicate_termination_var = True

    def window_title(self) -> str:
        """Get the window title for this visualizer."""
        return self.identifier()


class Viewer3D(object):
    """Open3D viewer wrapper for 3D visualization.

    This class wraps the Open3D visualization functionality to provide:

    * Window management
    * Geometry updates
    * Camera control
    * Coordinate frame display
    """

    def __init__(self, title: str) -> None:
        """Initialize the 3D viewer.

        :param title: Window title for the viewer
        """
        self.first_cloud: bool = True
        """Flag indicating if this is the first cloud being displayed"""

        self.CLOUD_NAME: str = "Viewer3D"
        """Name identifier for the point cloud"""

        self.rk_logger: logging.Logger = logging.getLogger(PACKAGE_NAME)
        """Logger instance"""

        app: o3d.visualization.gui.Application = (
            o3d.visualization.gui.Application.instance
        )
        """Open3D app instance"""

        app.initialize()

        self.rk_logger.info(
            "Starting O3DVisualizer. 3D output might be broken if no success message follows. "
            "Try rebooting the machine if running local or check GPU passthrough in Docker."
        )

        self.main_vis: o3d.visualization.O3DVisualizer = (
            o3d.visualization.O3DVisualizer(title)
        )
        """Open3D visualizer instance"""

        self.rk_logger.info("Starting O3DVisualizer was successful")

        app.add_window(self.main_vis)

        self.visualized_geometries: List[str] = []
        """List of the names of the currently visualized geometries"""

    def tick(self) -> Any:
        """Update the viewer display.

        :returns: False if visualization should terminate, True otherwise
        """
        app = o3d.visualization.gui.Application.instance
        tick_return = app.run_one_tick()
        if tick_return:
            self.main_vis.post_redraw()
        return tick_return

    def update_cloud(
        self, geometries: Optional[Union[o3d.geometry.Geometry, Dict, List]]
    ) -> None:
        """Update the displayed geometries.

        This method updates the Open3D visualizer based on the outputs of the annotators.
        For the first update, it also sets up the camera and coordinate frame.

        :param geometries: Geometries to display. Can be:

        .. note::
            The dict format follows Open3D's draw() convention. See:
            https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/draw.py
        """
        if geometries is None:
            return

        # local method to add a single geometry. either based on the geometry being fully
        # defined with a dict or being a plain geometry object
        def add(g: Union[o3d.geometry.PointCloud, Dict, List], n: int) -> None:
            # Skip empty point clouds as they generate errors during the update
            if isinstance(g, o3d.geometry.PointCloud) and len(g.points) == 0:
                return

            if isinstance(g, dict):
                self.main_vis.add_geometry(g)
                name = g["name"]
            else:
                name = "Object " + str(n)
                self.main_vis.add_geometry(name, g)

            self.visualized_geometries.append(name)

        # Add all geometries from the given parameter.
        # It is safe to either input a plain geometry object or a list of objects.
        def add_all(
            geometries_to_add: Union[o3d.geometry.Geometry, Dict, List],
        ) -> None:
            n = 1
            if isinstance(geometries_to_add, list):
                for g in geometries_to_add:
                    add(g, n)
                    n += 1
            elif geometries_to_add is not None:
                add(geometries_to_add, n)

        if self.first_cloud:

            def add_first_cloud() -> None:
                add_all(geometries)

                self.main_vis.reset_camera_to_default()
                self.main_vis.setup_camera(
                    60,
                    [0, 0, 3],  # gaze coordinates
                    [0, 0, -1.5],  # camera position
                    [0, -1, 0],
                )  # turn cloud in viewer? from tutorial

                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.3
                )

                # The origin should always stay in the window. It will not be added to the
                # house-keeping list self.visualized_geometries
                self.main_vis.add_geometry("origin", coordinate_frame)

            add_first_cloud()
            self.first_cloud = False
        else:

            def update_with_cloud() -> None:
                for vg in self.visualized_geometries:
                    self.main_vis.remove_geometry(vg)
                self.visualized_geometries = []

                add_all(geometries)

            update_with_cloud()
