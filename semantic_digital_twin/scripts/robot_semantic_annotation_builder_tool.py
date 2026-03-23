#!/usr/bin/env python

from __future__ import annotations

import os
import signal
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import rclpy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QGridLayout,
    QGroupBox,
    QCheckBox,
    QLabel,
    QDialog,
)

from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
    ShapeSource,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.robots.abstract_robot import Neck, Arm
from semantic_digital_twin.robots.robot_mixins import (
    HasNeck,
    HasArms,
    SpecifiesLeftRightArm,
)


@dataclass
class RobotSemanticAnnotationBuilderInterface:
    """
    Interface for managing robot semantic annotations.
    """

    world: World = field(init=False)
    """
    World instance for managing bodies.
    """

    def __post_init__(self):
        self.world = World()
        with self.world.modify_world():
            self.world.add_body(Body(name=PrefixedName("map")))
        VizMarkerPublisher(
            _world=self.world, node=rospy.node, shape_source=ShapeSource.COLLISION_ONLY
        ).with_tf_publisher()

    def load_urdf(self, urdf_path: str):
        """
        Loads a URDF file and merges it into the world.
        """
        robot_world = URDFParser.from_file(urdf_path).parse()
        with self.world.modify_world():
            self.world.clear()
            self.world.add_body(map_body := Body(name=PrefixedName("map")))
            self.world.merge_world(
                robot_world, FixedConnection(parent=map_body, child=robot_world.root)
            )

    def reset_body_colors(self):
        """
        Sets all body colors to white transparent.
        """
        with self.world.modify_world():
            for body in self.world.bodies_with_collision:
                for shape in body.collision.shapes:
                    shape.color = Color(1.0, 1.0, 1.0, 0.5)

    def highlight_body(self, body: Body):
        """
        Highlights the given body by changing its color.
        """
        highlight_color = Color(1.0, 0.0, 0.0, 1.0)
        self.reset_body_colors()
        with self.world.modify_world():
            body.collision.dye_shapes(highlight_color)

    @property
    def bodies(self) -> List[Body]:
        """
        Returns a sorted list of all bodies with collision.
        """
        return list(sorted(self.world.bodies_with_collision, key=lambda x: x.name.name))


@dataclass
class BodyButton(QPushButton):
    """
    A button representing a robot body.
    """

    body: Body
    application: Application

    def __post_init__(self):
        super().__init__(self.body.name.name)
        self.clicked.connect(self.on_click)
        self.setMinimumHeight(40)

    def on_click(self):
        """
        Callback for button click.
        """
        self.application.interface.highlight_body(self.body)
        self.application.last_highlighted_body = self.body


class ProgressBarWithText(QProgressBar):
    """
    A progress bar that displays text.
    """

    def set_progress(self, value: int, text: Optional[str] = None):
        """
        Sets the progress value and optional text.
        """
        value = int(min(max(value, 0), 100))
        self.setValue(value)
        if text is not None:
            self.setFormat(f"{text}: %p%")
        self.parent().repaint()


@dataclass
class ComponentSelectionDialog(QDialog):
    """
    Dialog for choosing robot components.
    """

    has_neck: bool = False
    has_arms: bool = False
    specifies_left_right_arm: bool = False

    def __post_init__(self):
        super().__init__()
        self.setWindowTitle("Choose robot components")
        self.init_ui()

    def init_ui(self):
        """
        Initialize the dialog UI.
        """
        layout = QVBoxLayout()

        self.has_neck_checkbox = QCheckBox("HasNeck")
        self.has_neck_checkbox.setChecked(self.has_neck)
        self.has_arms_checkbox = QCheckBox("HasArms")
        self.has_arms_checkbox.setChecked(self.has_arms)
        self.specifies_left_right_arm_checkbox = QCheckBox("SpecifiesLeftRightArm")
        self.specifies_left_right_arm_checkbox.setChecked(self.specifies_left_right_arm)

        self.has_arms_checkbox.toggled.connect(self._handle_has_arms_toggled)
        self.specifies_left_right_arm_checkbox.toggled.connect(
            self._handle_specifies_left_right_arm_toggled
        )

        layout.addWidget(self.has_neck_checkbox)
        layout.addWidget(self.has_arms_checkbox)
        layout.addWidget(self.specifies_left_right_arm_checkbox)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def _handle_has_arms_toggled(self, checked: bool):
        """
        Handles HasArms checkbox toggle, ensuring mutual exclusivity.
        """
        if checked:
            self.specifies_left_right_arm_checkbox.setChecked(False)

    def _handle_specifies_left_right_arm_toggled(self, checked: bool):
        """
        Handles SpecifiesLeftRightArm checkbox toggle, ensuring mutual exclusivity.
        """
        if checked:
            self.has_arms_checkbox.setChecked(False)

    def get_selection(self) -> dict:
        """
        Returns the current selection.
        """
        return {
            "HasNeck": self.has_neck_checkbox.isChecked(),
            "HasArms": self.has_arms_checkbox.isChecked(),
            "SpecifiesLeftRightArm": self.specifies_left_right_arm_checkbox.isChecked(),
        }


@dataclass
class NeckWidget(QGroupBox):
    """
    Widget for managing neck properties.
    """

    application: Application

    def __post_init__(self):
        super().__init__("Neck")
        self.init_ui()

    def init_ui(self):
        """
        Initializes the UI.
        """
        layout = QVBoxLayout()
        remove_button = QPushButton("Remove Neck")
        remove_button.clicked.connect(self.application.remove_neck)
        layout.addWidget(remove_button)

        pitch_body_name = (
            self.application.neck.pitch_body.name.name
            if self.application.neck.pitch_body
            else "None"
        )
        pitch_button = QPushButton(f"pitch body: {pitch_body_name}")
        pitch_button.clicked.connect(self.application.assign_body_to_neck_pitch)
        layout.addWidget(pitch_button)

        yaw_body_name = (
            self.application.neck.yaw_body.name.name
            if self.application.neck.yaw_body
            else "None"
        )
        yaw_button = QPushButton(f"yaw body: {yaw_body_name}")
        yaw_button.clicked.connect(self.application.assign_body_to_neck_yaw)
        layout.addWidget(yaw_button)

        root_body_name = (
            self.application.neck.root.name.name
            if self.application.neck.root
            else "None"
        )
        root_button = QPushButton(f"Root Body: {root_body_name}")
        root_button.clicked.connect(self.application.assign_body_to_neck_root)
        layout.addWidget(root_button)

        tip_body_name = (
            self.application.neck.tip.name.name if self.application.neck.tip else "None"
        )
        tip_button = QPushButton(f"Tip Body: {tip_body_name}")
        tip_button.clicked.connect(self.application.assign_body_to_neck_tip)
        layout.addWidget(tip_button)

        self.setLayout(layout)


@dataclass
class ArmWidget(QGroupBox):
    """
    Widget for managing arm properties.
    """

    application: Application
    arm: Arm

    def __post_init__(self):
        super().__init__("Arm")
        self.init_ui()

    def init_ui(self):
        """
        Initializes the UI.
        """
        layout = QVBoxLayout()
        remove_button = QPushButton("Remove Arm")
        remove_button.clicked.connect(lambda: self.application.remove_arm(self.arm))
        layout.addWidget(remove_button)

        root_body_name = self.arm.root.name.name if self.arm.root else "None"
        root_button = QPushButton(f"Root Body: {root_body_name}")
        root_button.clicked.connect(
            lambda: self.application.assign_body_to_arm_root(self.arm)
        )
        layout.addWidget(root_button)

        tip_body_name = self.arm.tip.name.name if self.arm.tip else "None"
        tip_button = QPushButton(f"Tip Body: {tip_body_name}")
        tip_button.clicked.connect(
            lambda: self.application.assign_body_to_arm_tip(self.arm)
        )
        layout.addWidget(tip_button)

        self.setLayout(layout)


@dataclass
class Application(QMainWindow):
    """
    The main application for the robot semantic annotation builder tool.
    """

    interface: RobotSemanticAnnotationBuilderInterface = field(
        init=False, default_factory=RobotSemanticAnnotationBuilderInterface
    )
    """
    Reference to a RobotSemanticAnnotationBuilderInterface instance.
    """
    timer: QTimer = field(init=False, default_factory=QTimer)
    """
    Timer used to update the ui periodically.
    """
    chosen_components: dict = field(
        default_factory=lambda: {
            "HasNeck": False,
            "HasArms": False,
            "SpecifiesLeftRightArm": False,
        }
    )
    """
    Dictionary storing the state of chosen robot components.
    """
    last_highlighted_body: Optional[Body] = None
    """
    The last highlighted body.
    """
    neck: Optional[Neck] = None
    """
    The neck of the robot.
    """
    arms: List[Arm] = field(default_factory=list)
    """
    The arms of the robot.
    """

    def __post_init__(self):
        super().__init__()
        self.timer.start(1000)
        self.timer.timeout.connect(lambda: None)
        self.init_ui_components()

    def init_ui_components(self):
        """
        Initialize all ui components.
        """
        self.setWindowTitle("Robot Semantic Annotation Builder Tool")
        self.setMinimumSize(1000, 600)

        self.urdf_progress = ProgressBarWithText(self)
        self.urdf_progress.set_progress(0, "No urdf loaded")

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.body_buttons_widget = QWidget()
        self.body_buttons_layout = QGridLayout(self.body_buttons_widget)
        self.body_buttons_layout.setSpacing(0)
        self.body_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.body_buttons_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.body_buttons_widget)

        left_layout = QVBoxLayout()
        left_layout.addLayout(self._create_urdf_box_layout())
        left_layout.addWidget(self.scroll_area)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=4)
        main_layout.addLayout(self._create_chosen_components_layout(), stretch=1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _create_chosen_components_layout(self) -> QVBoxLayout:
        """
        Creates the layout for chosen components on the right.
        """
        layout = QVBoxLayout()
        self.choose_components_button = QPushButton("Choose Robot Components")
        self.choose_components_button.setEnabled(False)
        self.choose_components_button.clicked.connect(
            self._open_component_selection_dialog
        )
        layout.addWidget(self.choose_components_button)

        self.chosen_components_label = QLabel("Chosen Components:")
        layout.addWidget(self.chosen_components_label)

        self.components_list_layout = QVBoxLayout()
        layout.addLayout(self.components_list_layout)
        layout.addStretch()
        return layout

    def _open_component_selection_dialog(self):
        """
        Opens the component selection dialog.
        """
        dialog = ComponentSelectionDialog(
            has_neck=self.chosen_components["HasNeck"],
            has_arms=self.chosen_components["HasArms"],
            specifies_left_right_arm=self.chosen_components["SpecifiesLeftRightArm"],
        )
        if dialog.exec_():
            self.chosen_components = dialog.get_selection()
            self._update_chosen_components_list()

    def _update_chosen_components_list(self):
        """
        Updates the list of chosen components on the right.
        """
        while self.components_list_layout.count():
            item = self.components_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for component, chosen in self.chosen_components.items():
            if chosen:
                self.components_list_layout.addWidget(QLabel(f"- {component}"))
                if component == "HasNeck":
                    add_neck_button = QPushButton("Add Neck")
                    add_neck_button.setEnabled(self.neck is None)
                    add_neck_button.clicked.connect(self._add_neck_callback)
                    self.components_list_layout.addWidget(add_neck_button)

                if component in ["HasArms", "SpecifiesLeftRightArm"]:
                    add_arm_button = QPushButton("Add Arm")
                    add_arm_button.clicked.connect(self._add_arm_callback)
                    self.components_list_layout.addWidget(add_arm_button)

        if self.neck:
            self.components_list_layout.addWidget(NeckWidget(self))

        for arm in self.arms:
            self.components_list_layout.addWidget(ArmWidget(self, arm))

    def _add_neck_callback(self):
        """
        Callback for adding a neck.
        """
        self.neck = Neck(name=PrefixedName("neck"))
        self._update_chosen_components_list()

    def _add_arm_callback(self):
        """
        Callback for adding an arm.
        """
        arm_name = f"arm_{len(self.arms)}"
        arm = Arm(name=PrefixedName(arm_name))
        self.arms.append(arm)
        self._update_chosen_components_list()

    def remove_neck(self):
        """
        Removes the neck.
        """
        self.neck = None
        self._update_chosen_components_list()

    def remove_arm(self, arm: Arm):
        """
        Removes the given arm.
        """
        self.arms.remove(arm)
        self._update_chosen_components_list()

    def assign_body_to_neck_pitch(self):
        """
        Assigns the last highlighted body to the neck's pitch body.
        """
        if self.neck and self.last_highlighted_body:
            self.neck.pitch_body = self.last_highlighted_body
            self._update_chosen_components_list()

    def assign_body_to_neck_yaw(self):
        """
        Assigns the last highlighted body to the neck's yaw body.
        """
        if self.neck and self.last_highlighted_body:
            self.neck.yaw_body = self.last_highlighted_body
            self._update_chosen_components_list()

    def assign_body_to_neck_root(self):
        """
        Assigns the last highlighted body to the neck's root body.
        """
        if self.neck and self.last_highlighted_body:
            self.neck.root = self.last_highlighted_body
            self._update_chosen_components_list()

    def assign_body_to_neck_tip(self):
        """
        Assigns the last highlighted body to the neck's tip body.
        """
        if self.neck and self.last_highlighted_body:
            self.neck.tip = self.last_highlighted_body
            self._update_chosen_components_list()

    def assign_body_to_arm_root(self, arm: Arm):
        """
        Assigns the last highlighted body to the arm's root body.
        """
        if self.last_highlighted_body:
            arm.root = self.last_highlighted_body
            self._update_chosen_components_list()

    def assign_body_to_arm_tip(self, arm: Arm):
        """
        Assigns the last highlighted body to the arm's tip body.
        """
        if self.last_highlighted_body:
            arm.tip = self.last_highlighted_body
            self._update_chosen_components_list()

    def _create_urdf_box_layout(self) -> QHBoxLayout:
        """
        Creates the layout for URDF loading.
        """
        self.load_urdf_file_button = QPushButton("Load urdf from file")
        self.load_urdf_file_button.clicked.connect(self._load_urdf_file_button_callback)
        urdf_section = QHBoxLayout()
        urdf_section.addWidget(self.load_urdf_file_button)
        urdf_section.addWidget(self.urdf_progress)
        return urdf_section

    def _load_urdf_file_button_callback(self):
        """
        Callback for the load URDF button.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        urdf_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select URDF File",
            "",
            "urdf files (*.urdf);;All files (*)",
            options=options,
        )
        if urdf_file:
            if not os.path.isfile(urdf_file):
                QMessageBox.critical(
                    self, "Error", f"File does not exist: \n{urdf_file}"
                )
                return

            self.interface.load_urdf(urdf_file)
            self.urdf_progress.set_progress(100, f"Loaded {urdf_file}")
            self.refresh_body_buttons()
            if self.interface.bodies:
                self.choose_components_button.setEnabled(True)

    def refresh_body_buttons(self):
        """
        Refreshes the list of body buttons in a grid layout.
        """
        while self.body_buttons_layout.count():
            item = self.body_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        columns = 5
        for index, body in enumerate(self.interface.bodies):
            button = BodyButton(body=body, application=self)
            row = index // columns
            column = index % columns
            self.body_buttons_layout.addWidget(button, row, column)


def handle_sigint(sig, frame):
    """Handler for the SIGINT signal."""
    rospy.shutdown()
    QApplication.quit()


if __name__ == "__main__":
    rospy.init_node("robot_semantic_annotation_builder")
    signal.signal(signal.SIGINT, handle_sigint)

    app = QApplication(sys.argv)
    window = Application()
    window.show()
    exit_code = app.exec_()
    rospy.shutdown()
    sys.exit(exit_code)
