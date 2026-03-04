#!/usr/bin/env python

from __future__ import annotations

import os
import signal
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Optional, Dict

import rclpy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTableWidget,
    QCheckBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QLabel,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QScrollArea,
)
from typing_extensions import Callable

from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
    ShapeSource,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.collision_checking.collision_matrix import CollisionCheck
from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body


class DisableCollisionReason(Enum):
    """
    Enum for reasons why two bodies are disabled from colliding.
    """

    Unknown = -1
    Never = 1
    Adjacent = 2
    Default = 3
    AlmostAlways = 4


reason_color_map = {
    DisableCollisionReason.Never: (163, 177, 233),  # blue
    DisableCollisionReason.Adjacent: (233, 163, 163),  # red
    DisableCollisionReason.AlmostAlways: (233, 163, 231),  # purple
    DisableCollisionReason.Default: (233, 231, 163),  # yellow
    DisableCollisionReason.Unknown: (153, 76, 0),  # brown
    None: (100, 255, 100),
}


@dataclass
class SelfCollisionMatrixInterface:
    """
    Interface for managing self-collision matrix.
    """

    world: World = field(init=False)
    """
    World instance for managing bodies.
    """
    _reasons: Dict[Tuple[Body, Body], DisableCollisionReason] = field(
        init=False, default_factory=dict
    )
    """
    Dictionary for storing reasons for allowing collisions for pairs of bodies.
    """
    self_collision_matrix_rule: SelfCollisionMatrixRule = field(init=False)
    """
    SelfCollisionMatrixRule instance for computing self-collision matrix.
    """
    robot: AbstractRobot = field(init=False)
    """
    MinimalRobot instance for computing self-collision matrix.
    """

    def __post_init__(self):
        self.world = World()
        with self.world.modify_world():
            self.world.add_body(Body(name=PrefixedName("map")))
        VizMarkerPublisher(
            _world=self.world, node=rospy.node, shape_source=ShapeSource.COLLISION_ONLY
        ).with_tf_publisher()

    def load_urdf(self, urdf_path: str):
        robot_world = URDFParser.from_file(urdf_path).parse()
        self.self_collision_matrix_rule = SelfCollisionMatrixRule()
        self.robot = MinimalRobot.from_world(robot_world)
        with self.world.modify_world():
            self.world.clear()
            self.world.add_body(map := Body(name=PrefixedName("map")))
            self.world.merge_world(
                robot_world, FixedConnection(parent=map, child=robot_world.root)
            )

    def dye_all_bodies_white_transparent(self):
        with self.world.modify_world():
            for body in self.world.bodies_with_collision:
                for shape in body.collision.shapes:
                    if body in self.self_collision_matrix_rule.allowed_collision_bodies:
                        shape.color = Color(1.0, 0.0, 0.0, 0.25)
                        continue
                    shape.color = Color(1.0, 1.0, 1.0, 0.5)

    @property
    def bodies(self) -> List[Body]:
        return list(sorted(self.world.bodies_with_collision, key=lambda x: x.name.name))

    @property
    def enabled_bodies(self) -> List[Body]:
        return [
            body
            for body in self.bodies
            if body not in self.self_collision_matrix_rule.allowed_collision_bodies
        ]

    def sort_bodies(self, body_a: Body, body_b: Body) -> tuple[Body, Body]:
        if body_a == body_b:
            return body_a, body_b
        collision_check = CollisionCheck.create_and_validate(body_a, body_b)
        return collision_check.body_a, collision_check.body_b

    def get_reason_for_pair(
        self, body_a: Body, body_b: Body
    ) -> Optional[DisableCollisionReason]:
        body_a, body_b = self.sort_bodies(body_a, body_b)
        return self._reasons.get((body_a, body_b), None)

    def set_reason_for_pair(
        self, body_a: Body, body_b: Body, reason: DisableCollisionReason | None
    ):
        body_a, body_b = self.sort_bodies(body_a, body_b)
        self._reasons[body_a, body_b] = reason

    def compute_self_collision_matrix(
        self, progress_bar: Callable[[int, str], None], **kwargs: dict
    ):
        self.self_collision_matrix_rule.compute_self_collision_matrix(
            robot=self.robot, progress_callback=progress_bar, **kwargs
        )
        self._reasons = {}
        for collision_check in self.self_collision_matrix_rule.allowed_collision_pairs:
            self.set_reason_for_pair(
                collision_check.body_a,
                collision_check.body_b,
                DisableCollisionReason.Unknown,
            )

    def load_srdf(self, srdf_path: str):
        self.self_collision_matrix_rule = SelfCollisionMatrixRule.from_collision_srdf(
            srdf_path, self.world
        )
        for collision_check in self.self_collision_matrix_rule.allowed_collision_pairs:
            self.set_reason_for_pair(
                collision_check.body_a,
                collision_check.body_b,
                DisableCollisionReason.Unknown,
            )

    def safe_srdf(self, file_path: str):
        self.self_collision_matrix_rule.save_self_collision_matrix(
            self.robot.name.name, file_path
        )

    def add_body(self, body: Body):
        self.self_collision_matrix_rule.allowed_collision_bodies.discard(body)

    def remove_body(self, body: Body):
        self.self_collision_matrix_rule.allowed_collision_bodies.add(body)

    def add_pair(self, body_a: Body, body_b: Body, reason: DisableCollisionReason):
        collision_check = CollisionCheck.create_and_validate(body_a, body_b)
        self.self_collision_matrix_rule.allowed_collision_pairs.add(collision_check)
        self.set_reason_for_pair(body_a, body_b, reason)

    def remove_pair(self, body_a: Body, body_b: Body):
        collision_check = CollisionCheck.create_and_validate(body_a, body_b)
        self.self_collision_matrix_rule.allowed_collision_pairs.discard(collision_check)
        self.set_reason_for_pair(body_a, body_b, None)


@dataclass
class ReasonCheckBox(QCheckBox):
    """
    A checkbox for the cells of Table.
    """

    row: int
    column: int
    table: CollisionMatrixTable
    """
    Backreference to the Table instance.
    """
    self_collision_matrix_interface: SelfCollisionMatrixInterface
    """
    Reference to a SelfCollisionMatrixInterface instance which is shares by other ui components.
    """

    def __post_init__(self):
        super().__init__()

    def connect_callback(self):
        self.stateChanged.connect(self.checkbox_callback)

    def sync_reason(self):
        """
        Synchronizes the checkbox with the reason for the pair.
        """
        body_a = self.table.table_id_to_body(self.row)
        body_b = self.table.table_id_to_body(self.column)
        reason = self.self_collision_matrix_interface.get_reason_for_pair(
            body_a, body_b
        )
        self.setChecked(reason is not None)
        self.setStyleSheet(f"background-color: rgb{reason_color_map[reason]};")

    def checkbox_callback(self, state, update_range: bool = True):
        if update_range:
            self.update_range(state)
        body_a = self.table.table_id_to_body(self.row)
        body_b = self.table.table_id_to_body(self.column)
        if state == Qt.Checked:
            reason = DisableCollisionReason.Unknown
        else:
            reason = None
        self.table.update_reason(body_a, body_b, reason)

    def update_range(self, state: Qt.CheckState):
        """
        Update all selected checkboxes in the table to the values of this checkbox.
        :param state: New state of this checkbox
        """
        self.table.selectedRanges()
        for range_ in self.table.selectedRanges():
            for row in range(range_.topRow(), range_.bottomRow() + 1):
                for column in range(range_.leftColumn(), range_.rightColumn() + 1):
                    if row == column:
                        continue
                    item = self.table.get_cell(row, column)
                    # if state != item.checkState():
                    item.checkbox_callback(state, False)


@dataclass
class CollisionMatrixTable(QTableWidget):
    """
    Table for displaying and editing self-collision matrix.
    """

    self_collision_matrix_interface: SelfCollisionMatrixInterface
    """
    Reference to a SelfCollisionMatrixInterface instance which is shares by other ui components.
    """

    def __post_init__(self):
        super().__init__()
        self.cellClicked.connect(self._table_item_callback)

    def get_cell(self, row, column):
        return self.cellWidget(row, column).layout().itemAt(0).widget()

    def table_id_to_body(self, index: int) -> Body:
        return self.bodies[index]

    def body_to_table_id(self, body: Body) -> int:
        return self.bodies.index(body)

    def update_reason(
        self, body_a: Body, body_b: Body, new_reason: Optional[DisableCollisionReason]
    ):
        if new_reason is None:
            self.self_collision_matrix_interface.remove_pair(body_a, body_b)
        self.self_collision_matrix_interface.set_reason_for_pair(
            body_a, body_b, new_reason
        )
        row = self.body_to_table_id(body_a)
        column = self.body_to_table_id(body_b)
        self.get_cell(row, column).sync_reason()
        self.get_cell(column, row).sync_reason()

    def _table_item_callback(self, row, column):
        body_a = self.table_id_to_body(row)
        body_b = self.table_id_to_body(column)
        color = reason_color_map[DisableCollisionReason.Unknown]
        color_msg = Color(
            R=color[0] / 255.0, G=color[1] / 255.0, B=color[2] / 255.0, A=1.0
        )

        with self.self_collision_matrix_interface.world.modify_world():
            self.self_collision_matrix_interface.dye_all_bodies_white_transparent()
            body_a.collision.dye_shapes(color_msg)
            body_b.collision.dye_shapes(color_msg)

    def add_table_item(self, row, column):
        checkbox = ReasonCheckBox(
            table=self,
            row=row,
            column=column,
            self_collision_matrix_interface=self.self_collision_matrix_interface,
        )
        checkbox.sync_reason()
        checkbox.connect_callback()
        if row == column:
            checkbox.setDisabled(True)

        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(checkbox)
        layout.setAlignment(checkbox, Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        widget.setLayout(layout)
        self.setCellWidget(row, column, widget)

    @property
    def bodies(self) -> list[Body]:
        return self.self_collision_matrix_interface.bodies

    @property
    def enabled_bodies(self) -> list[Body]:
        return self.self_collision_matrix_interface.enabled_bodies

    @property
    def body_names(self) -> list[str]:
        return [body.name.name for body in self.bodies]

    def synchronize(self):
        self.self_collision_matrix_interface.world.notify_state_change()
        self.clear()
        self.setRowCount(len(self.bodies))
        self.setColumnCount(len(self.bodies))
        self.setHorizontalHeaderLabels(self.body_names)
        self.setVerticalHeaderLabels(self.body_names)

        for row_id, link1 in enumerate(self.bodies):
            if link1 not in self.enabled_bodies:
                self.hideRow(row_id)
            for column_id, link2 in enumerate(self.bodies):
                self.add_table_item(row_id, column_id)
                if link2 not in self.enabled_bodies:
                    self.hideColumn(column_id)

        num_rows = self.rowCount()

        widths = []

        for row_id in range(num_rows):
            item = self.item(row_id, 0)
            if item is not None:
                widths.append(item.sizeHint().width())
        if widths:
            self.setColumnWidth(0, max(widths))


def get_readable_color(red: float, green: float, blue: float) -> Tuple[int, int, int]:
    luminance = ((0.299 * red) + (0.587 * green) + (0.114 * blue)) / 255
    if luminance > 0.5:
        return 0, 0, 0
    else:
        return 255, 255, 255


class ProgressBarWithText(QProgressBar):
    def set_progress(self, value: int, text: Optional[str] = None):
        value = int(min(max(value, 0), 100))
        self.setValue(value)
        if text is not None:
            self.setFormat(f"{text}: %p%")
        self.parent().repaint()


class HorizontalLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class ComputeSelfCollisionMatrixParameterDialog(QDialog):
    """
    Dialog for setting parameters for computing the self collision matrix.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Set Parameters")

        self.parameters = {}

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(
            QLabel(
                "Set Thresholds for computing the self collision matrix. \n"
                "Collision checks for entries in this matrix will not be performed."
            )
        )
        self.layout.addWidget(HorizontalLine())
        self.layout.addWidget(
            QLabel(
                "Phase 1: Add link pairs that are in contact in default joint state."
            )
        )
        self.layout.addLayout(
            self.make_parameter_entry(
                "Distance threshold:", 0.0, "distance_threshold_zero"
            )
        )
        self.layout.addWidget(HorizontalLine())
        self.layout.addWidget(
            QLabel("Phase 2: Add link pairs that are (almost) always in collision.")
        )
        self.layout.addLayout(
            self.make_parameter_entry(
                "Do distance checks for:",
                200,
                "number_of_tries_always",
                unit="random configurations.",
                int_=True,
            )
        )
        self.layout.addLayout(
            self.make_parameter_entry(
                "Add all pairs that were closer than",
                0.005,
                "distance_threshold_always",
            )
        )
        self.layout.addLayout(
            self.make_parameter_entry(
                "in", 0.95, "almost_percentage", unit="% of configurations."
            )
        )
        self.layout.addWidget(HorizontalLine())
        self.layout.addWidget(
            QLabel("Phase 3: Add link pairs that are never in collision.")
        )
        self.layout.addLayout(
            self.make_parameter_entry(
                "Do distance checks for ",
                10000,
                "number_of_tries_never",
                unit="random configurations.",
                int_=True,
            )
        )
        self.layout.addLayout(
            self.make_parameter_entry(
                "Out of all pairs that are between",
                -0.02,
                "distance_threshold_never_min",
            )
        )
        self.layout.addLayout(
            self.make_parameter_entry(
                "and ", 0.05, "distance_threshold_never_max", unit="m apart."
            )
        )
        self.layout.addLayout(
            self.make_parameter_entry(
                "Step 3.1: Add pairs that were always above",
                0.0,
                "distance_threshold_never_zero",
                unit="m apart.",
            )
        )
        self.layout.addLayout(
            self.make_parameter_entry(
                "Step 3.2: Add links that were never further than",
                0.05,
                "distance_threshold_never_range",
                unit="m apart.",
            )
        )

        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

    def make_parameter_entry(
        self,
        text: str,
        default: float,
        parameter_name: str,
        int_: bool = False,
        unit: str = "m",
    ) -> QVBoxLayout:
        inner_box = QHBoxLayout()
        edit = QLineEdit(self)
        inner_box.addWidget(QLabel(text))
        inner_box.addWidget(edit)
        inner_box.addWidget(QLabel(unit))
        edit.setText(str(default))
        if int_:
            edit.setValidator(QIntValidator(self))
        else:
            edit.setValidator(QDoubleValidator(self))

        outer_box = QVBoxLayout()
        outer_box.addLayout(inner_box)
        self.parameters[parameter_name] = edit
        return outer_box

    def get_parameter_map(self) -> Dict[str, float]:
        params = {
            param_name: float(edit.text())
            for param_name, edit in self.parameters.items()
        }
        return params


class ClickableLabel(QLabel):
    """
    A label that triggers the click of its parent checkbox.
    """

    def mousePressEvent(self, event):
        self.parent().checkbox.click()


@dataclass
class DisableBodyItem(QWidget):
    """
    One item in DisableBodiesDialog.
    """

    body: Body
    """
    The body represented by this item.
    """
    self_collision_matrix_interface: SelfCollisionMatrixInterface
    """
    Reference to a SelfCollisionMatrixInterface instance which is shares by other ui components.
    """
    parent: QWidget | None = None

    def __post_init__(self):
        super().__init__(self.parent)
        self.checkbox = QCheckBox()
        self.checkbox.stateChanged.connect(self.checkbox_callback)
        self.label = ClickableLabel(self.text, self)

        layout = QHBoxLayout()
        layout.addWidget(self.checkbox)
        layout.addWidget(self.label)
        layout.addStretch()
        self.setLayout(layout)

    @property
    def text(self) -> str:
        return self.body.name.name

    def checkbox_callback(self, state):
        if state == Qt.Checked:
            self.self_collision_matrix_interface.remove_body(self.body)
        else:
            self.self_collision_matrix_interface.add_body(self.body)
        self.self_collision_matrix_interface.dye_all_bodies_white_transparent()

    def set_checked(self, new_state: bool):
        self.checkbox.setChecked(new_state)

    def is_checked(self):
        return self.checkbox.isChecked()


@dataclass
class DisableBodiesDialog(QDialog):
    """
    Dialog for disabling bodies in the self collision matrix.
    """

    self_collision_matrix_interface: SelfCollisionMatrixInterface
    """
    Reference to a SelfCollisionMatrixInterface instance which is shares by other ui components.
    """

    def __post_init__(self):
        super().__init__()
        self.setWindowTitle("Disable Bodies")
        self.layout = QVBoxLayout(self)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout.addWidget(self.scrollArea)

        self.scrollLayout = QVBoxLayout(self.scrollAreaWidgetContents)

        self.checkbox_widgets = []
        for body in self.self_collision_matrix_interface.bodies:
            checkbox_widget = DisableBodyItem(
                body, self.self_collision_matrix_interface
            )
            self.checkbox_widgets.append(checkbox_widget)
            self.scrollLayout.addWidget(checkbox_widget)
            checkbox_widget.set_checked(
                body not in self.self_collision_matrix_interface.enabled_bodies
            )

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        self.buttonBox.accepted.connect(self.accept)
        self.layout.addWidget(self.buttonBox)


@dataclass
class Application(QMainWindow):
    """
    The main application for the collision matrix tool.
    """

    self_collision_matrix_interface: SelfCollisionMatrixInterface = field(init=False)
    """
    Reference to a SelfCollisionMatrixInterface instance which is shares by other ui components.
    """
    timer: QTimer = field(init=False, default_factory=QTimer)
    """
    Timer used to update the ui periodically.
    """

    def __post_init__(self):
        super().__init__()
        self.self_collision_matrix_interface = SelfCollisionMatrixInterface()
        self.timer.start(1000)  # Time in milliseconds
        self.timer.timeout.connect(lambda: None)
        self.init_ui_components()

    def init_ui_components(self):
        """
        Initialize all ui components.
        """
        self.setWindowTitle("Self Collision Matrix Tool")
        self.setMinimumSize(800, 600)

        self.progress = ProgressBarWithText(self)

        self.table = CollisionMatrixTable(self.self_collision_matrix_interface)

        layout = QVBoxLayout()
        layout.addLayout(self._create_urdf_box_layout())
        self.horizontalLine = QFrame()
        self.horizontalLine.setFrameShape(QFrame.HLine)
        self.horizontalLine.setFrameShadow(QFrame.Sunken)
        layout.addWidget(self.horizontalLine)
        layout.addLayout(self._create_srdf_box_layout())
        layout.addWidget(self.progress)
        layout.addLayout(self._create_legend_box_layout())
        layout.addWidget(self.table)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.progress.set_progress(0, "Load urdf")

    def _create_srdf_box_layout(self) -> QHBoxLayout:
        self.load_srdf_button = QPushButton("Load from srdf")
        self.load_srdf_button.clicked.connect(self._load_srdf_button_callback)
        self.compute_srdf_button = QPushButton("Compute self collision matrix")
        self.compute_srdf_button.clicked.connect(self._compute_srdf_button_callback)
        self.disable_bodies_button = QPushButton("Disable links")
        self.disable_bodies_button.clicked.connect(self._disable_bodies_button_callback)
        self.save_srdf_button = QPushButton("Save as srdf")
        self.save_srdf_button.clicked.connect(self._save_srdf_button_callback)
        srdf_bottoms = QHBoxLayout()
        srdf_bottoms.addWidget(self.compute_srdf_button)
        srdf_bottoms.addWidget(self.disable_bodies_button)
        srdf_bottoms.addWidget(self.load_srdf_button)
        srdf_bottoms.addWidget(self.save_srdf_button)
        self.disable_srdf_buttons()
        return srdf_bottoms

    def _create_urdf_box_layout(self) -> QHBoxLayout:
        self.load_urdf_file_button = QPushButton("Load urdf from file")
        self.load_urdf_file_button.clicked.connect(self._load_urdf_file_button_callback)
        self.urdf_progress = ProgressBarWithText(self)
        self.urdf_progress.set_progress(0, "No urdf loaded")
        urdf_section = QHBoxLayout()
        urdf_section.addWidget(self.load_urdf_file_button)
        urdf_section.addWidget(self.urdf_progress)
        return urdf_section

    def _create_legend_box_layout(self) -> QHBoxLayout:
        legend = QHBoxLayout()

        for reason, color in reason_color_map.items():
            if reason is not None:
                label = QLabel(reason.name)
            else:
                label = QLabel("check collision")
            label.setStyleSheet(
                f"background-color: rgb{color}; color: rgb{get_readable_color(*color)};"
            )

            match reason:
                case DisableCollisionReason.Never:
                    tooltip = "These links are never in contact."
                case DisableCollisionReason.Unknown:
                    tooltip = "This link pair was disabled for an unknown reason."
                case DisableCollisionReason.Adjacent:
                    tooltip = (
                        "This link pair is only connected by joints that cannot move."
                    )
                case DisableCollisionReason.Default:
                    tooltip = (
                        "This link pair is in collision in the robot's default state."
                    )
                case DisableCollisionReason.AlmostAlways:
                    tooltip = "This link pair is almost always in collision."
                case _:
                    tooltip = "Collisions will be computed."

            label.setToolTip(tooltip)
            legend.addWidget(label)
        return legend

    def _compute_srdf_button_callback(self):
        dialog = ComputeSelfCollisionMatrixParameterDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            parameters = dialog.get_parameter_map()
            self.self_collision_matrix_interface.compute_self_collision_matrix(
                progress_bar=self.progress.set_progress,
                **parameters,
            )
            self.table.synchronize()
            self.progress.set_progress(100, "Done checking collisions")
        else:
            self.progress.set_progress(0, "Canceled collision checking")

    def _disable_bodies_button_callback(self):
        dialog = DisableBodiesDialog(self.self_collision_matrix_interface)
        dialog.exec_()
        self.table.synchronize()

    def disable_srdf_buttons(self):
        self._disable_srdf_buttons(True)

    def enable_srdf_buttons(self):
        self._disable_srdf_buttons(False)

    def _disable_srdf_buttons(self, active: bool):
        self.save_srdf_button.setDisabled(active)
        self.load_srdf_button.setDisabled(active)
        self.disable_bodies_button.setDisabled(active)
        self.compute_srdf_button.setDisabled(active)

    def _load_srdf_button_callback(self):
        srdf_file = self.popup_srdf_path_with_dialog(False)
        if srdf_file is None:
            return
        try:
            if os.path.isfile(srdf_file):
                self.self_collision_matrix_interface.load_srdf(srdf_file)
                self.table.synchronize()
                self.progress.set_progress(100, f"Loaded {srdf_file}")
            else:
                QMessageBox.critical(
                    self, "Error", f"File does not exist: \n{srdf_file}"
                )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))

    def _load_urdf_file_button_callback(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        urdf_file, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
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

            self.self_collision_matrix_interface.load_urdf(urdf_file)
            self.urdf_progress.set_progress(0, f"Loading {urdf_file}")
            self.urdf_progress.set_progress(10, f"Parsing {urdf_file}")
            self.urdf_progress.set_progress(
                50, f"Applying vhacd to concave meshes of {urdf_file}"
            )
            self.urdf_progress.set_progress(80, f"Updating table {urdf_file}")
            self.table.synchronize()
            self.enable_srdf_buttons()
            self.urdf_progress.set_progress(100, f"Loaded {urdf_file}")

    def popup_srdf_path_with_dialog(self, save: bool) -> str | None:
        """
        Creates the popup dialog for selecting the path to the srdf file.
        :param save: whether to open a save file dialog or an open file dialog.
        :return: the selected file path
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        if save:
            srdf_file, _ = QFileDialog.getSaveFileName(
                self,
                "QFileDialog.getSaveFileName()",
                filter="srdf files (*.srdf);;All files (*)",
                options=options,
            )
        else:
            srdf_file, _ = QFileDialog.getOpenFileName(
                self,
                "QFileDialog.getOpenFileName()",
                filter="srdf files (*.srdf);;All files (*)",
                options=options,
            )

        if srdf_file:
            self.__srdf_path = srdf_file
        else:
            srdf_file = None

        return srdf_file

    def _save_srdf_button_callback(self):
        srdf_path = self.popup_srdf_path_with_dialog(True)
        if srdf_path is not None:
            self.self_collision_matrix_interface.safe_srdf(
                file_path=srdf_path,
            )
            self.progress.set_progress(100, f"Saved {self.__srdf_path}")

    def die(self):
        if not rclpy.ok():
            QApplication.quit()


def handle_sigint(sig, frame):
    """Handler for the SIGINT signal."""
    QApplication.quit()


if __name__ == "__main__":
    rospy.init_node("self_collision_matrix_updater")
    signal.signal(signal.SIGINT, handle_sigint)

    app = QApplication(sys.argv)
    window = Application()
    window.show()
    sys.exit(app.exec_())
