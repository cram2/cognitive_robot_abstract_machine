from dataclasses import dataclass, field, InitVar
from typing import List, Optional, Union
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
)
from PySide6.QtCore import Qt

from .controller import ModelController
from .variable_constraint_widget import VariableConstraintWidget
from PySide6.QtGui import QIcon
from random_events.product_algebra import SimpleEvent, Event


@dataclass
class ModeWidget(QWidget):
    """
    The Mode page widget of the GUI.
    Calculates and displays the Most Probable Explanation (MPE).
    """

    controller: ModelController
    """
    The model controller.
    """

    parent: InitVar[Optional[QWidget]] = None
    """
    The parent widget.
    """

    evidence_widgets: List[VariableConstraintWidget] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    """
    The list of variable constraint widgets for the evidence.
    """

    modes: List[SimpleEvent] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    """
    The list of simple events representing the calculated modes.
    """

    likelihood: float = field(default=0.0, init=False, repr=False, compare=False)
    """
    The maximum likelihood value of the modes.
    """

    current_mode_index: int = field(default=0, init=False, repr=False, compare=False)
    """
    The index of the currently displayed mode.
    """

    calculate_button: QPushButton = field(init=False, repr=False, compare=False)
    """
    The button to calculate the mode.
    """

    result_container: QWidget = field(init=False, repr=False, compare=False)
    """
    The widget containing the results and navigation.
    """

    result_layout: QVBoxLayout = field(init=False, repr=False, compare=False)
    """
    The layout for the results.
    """

    prev_button: QPushButton = field(init=False, repr=False, compare=False)
    """
    The button to navigate to the previous mode.
    """

    next_button: QPushButton = field(init=False, repr=False, compare=False)
    """
    The button to navigate to the next mode.
    """

    mode_info_label: QLabel = field(init=False, repr=False, compare=False)
    """
    The label displaying mode information and likelihood.
    """

    mode_details_scroll: QScrollArea = field(init=False, repr=False, compare=False)
    """
    The scroll area for mode details.
    """

    mode_details_container: QWidget = field(init=False, repr=False, compare=False)
    """
    The widget container for mode rows.
    """

    mode_details_layout: QVBoxLayout = field(init=False, repr=False, compare=False)
    """
    The layout for the mode rows.
    """

    def __post_init__(self, parent: Optional[QWidget]):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Evidence Section
        evidence_section = self.create_evidence_section()
        main_layout.addWidget(evidence_section)

        # Calculate Button
        self.calculate_button = QPushButton("Calculate Mode")
        self.calculate_button.setStyleSheet("font-size: 14pt; padding: 10px;")
        self.calculate_button.clicked.connect(self.on_calculate)
        main_layout.addWidget(self.calculate_button)

        # Results area
        self.result_container = QWidget()
        self.result_layout = QVBoxLayout(self.result_container)

        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton()
        self.prev_button.setIcon(QIcon("icon:/primary/leftarrow.svg"))
        self.prev_button.setFixedWidth(50)
        self.prev_button.clicked.connect(self.show_prev_mode)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)

        self.mode_info_label = QLabel("Click Calculate to see results")
        self.mode_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_info_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        nav_layout.addWidget(self.mode_info_label)

        self.next_button = QPushButton()
        self.next_button.setIcon(QIcon("icon:/primary/rightarrow.svg"))
        self.next_button.setFixedWidth(50)
        self.next_button.clicked.connect(self.show_next_mode)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)

        self.result_layout.addLayout(nav_layout)

        # Mode Details
        self.mode_details_scroll = QScrollArea()
        self.mode_details_scroll.setWidgetResizable(True)
        self.mode_details_container = QWidget()
        self.mode_details_layout = QVBoxLayout(self.mode_details_container)
        self.mode_details_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.mode_details_scroll.setWidget(self.mode_details_container)
        self.result_layout.addWidget(self.mode_details_scroll)

        main_layout.addWidget(self.result_container)

    def create_evidence_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)

        header_layout = QHBoxLayout()
        header_label = QLabel("Evidence")
        header_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        header_layout.addWidget(header_label)

        add_button = QPushButton()
        add_button.setIcon(QIcon("icon:/primary/checklist.svg"))
        add_button.setFixedWidth(30)
        add_button.clicked.connect(self.add_evidence_row)
        header_layout.addWidget(add_button)

        layout.addLayout(header_layout)

        self.evidence_scroll_area = QScrollArea()
        self.evidence_scroll_area.setWidgetResizable(True)
        self.evidence_scroll_area.setMaximumHeight(200)
        self.evidence_container = QWidget()
        self.evidence_rows_layout = QVBoxLayout(self.evidence_container)
        self.evidence_rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.evidence_scroll_area.setWidget(self.evidence_container)

        layout.addWidget(self.evidence_scroll_area)

        # Add initial row if model is already loaded
        if self.controller.model:
            self.add_evidence_row()

        return section

    def add_evidence_row(self):
        if not self.controller.model:
            return

        var_widget = VariableConstraintWidget(
            self.controller.model.variables, self.controller.priors
        )
        var_widget.changed.connect(self.update_variable_options)
        var_widget.removed.connect(
            lambda variable_widget=var_widget: self.remove_evidence_row(
                variable_widget, variable_widget
            )
        )
        self.evidence_widgets.append(var_widget)
        self.evidence_rows_layout.addWidget(var_widget)
        self.update_variable_options()

    def remove_evidence_row(
        self, row_widget: QWidget, var_widget: VariableConstraintWidget
    ):
        self.evidence_widgets.remove(var_widget)
        row_widget.deleteLater()
        self.update_variable_options()

    def update_variable_options(self):
        """
        Updates the available variables in all evidence widgets.
        """
        selected_vars = {
            w.variable_combo.currentData()
            for w in self.evidence_widgets
            if w.variable_combo.currentData() is not None
        }
        for w in self.evidence_widgets:
            w.update_available_variables(selected_vars)

    def build_evidence_event(self) -> Event:
        simple_event = SimpleEvent()
        for widget in self.evidence_widgets:
            constraint = widget.get_constraint()
            if constraint:
                var, val = constraint
                if var in simple_event:
                    # Treat multiple constraints on the same variable as a union
                    simple_event[var] = simple_event[var].union_with(val)
                else:
                    simple_event[var] = val

        if not simple_event:
            return Event()

        return Event(simple_event)

    def on_calculate(self):
        if not self.controller.model:
            self.mode_info_label.setText("No model loaded")
            return

        evidence_event = self.build_evidence_event()
        result = self.controller.calculate_mode(evidence_event)

        if result is None:
            self.mode_info_label.setText("Unsatisfiable")
            self.clear_mode_details()
            self.modes = []
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        mode_event, likelihood = result
        self.modes = mode_event.simple_sets
        self.likelihood = likelihood
        self.current_mode_index = 0
        self.update_mode_display()

    def update_mode_display(self):
        if not self.modes:
            return

        self.mode_info_label.setText(
            f"Result {self.current_mode_index + 1}/{len(self.modes)} (Likelihood: {self.likelihood:.5f})"
        )
        self.prev_button.setEnabled(self.current_mode_index > 0)
        self.next_button.setEnabled(self.current_mode_index < len(self.modes) - 1)

        self.display_mode(self.modes[self.current_mode_index])

    def clear_mode_details(self):
        while self.mode_details_layout.count():
            item = self.mode_details_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def display_mode(self, simple_event: SimpleEvent):
        self.clear_mode_details()

        # Display each variable and its restriction
        # Sort variables by name for consistency
        sorted_vars = sorted(simple_event.keys(), key=lambda v: v.name)

        for var in sorted_vars:
            restriction = simple_event[var]

            row = QFrame()
            row.setFrameShape(QFrame.Shape.StyledPanel)
            row_layout = QHBoxLayout(row)

            var_label = QLabel(var.name)
            var_label.setFixedWidth(150)
            var_label.setStyleSheet("font-weight: bold;")
            row_layout.addWidget(var_label)

            # Reusing the existing VariableConstraintWidget for display
            display_widget = VariableConstraintWidget(
                self.controller.model.variables, self.controller.priors
            )
            display_widget.set_constraint(var, restriction)
            display_widget.set_read_only(True)  # Make it read-only
            row_layout.addWidget(display_widget)

            self.mode_details_layout.addWidget(row)

    def show_prev_mode(self):
        if self.current_mode_index > 0:
            self.current_mode_index -= 1
            self.update_mode_display()

    def show_next_mode(self):
        if self.current_mode_index < len(self.modes) - 1:
            self.current_mode_index += 1
            self.update_mode_display()

    def refresh(self):
        """Called when a new model is loaded."""
        # Clear evidence rows
        for w in self.evidence_widgets:
            # The parent widget of VariableConstraintWidget is the row_widget
            w.parentWidget().deleteLater()
        self.evidence_widgets.clear()

        # Clear results
        self.clear_mode_details()
        self.modes = []
        self.mode_info_label.setText("Click Calculate to see results")
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

        # Add initial row
        self.add_evidence_row()
