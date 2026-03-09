from dataclasses import dataclass, field, InitVar
from typing import List, Optional
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
class QueryWidget(QWidget):
    """
    The Query page widget of the GUI.
    Allows constructing queries and evidence to calculate probabilities.
    """

    controller: ModelController
    """
    The model controller.
    """

    parent: InitVar[Optional[QWidget]] = None
    """
    The parent widget.
    """

    query_widgets: List[VariableConstraintWidget] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    """
    The list of variable constraint widgets for the query.
    """

    evidence_widgets: List[VariableConstraintWidget] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    """
    The list of variable constraint widgets for the evidence.
    """

    calculate_button: QPushButton = field(init=False, repr=False, compare=False)
    """
    The button to calculate the probability.
    """

    result_label: QLabel = field(init=False, repr=False, compare=False)
    """
    The label to display the calculation result.
    """

    def __post_init__(self, parent: Optional[QWidget]):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Content Area
        content_layout = QHBoxLayout()

        # Query Section
        query_section = self.create_section("Query", self.query_widgets)
        content_layout.addWidget(query_section)

        # Vertical separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        content_layout.addWidget(line)

        # Evidence Section
        evidence_section = self.create_section("Evidence", self.evidence_widgets)
        content_layout.addWidget(evidence_section)

        main_layout.addLayout(content_layout)

        # Result Area
        result_layout = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate Probability")
        self.calculate_button.clicked.connect(self.on_calculate)
        result_layout.addWidget(self.calculate_button)

        self.result_label = QLabel("P(Q | E) = ...")
        self.result_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        result_layout.addWidget(self.result_label)

        main_layout.addLayout(result_layout)

    def create_section(
        self, title: str, widgets_list: List[VariableConstraintWidget]
    ) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)

        header_layout = QHBoxLayout()
        header_label = QLabel(title)
        header_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        header_layout.addWidget(header_label)

        add_button = QPushButton()
        add_button.setIcon(QIcon("icon:/primary/checklist.svg"))
        add_button.setFixedWidth(30)
        add_button.clicked.connect(
            lambda: self.add_variable_row(widgets_list, layout_container)
        )
        header_layout.addWidget(add_button)

        layout.addLayout(header_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        layout_container = QWidget()
        self.rows_layout = QVBoxLayout(layout_container)
        self.rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_area.setWidget(layout_container)

        layout.addWidget(scroll_area)

        # Add initial row
        self.add_variable_row(widgets_list, layout_container)

        return section

    def add_variable_row(
        self, widgets_list: List[VariableConstraintWidget], container: QWidget
    ):
        if not self.controller.model:
            return

        var_widget = VariableConstraintWidget(
            self.controller.model.variables, self.controller.priors
        )
        var_widget.changed.connect(lambda: self.update_variable_options(widgets_list))
        var_widget.changed.connect(
            self.on_calculate
        )  # Auto-calculate or at least clear result
        var_widget.removed.connect(
            lambda variable_widget=var_widget: self.remove_variable_row(
                variable_widget, variable_widget, widgets_list
            )
        )
        widgets_list.append(var_widget)
        container.layout().addWidget(var_widget)
        self.update_variable_options(widgets_list)

    def remove_variable_row(
        self,
        row_widget: QWidget,
        var_widget: VariableConstraintWidget,
        widgets_list: List[VariableConstraintWidget],
    ):
        widgets_list.remove(var_widget)
        row_widget.deleteLater()
        self.update_variable_options(widgets_list)
        self.on_calculate()

    def update_variable_options(self, widgets_list: List[VariableConstraintWidget]):
        """
        Updates the available variables in all widgets of the list.
        """
        selected_vars = {
            w.variable_combo.currentData()
            for w in widgets_list
            if w.variable_combo.currentData() is not None
        }
        for w in widgets_list:
            w.update_available_variables(selected_vars)

    def on_calculate(self):
        if not self.controller.model:
            self.result_label.setText("No model loaded")
            return

        query_event = self.build_event(self.query_widgets)
        evidence_event = self.build_event(self.evidence_widgets)

        if not query_event.simple_sets:
            self.result_label.setText("P(Q | E) = ...")
            return

        prob = self.controller.calculate_probability(query_event, evidence_event)
        self.result_label.setText(f"P(Q | E) = {prob:.5f}")

    def build_event(self, widgets_list: List[VariableConstraintWidget]) -> Event:
        simple_event = SimpleEvent()
        for widget in widgets_list:
            constraint = widget.get_constraint()
            if constraint:
                var, val = constraint
                # SimpleEvent in random_events can only have one constraint per variable.
                # If multiple rows have same variable, they should probably be intersected.
                # For now, following Dash's simple approach:
                # (actually Dash might handle multiple rows as intersection or union depending on logic)
                # In components.py div_to_event intersections were used.
                if var in simple_event:
                    # Treat multiple constraints on the same variable as a union
                    simple_event[var] = simple_event[var].union_with(val)
                else:
                    simple_event[var] = val

        if not simple_event:
            return Event()

        return Event(simple_event)

    def refresh(self):
        """Called when a new model is loaded."""
        # Clear all rows and re-initialize
        for w in self.query_widgets:
            w.parentWidget().deleteLater()
        self.query_widgets.clear()

        for w in self.evidence_widgets:
            w.parentWidget().deleteLater()
        self.evidence_widgets.clear()

        # The UI layout needs to be re-populated or we can just re-init the whole thing.
        # For simplicity, let's just make sure we can add a row.
        # Actually, maybe the MainWindow should recreate QueryWidget.
