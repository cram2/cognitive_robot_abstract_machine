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
    QListWidget,
    QAbstractItemView,
)
from .plotting import ProbabilisticModelPlotWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from .controller import ModelController
from .variable_constraint_widget import VariableConstraintWidget
from random_events.product_algebra import SimpleEvent, Event
from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class PosteriorWidget(QWidget):
    """
    The Posterior page widget of the GUI.
    Allows calculating and viewing posterior distributions given evidence.
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

    current_posterior_model: Optional[ProbabilisticModel] = field(
        default=None, init=False, repr=False, compare=False
    )
    """
    The current posterior model after calculation.
    """

    query_vars: List[str] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    """
    The list of selected variables to plot.
    """

    current_var_index: int = field(default=0, init=False, repr=False, compare=False)
    """
    The index of the currently plotted variable.
    """

    calculate_button: QPushButton = field(init=False, repr=False, compare=False)
    """
    The button to calculate the posterior.
    """

    query_vars_list: QListWidget = field(init=False, repr=False, compare=False)
    """
    The list widget to select query variables.
    """

    evidence_container: QWidget = field(init=False, repr=False, compare=False)
    """
    The container for evidence variable rows.
    """

    evidence_layout: QVBoxLayout = field(init=False, repr=False, compare=False)
    """
    The layout for evidence variable rows.
    """

    prev_button: QPushButton = field(init=False, repr=False, compare=False)
    """
    The button to show the previous plot.
    """

    next_button: QPushButton = field(init=False, repr=False, compare=False)
    """
    The button to show the next plot.
    """

    plot_container: QVBoxLayout = field(init=False, repr=False, compare=False)
    """
    The layout container for the plot and title.
    """

    plot_title: QLabel = field(init=False, repr=False, compare=False)
    """
    The label displaying the plot title.
    """

    plot_widget: ProbabilisticModelPlotWidget = field(
        init=False, repr=False, compare=False
    )
    """
    The widget that renders the distribution plots.
    """

    def __post_init__(self, parent: Optional[QWidget]):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Content Area
        content_layout = QHBoxLayout()

        # Query Variables Section
        query_vars_section = self.create_query_vars_section()
        content_layout.addWidget(query_vars_section, 1)

        # Vertical separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        content_layout.addWidget(line)

        # Evidence Section
        evidence_section = self.create_evidence_section()
        content_layout.addWidget(evidence_section, 1)

        main_layout.addLayout(content_layout)

        # Calculate Button
        self.calculate_button = QPushButton("Calculate Posterior")
        self.calculate_button.clicked.connect(self.on_calculate)
        main_layout.addWidget(self.calculate_button)

        # Result Area
        result_area = QHBoxLayout()

        # Navigation Buttons
        self.prev_button = QPushButton()
        self.prev_button.setIcon(QIcon("icon:/primary/leftarrow.svg"))
        self.prev_button.setFixedWidth(40)
        self.prev_button.clicked.connect(self.show_previous)
        self.prev_button.setEnabled(False)
        result_area.addWidget(self.prev_button)

        # Plot Area
        self.plot_container = QVBoxLayout()
        self.plot_title = QLabel("")
        self.plot_title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.plot_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_container.addWidget(self.plot_title)

        self.plot_widget = ProbabilisticModelPlotWidget()
        self.plot_widget.setMinimumHeight(400)
        self.plot_container.addWidget(self.plot_widget)

        result_area.addLayout(self.plot_container, 4)

        self.next_button = QPushButton()
        self.next_button.setIcon(QIcon("icon:/primary/rightarrow.svg"))
        self.next_button.setFixedWidth(40)
        self.next_button.clicked.connect(self.show_next)
        self.next_button.setEnabled(False)
        result_area.addWidget(self.next_button)

        main_layout.addLayout(result_area, 2)

    def create_query_vars_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)

        header_label = QLabel("Query Variables")
        header_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(header_label)

        self.query_vars_list = QListWidget()
        self.query_vars_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        if self.controller.model:
            for var in sorted(self.controller.model.variables, key=lambda v: v.name):
                self.query_vars_list.addItem(var.name)

        layout.addWidget(self.query_vars_list)
        return section

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

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.evidence_container = QWidget()
        self.evidence_layout = QVBoxLayout(self.evidence_container)
        self.evidence_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_area.setWidget(self.evidence_container)

        layout.addWidget(scroll_area)

        # Add initial row
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
        self.evidence_layout.addWidget(var_widget)
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

    def on_calculate(self):
        if not self.controller.model:
            return

        evidence_event = self.build_event(self.evidence_widgets)
        self.current_posterior_model = self.controller.calculate_posterior(
            evidence_event
        )

        selected_items = self.query_vars_list.selectedItems()
        self.query_vars = [item.text() for item in selected_items]

        if not self.query_vars:
            # Default to all if none selected? Or show nothing?
            # Original app seems to default to all if nothing selected in dropdown.
            pass

        self.current_var_index = 0
        self.update_plot()

    def build_event(self, widgets_list: List[VariableConstraintWidget]) -> Event:
        simple_event = SimpleEvent()
        for widget in widgets_list:
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

        return Event.from_simple_sets(simple_event)

    def update_plot(self):
        if not self.current_posterior_model or not self.query_vars:
            self.plot_title.setText("")
            # Maybe show a placeholder or clear the chart
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        var_name = self.query_vars[self.current_var_index]
        self.plot_title.setText(f"Posterior Distribution of {var_name}")

        variable = self.controller.variable_map[var_name]

        try:
            # Calculate marginal for the specific variable
            marginal = self.current_posterior_model.marginal([variable])
            self.plot_widget.set_model(marginal)

        except Exception as e:
            print(f"Error generating plot: {e}")
            import traceback

            traceback.print_exc()

        self.prev_button.setEnabled(self.current_var_index > 0)
        self.next_button.setEnabled(self.current_var_index < len(self.query_vars) - 1)

    def show_previous(self):
        if self.current_var_index > 0:
            self.current_var_index -= 1
            self.update_plot()

    def show_next(self):
        if self.current_var_index < len(self.query_vars) - 1:
            self.current_var_index += 1
            self.update_plot()

    def refresh(self):
        """Called when a new model is loaded."""
        if not self.controller.model:
            return

        # Refresh variable list
        self.query_vars_list.clear()
        for var in sorted(self.controller.model.variables, key=lambda v: v.name):
            self.query_vars_list.addItem(var.name)

        # Clear evidence rows
        for w in self.evidence_widgets:
            # The widget is inside a row_widget which is inside evidence_layout
            w.parentWidget().deleteLater()
        self.evidence_widgets.clear()
        self.add_evidence_row()

        self.current_posterior_model = None
        self.query_vars = []
        self.update_plot()
