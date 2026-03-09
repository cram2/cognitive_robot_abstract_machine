from dataclasses import dataclass, field, InitVar
from typing import Optional, List, Union, Tuple, Set as TypingSet
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QComboBox,
    QListWidget,
    QAbstractItemView,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QScrollArea,
    QFrame,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QIcon
from .utils import (
    get_primary_color,
    get_secondary_dark_color,
    get_secondary_light_color,
)
from superqt import QRangeSlider, QDoubleRangeSlider

from random_events.variable import Variable, Continuous, Symbolic, Integer
from random_events.product_algebra import SimpleEvent, Event, VariableMap
from random_events.interval import closed, Interval, SimpleInterval, Bound
from random_events.set import Set, SetElement


@dataclass(init=False, eq=False, repr=False)
class NumericIntervalWidget(QWidget):
    """
    A widget representing a single interval for a numeric variable.
    Includes spin boxes for precise input and a range slider for visual adjustment.
    """

    variable: Variable
    """
    The variable this widget represents.
    """

    minimum_bound: float
    """
    The minimum value allowed for the variable.
    """

    maximum_bound: float
    """
    The maximum value allowed for the variable.
    """

    initial_values: Tuple[float, float]
    """
    The initial values for the lower and upper bounds of the interval.
    """

    minimum_spin_box: Union[QDoubleSpinBox, QSpinBox]
    """
    The spin box for the minimum interval value.
    """

    maximum_spin_box: Union[QDoubleSpinBox, QSpinBox]
    """
    The spin box for the maximum interval value.
    """

    slider: Union[QDoubleRangeSlider, QRangeSlider]
    """
    The range slider for visual adjustment of the interval.
    """

    remove_button: QPushButton
    """
    The button to remove this interval row.
    """

    changed = Signal()
    removed = Signal()

    def __init__(
        self,
        variable: Variable,
        minimum_bound: float,
        maximum_bound: float,
        initial_values: Tuple[float, float],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.variable = variable
        self.minimum_bound = float(minimum_bound)
        self.maximum_bound = float(maximum_bound)
        self.initial_values = initial_values
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if isinstance(self.variable, Continuous):
            self.minimum_spin_box = QDoubleSpinBox()
            self.maximum_spin_box = QDoubleSpinBox()
            self.slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
            for spin_box in [self.minimum_spin_box, self.maximum_spin_box]:
                spin_box.setRange(self.minimum_bound, self.maximum_bound)
                spin_box.setDecimals(3)
                spin_box.setSingleStep(
                    (self.maximum_bound - self.minimum_bound) / 100.0
                )
        else:
            self.minimum_spin_box = QSpinBox()
            self.maximum_spin_box = QSpinBox()
            self.slider = QRangeSlider(Qt.Orientation.Horizontal)
            for spin_box in [self.minimum_spin_box, self.maximum_spin_box]:
                spin_box.setRange(int(self.minimum_bound), int(self.maximum_bound))

        self.slider.setRange(self.minimum_bound, self.maximum_bound)
        self.slider.setValue(self.initial_values)
        self.minimum_spin_box.setValue(self.initial_values[0])
        self.maximum_spin_box.setValue(self.initial_values[1])

        primary = get_primary_color()
        secondary_light = get_secondary_light_color()

        # Enhanced stylesheet for slider visibility
        self.setStyleSheet(
            f"NumericIntervalWidget {{ "
            f"   min-height: 40px; "
            f"}} "
            f"QRangeSlider, QDoubleRangeSlider {{ "
            f"   min-height: 24px; "
            f"   background: none; "
            f"   border: none; "
            f"}} "
            f"QRangeSlider::handle, QDoubleRangeSlider::handle {{ "
            f"   background-color: {primary}; "
            f"   border: 1px solid {primary}; "
            f"   width: 14px; "
            f"   height: 14px; "
            f"   margin: -4px 0; "
            f"   border-radius: 8px; "
            f"}} "
            f"QRangeSlider::handle:hover, QDoubleRangeSlider::handle:hover {{ "
            f"   background-color: white; "
            f"   border: 1px solid {primary}; "
            f"}} "
            f"QRangeSlider::groove, QDoubleRangeSlider::groove {{ "
            f"   background-color: {secondary_light}; "
            f"   height: 4px; "
            f"   border-radius: 2px; "
            f"}} "
            f"QRangeSlider::bar, QDoubleRangeSlider::bar {{ "
            f"   background-color: {primary}; "
            f"}} "
        )

        # Connections
        self.slider.sliderMoved.connect(self._on_slider_changed)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.minimum_spin_box.valueChanged.connect(self._on_spin_changed)
        self.maximum_spin_box.valueChanged.connect(self._on_spin_changed)

        self.remove_button = QPushButton()
        self.remove_button.setIcon(QIcon("icon:/primary/close.svg"))
        self.remove_button.setFixedWidth(30)
        self.remove_button.clicked.connect(self.removed.emit)

        layout.addWidget(self.minimum_spin_box)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.maximum_spin_box)
        layout.addWidget(self.remove_button)

    def _on_slider_changed(self, values: Tuple[float, float]):
        """
        Updates the spin boxes when the slider values change.
        """
        self.minimum_spin_box.setValue(values[0])
        self.maximum_spin_box.setValue(values[1])
        self.changed.emit()

    def _on_slider_pressed(self):
        """
        Blocks signals from spin boxes when the slider is pressed.
        """
        self.minimum_spin_box.blockSignals(True)
        self.maximum_spin_box.blockSignals(True)

    def _on_slider_released(self):
        """
        Unblocks signals from spin boxes and emits changed when the slider is released.
        """
        self.minimum_spin_box.blockSignals(False)
        self.maximum_spin_box.blockSignals(False)
        self.changed.emit()

    def _on_spin_changed(self):
        """
        Updates the slider when the spin box values change.
        """
        if self.minimum_spin_box.signalsBlocked():
            return

        value_1 = self.minimum_spin_box.value()
        value_2 = self.maximum_spin_box.value()
        low, high = min(value_1, value_2), max(value_1, value_2)
        self.slider.blockSignals(True)
        self.slider.setValue((low, high))
        self.slider.blockSignals(False)
        self.changed.emit()

    def value(self) -> Tuple[float, float]:
        """
        Returns the current low and high values of the interval.
        """
        return (self.minimum_spin_box.value(), self.maximum_spin_box.value())


@dataclass(init=False, eq=False, repr=False)
class VariableConstraintWidget(QFrame):
    """
    A widget that allows selecting a variable and defining its constraints.
    """

    variables: List[Variable]
    """
    The list of variables to select from.
    """

    priors: Optional[VariableMap]
    """
    The prior distributions for the variables (optional).
    """

    main_layout: QVBoxLayout
    """
    The main layout of the widget.
    """

    variable_combo: QComboBox
    """
    The combo box to select a variable.
    """

    remove_button: QPushButton
    """
    The button to remove this variable constraint widget.
    """

    constraint_container: QWidget
    """
    The container widget for constraints.
    """

    constraint_layout: QVBoxLayout
    """
    The layout for the constraint container.
    """

    constraint_widget: Optional[Union[QScrollArea, QListWidget]]
    """
    The widget used for defining constraints (QScrollArea or QListWidget).
    """

    intervals_container: QWidget
    """
    The container for numeric interval rows.
    """

    intervals_layout: QVBoxLayout
    """
    The layout for the numeric intervals container.
    """

    add_range_button: QPushButton
    """
    The button to add a new numeric interval.
    """

    scroll_area: QScrollArea
    """
    The scroll area for numeric intervals.
    """

    interval_widgets: List[NumericIntervalWidget]
    """
    The list of numeric interval widgets.
    """

    value_label: QLabel
    """
    The label displaying the range of the variable.
    """

    changed = Signal()
    removed = Signal()

    _read_only: bool = False

    def __init__(
        self,
        variables: List[Variable],
        priors: Optional[VariableMap] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.variables = sorted(variables, key=lambda v: v.name)
        self.priors = priors
        self.init_ui()

    def init_ui(self):
        primary = get_primary_color()
        secondary_dark = get_secondary_dark_color()

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setStyleSheet(
            f"VariableConstraintWidget {{ "
            f"   border: 1px solid {primary}; "
            f"   border-radius: 8px; "
            f"   background-color: {secondary_dark}; "
            f"   margin-bottom: 10px; "
            f"}} "
        )

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Header Row: [Combo] [Stretch] [Remove Button]
        header_layout = QHBoxLayout()

        # Variable selector
        self.variable_combo = QComboBox()
        self.variable_combo.addItem("Select Variable...", None)
        for var in self.variables:
            self.variable_combo.addItem(var.name, var)

        self.variable_combo.currentIndexChanged.connect(self.on_variable_changed)
        header_layout.addWidget(self.variable_combo, 1)

        self.value_label = QLabel("")
        self.value_label.setStyleSheet("color: #888888; margin-left: 10px;")
        header_layout.addWidget(self.value_label)

        header_layout.addStretch()

        # Row removal button
        self.remove_button = QPushButton()
        self.remove_button.setIcon(QIcon("icon:/primary/close.svg"))
        self.remove_button.setFixedWidth(30)
        self.remove_button.clicked.connect(self.removed.emit)
        header_layout.addWidget(self.remove_button)

        self.main_layout.addLayout(header_layout)

        # Container for constraint input
        self.constraint_container = QWidget()
        self.constraint_layout = QVBoxLayout(self.constraint_container)
        self.constraint_layout.setContentsMargins(0, 0, 0, 0)
        # Avoid unnecessary re-layouts during heavy widget manipulation
        self.constraint_layout.setEnabled(False)
        self.main_layout.addWidget(self.constraint_container)
        self.constraint_layout.setEnabled(True)

        self.constraint_widget = None

    def on_variable_changed(self):
        # Clear previous constraint widgets
        self.constraint_layout.setEnabled(False)
        try:
            while self.constraint_layout.count():
                item = self.constraint_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    # Disconnect all signals before hiding/deleting to avoid issues during teardown
                    widget.blockSignals(True)
                    widget.hide()
                    widget.deleteLater()
                else:
                    # Could be a layout
                    sub_layout = item.layout()
                    if sub_layout:
                        # Clear sub_layout
                        while sub_layout.count():
                            sub_item = sub_layout.takeAt(0)
                            sub_widget = sub_item.widget()
                            if sub_widget:
                                sub_widget.blockSignals(True)
                                sub_widget.hide()
                                sub_widget.deleteLater()
            self.constraint_widget = None

            variable = self.variable_combo.currentData()
            if variable:
                self.create_constraint_widget(variable)

            if self._read_only:
                self.set_read_only(True)
        finally:
            self.constraint_layout.setEnabled(True)

        self.changed.emit()

    def update_available_variables(self, excluded_variables: TypingSet[Variable]):
        """
        Disables variables in the combo box that are in the excluded_variables set,
        except for the currently selected variable.

        :param excluded_variables: A set of variables that are selected elsewhere.
        """
        current_var = self.variable_combo.currentData()
        model = self.variable_combo.model()
        for i in range(1, self.variable_combo.count()):
            var = self.variable_combo.itemData(i)
            # Disable if selected elsewhere, but NOT if it is the current selection of THIS widget
            is_currently_selected = current_var is not None and var == current_var
            should_disable = var in excluded_variables and not is_currently_selected

            # QComboBox uses QStandardItemModel by default
            item = model.item(i)
            if item:
                item.setEnabled(not should_disable)

    def create_constraint_widget(self, variable: Variable):
        if variable.is_numeric:
            # Container for interval rows
            self.intervals_container = QWidget()
            self.intervals_layout = QVBoxLayout(self.intervals_container)
            self.intervals_layout.setContentsMargins(0, 0, 0, 0)
            self.intervals_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            # Header Row: [Add Button]
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)

            header_layout.addStretch()

            self.add_range_button = QPushButton()
            self.add_range_button.setIcon(QIcon("icon:/primary/checklist.svg"))
            self.add_range_button.setFixedWidth(30)
            self.add_range_button.clicked.connect(
                lambda: self.add_numeric_interval(variable)
            )
            header_layout.addWidget(self.add_range_button)

            self.constraint_layout.addWidget(header_widget)

            # Scroll Area for intervals
            self.scroll_area = QScrollArea()
            self.scroll_area.setWidgetResizable(True)
            self.scroll_area.setWidget(self.intervals_container)
            self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
            self.scroll_area.setMinimumHeight(150)

            self.constraint_widget = self.scroll_area

            self.interval_widgets: List[NumericIntervalWidget] = []

            # Determine bounds
            minimum, maximum = self._get_variable_bounds(variable)
            self.slider_min, self.slider_max = minimum, maximum

            if self.value_label:
                self.value_label.setText(f"Range: [{minimum:.2f}, {maximum:.2f}]")

            # Add marks (for testing purposes and clarity)
            marks_layout = QHBoxLayout()
            marks_layout.setContentsMargins(50, 0, 50, 0)  # Align with slider roughly
            for i in range(5):
                val = minimum + (maximum - minimum) * (i / 4.0)
                mark = QLabel(f"{val:.2f}")
                mark.setStyleSheet("font-size: 8pt; color: #666666;")
                marks_layout.addWidget(mark)
                if i < 4:
                    marks_layout.addStretch()
            self.constraint_layout.addLayout(marks_layout)

            # Add initial interval
            self.add_numeric_interval(variable, (minimum, maximum))
        else:
            # List Widget for Symbolic (multi-selection)
            if self.value_label:
                self.value_label.setText("")
            list_widget = QListWidget()
            list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

            # variable.domain is a Set for Symbolic variables
            for element in variable.domain.all_elements:
                list_widget.addItem(str(element))
                item = list_widget.item(list_widget.count() - 1)
                item.setSelected(True)  # Default select all

            list_widget.itemSelectionChanged.connect(self.changed.emit)
            # Set a reasonable height for the list
            list_widget.setMaximumHeight(100)
            self.constraint_widget = list_widget

        self.constraint_layout.addWidget(self.constraint_widget)

    def _get_variable_bounds(self, variable: Variable) -> Tuple[float, float]:
        """
        Calculates the minimum and maximum bounds for the variable based on its domain and priors.
        """
        # Default to domain
        try:
            minimum = variable.domain.simple_sets[0].lower
            maximum = variable.domain.simple_sets[-1].upper
        except (AttributeError, IndexError):
            # Fallback for composite sets
            try:
                minimum = variable.domain.simple_sets[0].simple_sets[0].lower
                maximum = variable.domain.simple_sets[-1].simple_sets[-1].upper
            except (AttributeError, IndexError):
                minimum, maximum = 0.0, 1.0

        # Try to get from priors (support)
        if self.priors and variable in self.priors:
            try:
                support = self.priors[variable].support
                if support.simple_sets:
                    minimum = support.simple_sets[0][variable].simple_sets[0].lower
                    maximum = support.simple_sets[-1][variable].simple_sets[-1].upper
            except Exception:
                pass

        # Handle infinity
        if minimum == float("-inf"):
            minimum = -100.0
        if maximum == float("inf"):
            maximum = 100.0

        # Handle equality
        if minimum == maximum:
            minimum -= 1.0
            maximum += 1.0

        return minimum, maximum

    def add_numeric_interval(
        self, variable: Variable, values: Optional[Tuple[float, float]] = None
    ):
        """
        Adds a new numeric interval row.
        """
        if values is None:
            # Default new interval values based on current last interval
            if self.interval_widgets:
                _, last_high = self.interval_widgets[-1].value()
                remaining = self.slider_max - last_high
                if remaining > (self.slider_max - self.slider_min) * 0.1:
                    values = (
                        last_high + remaining * 0.05,
                        last_high + remaining * 0.15,
                    )
                else:
                    # Add at the end if not enough space
                    values = (
                        self.slider_max - (self.slider_max - self.slider_min) * 0.1,
                        self.slider_max,
                    )
            else:
                values = (self.slider_min, self.slider_max)

        interval_widget = NumericIntervalWidget(
            variable, self.slider_min, self.slider_max, values
        )
        interval_widget.changed.connect(self.changed.emit)
        interval_widget.removed.connect(
            lambda: self.remove_numeric_interval(interval_widget)
        )

        self.interval_widgets.append(interval_widget)
        self.intervals_layout.addWidget(interval_widget)
        self.changed.emit()

    def remove_numeric_interval(self, widget: NumericIntervalWidget):
        """
        Removes a numeric interval row.
        """
        if len(self.interval_widgets) > 1:
            self.interval_widgets.remove(widget)
            widget.deleteLater()
            self.changed.emit()

    def set_read_only(self, read_only: bool):
        """
        Sets the widget to read-only mode.
        In read-only mode, the user cannot change the variable or its constraints,
        but can still view them (and scroll if necessary).
        """
        self._read_only = read_only
        self.variable_combo.setEnabled(not read_only)
        self.remove_button.setVisible(not read_only)

        if self.constraint_widget:
            if isinstance(self.constraint_widget, QListWidget):
                # For Symbolic variables: keep scrollable but disable selection change
                # We do NOT change selection mode to NoSelection, as it would hide selection.
                # Instead, we make items non-selectable.
                for i in range(self.constraint_widget.count()):
                    item = self.constraint_widget.item(i)
                    if read_only:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                    else:
                        item.setFlags(
                            item.flags()
                            | Qt.ItemFlag.ItemIsSelectable
                            | Qt.ItemFlag.ItemIsEnabled
                        )
            elif isinstance(self.constraint_widget, QScrollArea):
                # For Numeric variables
                if hasattr(self, "add_range_button"):
                    self.add_range_button.setVisible(not read_only)

                # Each NumericIntervalWidget needs to be set read-only
                for widget in self.interval_widgets:
                    widget.minimum_spin_box.setReadOnly(read_only)
                    widget.maximum_spin_box.setReadOnly(read_only)
                    widget.slider.setEnabled(not read_only)
                    widget.remove_button.setVisible(not read_only)

    def get_constraint(self) -> Optional[Tuple[Variable, Union[Interval, Set]]]:
        variable = self.variable_combo.currentData()
        if not variable:
            return None

        if isinstance(variable, (Continuous, Integer)):
            intervals = []
            for widget in self.interval_widgets:
                values = widget.value()
                low, high = min(values), max(values)
                intervals.append(SimpleInterval(low, high, Bound.CLOSED, Bound.CLOSED))

            if not intervals:
                return variable, variable.domain

            return variable, Interval(*intervals)
        elif isinstance(variable, Symbolic):
            list_widget: QListWidget = self.constraint_widget
            selected_items = list_widget.selectedItems()
            selected_values = [item.text() for item in selected_items]
            # Create a Set from selected values
            # Need to be careful with types here, selected_values are strings
            # and the original domain might have different types.
            # For now, assuming strings or using match.
            all_elements = variable.domain.all_elements
            matched_elements = [e for e in all_elements if str(e) in selected_values]

            if not matched_elements:
                return variable, Set()  # Empty set

            return variable, Set(
                *[SetElement(e, all_elements) for e in matched_elements]
            )

        return None

    def set_constraint(self, variable: Variable, constraint: Union[Interval, Set]):
        """
        Sets the variable and its constraint programmatically.
        """
        # Select the variable in the combo box
        for i in range(self.variable_combo.count()):
            data = self.variable_combo.itemData(i)
            if data is not None and data == variable:
                self.variable_combo.setCurrentIndex(i)
                break

        # Now the constraint widget should be created via on_variable_changed
        if isinstance(variable, (Continuous, Integer)):
            # Clear existing interval widgets
            for widget in self.interval_widgets:
                widget.deleteLater()
            self.interval_widgets.clear()

            # constraint is an Interval (composite set)
            # Sort simple sets by lower bound
            sorted_simple_sets = sorted(constraint.simple_sets, key=lambda s: s.lower)
            for simple_set in sorted_simple_sets:
                self.add_numeric_interval(
                    variable, (simple_set.lower, simple_set.upper)
                )

        elif isinstance(variable, Symbolic):
            if not self.constraint_widget:
                return
            list_widget: QListWidget = self.constraint_widget
            # constraint is a Set
            selected_str_values = [str(e.element) for e in constraint.simple_sets]
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                item.setSelected(item.text() in selected_str_values)
