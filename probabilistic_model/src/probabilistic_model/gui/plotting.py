from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from typing import Optional, List, Union, Tuple
import numpy as np
import plotly.colors

from PySide6.QtWidgets import QWidget, QVBoxLayout, QToolTip
from PySide6.QtCharts import (
    QChart,
    QChartView,
    QLineSeries,
    QPieSeries,
    QPieSlice,
    QValueAxis,
)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QColor, QCursor, QMouseEvent

from probabilistic_model.probabilistic_model import ProbabilisticModel
from random_events.variable import Symbolic, Continuous, Integer
from random_events.interval import SimpleInterval, Interval
from probabilistic_model.utils import neighbouring_points
from probabilistic_model.error import UndefinedOperationError
from .utils import (
    get_primary_color,
    get_secondary_light_color,
    get_primary_light_color,
    get_primary_text_color,
    get_secondary_text_color,
    is_dark_theme,
)
from probabilistic_model.constants import (
    PDF_TRACE_COLOR,
    CDF_TRACE_COLOR,
    MODE_TRACE_COLOR,
    EXPECTATION_TRACE_COLOR,
    PDF_TRACE_NAME,
    CDF_TRACE_NAME,
    MODE_TRACE_NAME,
    EXPECTATION_TRACE_NAME,
    SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT,
)


@dataclass
class InteractiveChartView(QChartView):
    """
    A custom QChartView that supports zooming and resetting.
    """

    parent: InitVar[Optional[QWidget]] = None
    """
    The parent widget.
    """

    def __post_init__(self, parent: Optional[QWidget]):
        super().__init__(parent)
        self.setRubberBand(QChartView.RectangleRubberBand)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.chart().zoomReset()
        else:
            super().mousePressEvent(event)

    def wheelEvent(self, event):
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.chart().zoom(factor)
        super().wheelEvent(event)


@dataclass
class ProbabilisticModelPlotWidget(QWidget):
    """
    A widget for plotting 1D probabilistic models using QtCharts.
    """

    parent: InitVar[Optional[QWidget]] = None
    """
    The parent widget.
    """

    layout: QVBoxLayout = field(init=False)
    """
    The main layout of the widget.
    """

    chart_view: InteractiveChartView = field(init=False)
    """
    The chart view that displays the plot.
    """

    def __post_init__(self, parent: Optional[QWidget]):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.chart_view = InteractiveChartView()
        self.chart_view.setStyleSheet("background: transparent;")
        self.layout.addWidget(self.chart_view)

    def on_line_hovered(self, point: QPointF, state: bool):
        """
        Shows a tooltip when a point on a line series is hovered.
        """
        if state:
            QToolTip.showText(
                QCursor.pos(), f"x: {point.x():.4f}\ny: {point.y():.4f}", self
            )
        else:
            QToolTip.hideText()

    def on_slice_hovered(self, slice: QPieSlice, state: bool):
        """
        Shows a tooltip when a pie slice is hovered.
        """
        if state:
            QToolTip.showText(
                QCursor.pos(),
                f"{slice.label()}\nProbability: {slice.percentage() * 100:.2f}% ({slice.value():.4f})",
                self,
            )
        else:
            QToolTip.hideText()

    def set_model(self, model: ProbabilisticModel, number_of_samples: int = 1000):
        """
        Sets the model to be plotted and updates the chart.

        :param model: The probabilistic model to plot.
        :param number_of_samples: The number of samples to use for numeric plots.
        """
        if len(model.variables) != 1:
            # For now, only 1D plots are supported
            return

        variable = model.variables[0]
        if variable.is_numeric:
            self.plot_1d_numeric(model, number_of_samples)
        else:
            self.plot_1d_symbolic(model)

    def plot_1d_symbolic(self, model: ProbabilisticModel):
        """
        Plots a 1D symbolic model using a pie chart.
        """
        variable: Symbolic = model.variables[0]
        chart = QChart()
        if is_dark_theme():
            chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        else:
            chart.setTheme(QChart.ChartTheme.ChartThemeLight)
        chart.setBackgroundVisible(False)
        chart.setTitle(f"Distribution of {variable.name}")

        # Calculate probabilities
        probabilities = {}
        for element in variable.domain:
            from random_events.product_algebra import SimpleEvent

            event = SimpleEvent({variable: element})
            probabilities[str(element)] = model.probability_of_simple_event(event)

        series = QPieSeries()
        max_prob = max(probabilities.values()) if probabilities else 0
        colors = plotly.colors.qualitative.Plotly
        primary_mode_color = get_primary_color(MODE_TRACE_COLOR).upper()

        # Create slices
        color_idx = 0
        for category, prob in probabilities.items():
            if prob > 0:
                slice = series.append(category, prob)

                if prob == max_prob:
                    slice.setExploded(True)
                    slice.setLabelVisible(True)
                    # For the mode, use the theme's mode color (consistent with other plots)
                    slice.setBrush(QColor(primary_mode_color))
                else:
                    # Assign unique color from palette, avoiding the mode color if possible
                    while colors[color_idx % len(colors)].upper() == primary_mode_color:
                        color_idx += 1
                    color_hex = colors[color_idx % len(colors)]
                    slice.setBrush(QColor(color_hex))
                    color_idx += 1

        series.hovered.connect(self.on_slice_hovered)
        chart.addSeries(series)

        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignRight)

        self.chart_view.setChart(chart)

    def plot_1d_numeric(self, model: ProbabilisticModel, number_of_samples: int):
        """
        Plots a 1D numeric model using a line chart.
        """
        variable = model.variables[0]
        chart = QChart()
        if is_dark_theme():
            chart.setTheme(QChart.ChartTheme.ChartThemeDark)
        else:
            chart.setTheme(QChart.ChartTheme.ChartThemeLight)
        chart.setBackgroundVisible(False)
        chart.setTitle(f"Distribution of {variable.name}")

        # Sample and prepare points
        samples = model.sample(number_of_samples)[:, 0]

        # Supporting interval
        supporting_interval: Interval = model.support.simple_sets[0][variable]

        # Border points
        for simple_interval in supporting_interval.simple_sets:
            lower, upper = simple_interval.lower, simple_interval.upper
            if lower > -np.inf:
                samples = np.concatenate((samples, neighbouring_points(lower)))
            if upper < np.inf:
                samples = np.concatenate((samples, neighbouring_points(upper)))

        samples = np.sort(samples)
        lowest = samples[0]
        highest = samples[-1]
        size = highest - lowest

        if size == 0:
            size = 1.0

        samples = np.concatenate(
            (
                np.array([lowest - size * 0.05]),
                samples,
                np.array([highest + size * 0.05]),
            )
        )

        # PDF Series
        pdf = model.likelihood(samples.reshape(-1, 1))
        pdf_series = QLineSeries()
        pdf_series.setName(PDF_TRACE_NAME)
        pdf_series.setPointsVisible(True)
        pdf_series.hovered.connect(self.on_line_hovered)
        for x, y in zip(samples, pdf):
            pdf_series.append(x, y)

        chart.addSeries(pdf_series)

        # CDF Series if available
        try:
            cdf = model.cdf(samples.reshape(-1, 1))
            cdf_series = QLineSeries()
            cdf_series.setName(CDF_TRACE_NAME)
            cdf_series.setPointsVisible(True)
            cdf_series.hovered.connect(self.on_line_hovered)
            for x, y in zip(samples, cdf):
                cdf_series.append(x, y)
            chart.addSeries(cdf_series)
        except (UndefinedOperationError, NotImplementedError):
            pass

        # Mode and Expectation
        try:
            mode_event, max_likelihood = model.mode()
        except Exception:
            max_likelihood = np.max(pdf)
            mode_event = None

        height = max_likelihood * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT

        # Expectation
        try:
            expectation = model.expectation([variable])[variable]
            expectation_series = QLineSeries()
            expectation_series.setName(EXPECTATION_TRACE_NAME)
            expectation_series.append(expectation, 0)
            expectation_series.append(expectation, height)
            expectation_series.setPointsVisible(True)
            expectation_series.hovered.connect(self.on_line_hovered)
            chart.addSeries(expectation_series)
        except (UndefinedOperationError, NotImplementedError, KeyError, Exception):
            pass

        # Mode
        if mode_event:
            first_mode = True
            for simple_event in mode_event.simple_sets:
                interval = simple_event[variable]
                for simple_interval in interval.simple_sets:
                    mode_series = QLineSeries()
                    if first_mode:
                        mode_series.setName(MODE_TRACE_NAME)
                        first_mode = False

                    mode_series.append(simple_interval.lower, 0)
                    mode_series.append(simple_interval.lower, height)
                    mode_series.append(simple_interval.upper, height)
                    mode_series.append(simple_interval.upper, 0)
                    mode_series.setPointsVisible(True)
                    mode_series.hovered.connect(self.on_line_hovered)
                    chart.addSeries(mode_series)

        # Axes
        axis_x = QValueAxis()
        axis_x.setTitleText(variable.name)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)

        axis_y = QValueAxis()
        axis_y.setTitleText("Density / Probability")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)

        # Colors
        primary = get_primary_color(PDF_TRACE_COLOR)
        primary_light = get_primary_light_color(CDF_TRACE_COLOR)
        primary_text = get_primary_text_color(EXPECTATION_TRACE_COLOR)

        for series in chart.series():
            name = series.name()
            if name == PDF_TRACE_NAME:
                series.setColor(QColor(primary))
            elif name == CDF_TRACE_NAME:
                series.setColor(QColor(primary_light))
            elif name == EXPECTATION_TRACE_NAME:
                series.setColor(QColor(primary_text))
            elif name == MODE_TRACE_NAME or not name:
                # Unnamed series are likely disjoint mode pieces
                series.setColor(QColor(primary))

        self.chart_view.setChart(chart)
