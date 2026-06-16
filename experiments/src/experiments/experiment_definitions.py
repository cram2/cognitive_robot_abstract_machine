from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List


from krrood.class_diagrams.attribute_introspector import (
    DataclassOnlyIntrospector,
    AttributeIntrospector,
    DiscoveredAttribute,
)


@dataclass
class MeanAndStandardDeviation:
    """
    Class that represents a mean and standard deviation for tables that will be directly rendered as
    mean +- standard deviation.

    Use this in experiment results when you want to render a mean +- standard deviation.
    """

    mean: float
    """
    The mean of the measurements.
    """

    standard_deviation: float
    """
    The standard deviation of the measurements.
    """

    def __str__(self) -> str:
        return f"{round(self.mean, 2)} ± {round(self.standard_deviation, 2)}"

    @classmethod
    def from_measurements(cls, measurements: List[float]) -> MeanAndStandardDeviation:
        std = 0.0 if len(measurements) < 2 else statistics.stdev(measurements)
        return cls(
            mean=round(statistics.mean(measurements), 2),
            standard_deviation=round(std, 4),
        )


@dataclass
class ExperimentResult:
    """
    Class for results from experiments.
    Use this when you want to create a table of results (measurements) from experiments for scientific articles.

    This class is like a single row in a table of experiments.

    Assumptions made here are that there are only built in like fields or one-to-one relationships with other
    ExperimentResult classes.
    """

    @classmethod
    def introspector(cls) -> AttributeIntrospector:
        return DataclassOnlyIntrospector()

    @classmethod
    def recursive_fields(cls) -> List[DiscoveredAttribute]:
        result = []
        for field_ in cls.introspector().discover(cls):
            if issubclass(field_.field.type, ExperimentResult):
                result.extend(field_.field.type.recursive_fields())
            else:
                result.append(field_)
        return result

    @classmethod
    def get_column_names(cls) -> list[str]:
        return [field_.field.name for field_ in cls.recursive_fields()]

    def get_column_values(self) -> list[str]:
        return [getattr(self, field_.field.name) for field_ in self.recursive_fields()]


@dataclass
class ExperimentsTable:
    """
    A collection of experiments ready to be presented as a table in a scientific article.

    This class assumes that all rows in the table have the same type and are a subclass of ExperimentResult.
    """

    experiments: list[ExperimentResult]
    """
    The list of experiments to be presented in the table.
    """

    def __post_init__(self):
        if not self.experiments:
            return
        row_types = {type(row) for row in self.experiments}
        assert len(row_types) == 1 and issubclass(
            list(row_types)[0], ExperimentResult
        ), "Tables can only be constructed over rows that have the same type everywhere."

    @property
    def row_class(self) -> type[ExperimentResult] | None:
        if not self.experiments:
            return None
        return type(self.experiments[0])


@dataclass
class TypstRenderer:
    """
    Represents a renderer for converting an ExperimentsTable into Typst markup.
    """

    experiments_table: ExperimentsTable
    """
    The experiments to render.
    """

    def render_row(self, row: ExperimentResult) -> str:
        """Renders the cells of a single row in Typst format."""
        return ", ".join([f"[{v}]" for v in row.get_column_values()])

    def render_table(self) -> str:
        """Renders the entire ExperimentsTable into a valid Typst #table markup string."""
        row_class = self.experiments_table.row_class

        # Handle empty table edge-case gracefully
        if not row_class:
            return "#table()"

        # 1. Extract headers and setup column configuration
        headers = row_class.get_column_names()
        columns_count = len(headers)

        # 2. Build the Typst header block
        header_cells = ", ".join(
            [f"[*{name.replace('_', ' ').title()}*]" for name in headers]
        )

        # 3. Build the rows content
        rows_content = []
        for row in self.experiments_table.experiments:
            rows_content.append(self.render_row(row))

        all_cells = header_cells
        if rows_content:
            all_cells += ",\n  " + ",\n  ".join(rows_content)

        # 4. Construct complete Typst syntax block
        typst_markup = (
            f"#table(\n"
            f"  columns: {columns_count},\n"
            f"  align: center + horizon,\n"
            f"  {all_cells}\n"
            f")"
        )
        return typst_markup
