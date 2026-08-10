from __future__ import annotations

import enum
import json
import pathlib
import statistics
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum

from typing_extensions import Any

from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.exceptions import DataclassException
from semantic_digital_twin.world_description.geometry import Bounds


from krrood.class_diagrams.attribute_introspector import (
    DataclassOnlyIntrospector,
    AttributeIntrospector,
    DiscoveredAttribute,
)


class Unit(StrEnum):
    """
    The physical unit a :class:`MeanAndStandardDeviation` is expressed in.
    """

    NONE = ""
    SECONDS = "s"
    MILLISECONDS = "ms"


_SECONDS_PER_UNIT: dict[Unit, float] = {
    Unit.SECONDS: 1.0,
    Unit.MILLISECONDS: 0.001,
}


@dataclass
class IncompatibleUnitConversionError(DataclassException):
    """
    Raised when a :class:`MeanAndStandardDeviation` is converted to or from a unit that
    has no known conversion factor.
    """

    source_unit: Unit
    """
    The unit the value was expressed in.
    """

    target_unit: Unit
    """
    The unit conversion was requested to.
    """

    def error_message(self) -> str:
        return (
            f"Cannot convert a value from {self.source_unit} to {self.target_unit}: "
            f"no conversion factor is known for one of these units."
        )

    def suggest_correction(self) -> str:
        return "Only convert between units that are both listed in _SECONDS_PER_UNIT."


@dataclass
class MeanAndStandardDeviation:
    """
    Class that represents a mean and standard deviation for tables that will be directly
    rendered as mean +- standard deviation.

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

    unit: Unit = Unit.NONE
    """
    The physical unit :attr:`mean` and :attr:`standard_deviation` are expressed in.
    """

    def __str__(self) -> str:
        suffix = f" {self.unit.value}" if self.unit is not Unit.NONE else ""
        return f"{round(self.mean, 2)} ± {round(self.standard_deviation, 2)}{suffix}"

    @classmethod
    def from_measurements(
        cls, measurements: list[float], unit: Unit = Unit.NONE
    ) -> MeanAndStandardDeviation:
        std = 0.0 if len(measurements) < 2 else statistics.stdev(measurements)
        return cls(
            mean=round(statistics.mean(measurements), 2),
            standard_deviation=round(std, 4),
            unit=unit,
        )

    def to(self, unit: Unit) -> MeanAndStandardDeviation:
        """
        Convert this value to another unit.

        :param unit: The unit to convert to.
        :raises IncompatibleUnitConversionError: If either the current or the requested
            unit has no known conversion factor.
        """
        if self.unit not in _SECONDS_PER_UNIT or unit not in _SECONDS_PER_UNIT:
            raise IncompatibleUnitConversionError(
                source_unit=self.unit, target_unit=unit
            )
        factor = _SECONDS_PER_UNIT[self.unit] / _SECONDS_PER_UNIT[unit]
        return MeanAndStandardDeviation(
            mean=round(self.mean * factor, 2),
            standard_deviation=round(self.standard_deviation * factor, 4),
            unit=unit,
        )


@dataclass
class VolumeBound(Bounds[float]):
    """
    A :class:`Bounds` interval known to contain some volume, for tables that report a
    bounded estimate instead of a single point value.

    Use this in experiment results whenever the true volume cannot be measured exactly
    (e.g. it was estimated by sampling) but valid lower and/or upper bounds can be.
    """

    def __str__(self) -> str:
        return f"[{round(self.lower, 4)}, {round(self.upper, 4)}]"


@dataclass
class PercentageBound(Bounds[float]):
    """
    A :class:`Bounds` interval, in percent, bounding the ratio of two
    :class:`VolumeBound` quantities.

    Use this to express one bounded volume as a share of another (e.g. how much of the
    true free volume a covering is known to reach) without collapsing either bound into
    a single, falsely precise point estimate.
    """

    def __str__(self) -> str:
        return f"[{round(self.lower, 2)}%, {round(self.upper, 2)}%]"

    @classmethod
    def ratio_of(
        cls, numerator: VolumeBound, denominator: VolumeBound
    ) -> PercentageBound:
        """
        :param numerator: Bound on the covered quantity.
        :param denominator: Bound on the reference quantity.
        :return: A percentage bound on ``numerator / denominator``, clipped to
            ``[0, 100]``. The worst case for each end is used: the lower end pairs the
            smallest numerator with the largest denominator, and vice versa for the
            upper end.
        """
        raw_lower = 100.0 * numerator.lower / denominator.upper
        raw_upper = 100.0 * numerator.upper / denominator.lower
        lower = max(0.0, min(raw_lower, 100.0))
        upper = max(lower, min(raw_upper, 100.0))
        return cls(lower=lower, upper=upper)


@dataclass
class ExperimentResult:
    """
    Class for results from experiments.

    Use this when you want to create a table of results (measurements) from experiments
    for scientific articles.

    This class is like a single row in a table of experiments.

    Assumptions made here are that there are only built in like fields or one-to-one
    relationships with other ExperimentResult classes.
    """

    @classmethod
    def introspector(cls) -> AttributeIntrospector:
        return DataclassOnlyIntrospector()

    @classmethod
    def recursive_fields(cls) -> list[DiscoveredAttribute]:
        """
        :return: One attribute per column of the row, with the attributes of a result held
            by this one taking the place of the field holding it.

        ..note:: A field annotated with anything other than a class, such as a measurement
            that may be absent, reports a value of its own and is therefore a column.
        """
        result = []
        type_hints = get_type_hints_of_object(cls)
        for field_ in cls.introspector().discover(cls):
            resolved_type = type_hints[field_.field.name]
            if isinstance(resolved_type, type) and issubclass(
                resolved_type, ExperimentResult
            ):
                result.extend(resolved_type.recursive_fields())
            else:
                result.append(field_)
        return result

    @classmethod
    def get_column_names(cls) -> list[str]:
        return [field_.field.name for field_ in cls.recursive_fields()]

    def get_column_values(self) -> list[Any]:
        """
        :return: One value per column of the row, with a result held by this one
            contributing its own values in place of the field holding it.

        ..note:: The values are recursed the same way :meth:`recursive_fields` recurses the
            columns, since a nested result's values live on the nested result and not on the
            row holding it.
        """
        values: list[Any] = []
        type_hints = get_type_hints_of_object(type(self))
        for field_ in self.introspector().discover(type(self)):
            value = getattr(self, field_.field.name)
            resolved_type = type_hints[field_.field.name]
            if isinstance(resolved_type, type) and issubclass(
                resolved_type, ExperimentResult
            ):
                values.extend(value.get_column_values())
            else:
                values.append(value)
        return values


class RowsOfDifferingTypes(Exception):
    """
    Raised when a table is built over rows of more than one type, since a table's
    columns come from its row type and rows of different types have no common set of
    columns.
    """


class RowIsNotAnExperimentResult(Exception):
    """
    Raised when a table is built over rows that are not experiment results, which have
    no columns to present.
    """


@dataclass
class ExperimentsTable:
    """
    A collection of experiments ready to be presented as a table in a scientific
    article.

    This class assumes that all rows in the table have the same type and are a subclass
    of ExperimentResult.
    """

    experiments: list[ExperimentResult]
    """
    The list of experiments to be presented in the table.
    """

    def __post_init__(self):
        if not self.experiments:
            return
        row_types = {type(row) for row in self.experiments}
        if len(row_types) > 1:
            raise RowsOfDifferingTypes(
                "a table presents one kind of row, but was given "
                f"{', '.join(sorted(each.__name__ for each in row_types))}"
            )
        [row_type] = row_types
        if not issubclass(row_type, ExperimentResult):
            raise RowIsNotAnExperimentResult(
                f"{row_type.__name__} is not an experiment result, so it has no columns"
            )

    @property
    def row_class(self) -> type[ExperimentResult] | None:
        if not self.experiments:
            return None
        return type(self.experiments[0])

    def write_manifest(
        self, output_directory: pathlib.Path, file_name: str
    ) -> pathlib.Path:
        """
        Records the rows as JSON alongside the table presenting them.

        :param output_directory: Directory the manifest is written to.
        :param file_name: Name of the manifest file.
        :return: Path the manifest was written to.
        """
        manifest_path = output_directory / file_name
        manifest_path.write_text(json.dumps(self.as_json_rows(), indent=2))
        return manifest_path

    def as_json_rows(self) -> list[dict[str, Any]]:
        """
        :return: One entry per row, holding the same columns the table presents, so that a
            result reported within another is recorded flat rather than nested.
        """
        return [
            {
                field_.field.name: self.as_json_value(value)
                for field_, value in zip(
                    row.recursive_fields(), row.get_column_values()
                )
            }
            for row in self.experiments
        ]

    @classmethod
    def as_json_value(cls, value: Any) -> Any:
        """
        :param value: A value that may contain enum members or nested results, neither of
            which :mod:`json` can serialise on its own.
        :return: The value with every enum member replaced by its name and every nested
            result replaced by its fields.
        """
        if isinstance(value, enum.Enum):
            return value.name
        if isinstance(value, dict):
            return {key: cls.as_json_value(entry) for key, entry in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.as_json_value(entry) for entry in value]
        if is_dataclass(value) and not isinstance(value, type):
            return cls.as_json_value(asdict(value))
        return value


@dataclass
class TypstRenderer:
    """
    Represents a renderer for converting an ExperimentsTable into Typst markup.
    """

    experiments_table: ExperimentsTable
    """
    The experiments to render.
    """

    reported_decimals: int = 2
    """
    Number of decimals a measured quantity is reported to.
    """

    absent_measurement: str = "--"
    """
    What is written where an experiment established no value.
    """

    def render_cell(self, value: Any) -> str:
        """
        Renders a single value as a reader of the table should see it.

        A value the experiment holds as a name is written as that name, a measured
        quantity is written at the precision the table reports, and a measurement the
        experiment did not establish is marked absent rather than written as a number.

        :param value: One value of a row.
        :return: The Typst cell content for it.
        """
        if value is None:
            return self.absent_measurement
        if isinstance(value, enum.Enum):
            return value.name.replace("_", " ").title()
        if isinstance(value, float):
            return f"{value:.{self.reported_decimals}f}"
        return str(value)

    def render_row(self, row: ExperimentResult) -> str:
        """
        Renders the cells of a single row in Typst format.
        """
        return ", ".join(
            [f"[{self.render_cell(value)}]" for value in row.get_column_values()]
        )

    def render_table(self) -> str:
        """
        Renders the entire ExperimentsTable into a valid Typst #table markup string.
        """
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

    def render_figure(self, caption: str) -> str:
        """
        Renders the table wrapped in a Typst #figure with a caption.

        Use this instead of :meth:`render_table` whenever the table is presented to a
        reader, so every table in a document explains what it shows.

        :param caption: Caption text describing what the table shows.
        :return: Typst markup for a captioned figure.
        """
        return f"#figure(\n{self.render_table()},\n  caption: [{caption}]\n)"
