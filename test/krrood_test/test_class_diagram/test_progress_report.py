import pytest

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.progress_report import (
    ClassDiagramProgress,
    LINE_PREFIX,
    ProgressEnvironmentVariable,
)

from ..dataset.department_and_employee import Department, Employee

# %% the report a line carries


def test_a_report_survives_being_written_and_read_back():
    report = ClassDiagramProgress("Employee", 12)

    assert ClassDiagramProgress.from_line(report.to_line()) == report


def test_a_line_of_logging_carries_no_report():
    assert ClassDiagramProgress.from_line("INFO:krrood:resolving Employee.name") is None


# %% reporting while a diagram is built


@pytest.fixture
def reported(capfd, monkeypatch) -> list:
    """
    Build a diagram with progress asked for, and read back what it reported.

    :return: The reports the build wrote, in the order it wrote them.
    """
    monkeypatch.setenv(ProgressEnvironmentVariable.REPORT_PROGRESS, "1")
    ClassDiagram([Department, Employee])
    return [
        ClassDiagramProgress.from_line(line)
        for line in capfd.readouterr().out.splitlines()
        if ClassDiagramProgress.from_line(line) is not None
    ]


def test_every_class_of_the_diagram_is_reported(reported):
    assert [report.class_name for report in reported] == [
        Department.__name__,
        Employee.__name__,
    ]


def test_a_report_names_how_many_classes_there_are(reported):
    assert {report.total_classes for report in reported} == {2}


def test_nothing_is_reported_when_no_one_asked(capfd, monkeypatch):
    monkeypatch.delenv(ProgressEnvironmentVariable.REPORT_PROGRESS, raising=False)

    ClassDiagram([Department, Employee])

    assert LINE_PREFIX not in capfd.readouterr().out
