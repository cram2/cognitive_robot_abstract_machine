import logging
import tempfile

import pytest
import yaml

from robokudo.utils.logging_configuration import (
    DynamicCompactFormatter,
    configure_logging,
)


class TestUtilsLoggingConfiguration:

    def test_dynamic_compact_formatter_line_width_calculation(self):
        formatter = DynamicCompactFormatter("%(name)s %(filename_line)-50s %(message)s")
        assert formatter.max_file_line_width == 50

    def test_dynamic_compact_formatter_default_line_width(self):
        formatter = DynamicCompactFormatter("%(name)s %(filename_line)s %(message)s")
        assert formatter.max_file_line_width == 44

    def test_dynamic_compact_formatter_default_fmt(self):
        formatter = DynamicCompactFormatter(None)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/path/to/short_filename.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="short_function",
        )
        formatted = formatter.format(record)
        assert formatted == record.msg

    def test_dynamic_compact_formatter_truncate_file_and_method(self):
        formatter = DynamicCompactFormatter("%(name)s %(filename_line)-10s %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/path/to/very_long_filename.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="very_long_function_name",
        )
        formatted = formatter.format(record)
        assert ".../...:42" == formatted.split(" ")[1]

    def test_dynamic_compact_formatter_truncate_file(self):
        formatter = DynamicCompactFormatter("%(name)s %(filename_line)-31s %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/some/path/to/a/very_very_long_filename.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="short_function_name",
        )
        formatted = formatter.format(record)
        assert "ve.../short_function_name:42" == formatted.split(" ")[1]

    def test_dynamic_compact_formatter_truncate_fallback(self):
        formatter = DynamicCompactFormatter("%(name)s %(filename_line)-6s %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/path/to/very_long_filename.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="very_long_function_name",
        )
        formatted = formatter.format(record)
        assert "...:42" == formatted.split(" ")[1]

    @pytest.mark.parametrize(
        ["level", "name"],
        [
            (10, logging.DEBUG),
            (20, logging.INFO),
            (30, logging.WARNING),
            (40, logging.ERROR),
            (50, logging.CRITICAL),
        ],
    )
    def test_configure_logging(self, level: int, name: int):
        config = {
            "root": level,
            "module0": 10,
            "module1": 20,
            "module2": 30,
            "module3": 40,
            "module4": 50,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            configure_logging(f.name)

            assert logging.getLogger("root").level == name
            assert logging.getLogger("module0").level == logging.DEBUG
            assert logging.getLogger("module1").level == logging.INFO
            assert logging.getLogger("module2").level == logging.WARNING
            assert logging.getLogger("module3").level == logging.ERROR
            assert logging.getLogger("module4").level == logging.CRITICAL
