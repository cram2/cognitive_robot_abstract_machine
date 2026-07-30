"""Tests for the :class:`CodeGenerator` base class in ``krrood.code_generation``."""

from __future__ import annotations

import os
import tempfile

import pytest

from krrood.code_generation.generator import CodeGenerator


class TestCodeGenerator:
    """Tests for :class:`CodeGenerator`."""

    def test_construction(self):
        """CodeGenerator can be constructed with a valid template directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = CodeGenerator(template_directory=tmpdir)
            assert gen.template_directory == tmpdir
            assert gen.environment is not None

    def test_render_template(self):
        """A simple template renders with the given context variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = os.path.join(tmpdir, "test.py.jinja")
            with open(template_path, "w") as f:
                f.write('"""{{ docstring }}"""\n{{ var_name }} = {{ value }}')

            gen = CodeGenerator(template_directory=tmpdir)
            output = gen.render(
                "test.py.jinja", docstring="Test module", var_name="x", value=42
            )
            assert '"""Test module"""' in output
            assert "x = 42" in output

    def test_render_empty_template(self):
        """An empty template renders to an empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = os.path.join(tmpdir, "empty.py.jinja")
            with open(template_path, "w") as f:
                f.write("")

            gen = CodeGenerator(template_directory=tmpdir)
            output = gen.render("empty.py.jinja")
            assert output.strip() == ""

    def test_template_not_found(self):
        """Missing template raises jinja2.TemplateNotFound."""
        import jinja2

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = CodeGenerator(template_directory=tmpdir)
            with pytest.raises(jinja2.TemplateNotFound):
                gen.render("nonexistent.py.jinja")
