"""
Base class for Jinja2-based Python code generation.

Follows the pattern established by
:class:`krrood.ormatic.sqlalchemy_generator.SQLAlchemyGenerator`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import jinja2


@dataclass
class CodeGenerator:
    """Base class for Jinja2-based code generation.

    Each domain package that needs Jinja2 templates instantiates this class
    with its own *template_dir* so templates live alongside the code that
    uses them::

        generator = CodeGenerator(
            template_dir=os.path.join(os.path.dirname(__file__), "templates")
        )
        output = generator.render("my_template.py.jinja", var1=..., var2=...)

    :param template_dir: Absolute path to the directory containing ``.py.jinja``
        template files.
    """

    template_dir: str
    """Absolute path to the templates directory."""

    env: jinja2.Environment = field(init=False, default=None)
    """The Jinja2 :class:`~jinja2.Environment` used to load and render templates."""

    def __post_init__(self):
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **context: object) -> str:
        """Load *template_name* and render it with *context*.

        :param template_name: The ``.py.jinja`` file name (e.g. ``"rdr_module.py.jinja"``).
        :param context: Keyword arguments passed to the template as variables.
        :returns: The rendered source string.
        """
        template = self.env.get_template(template_name)
        return template.render(**context)
