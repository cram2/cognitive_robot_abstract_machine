"""Base class for Jinja2-based Python code generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import jinja2


@dataclass
class CodeGenerator:
    """Base class for Jinja2-based code generation.

    Each domain package that needs Jinja2 templates instantiates this class
    with its own *template_directory* so templates live alongside the code that
    uses them.

    .. code-block:: python

        from importlib.resources import files

        generator = CodeGenerator(
            template_directory=str(files("my.package") / "templates")
        )
        output = generator.render("my_template.py.jinja", variable=...)

    :param template_directory: Absolute path to the directory containing
        ``.py.jinja`` template files.
    """

    template_directory: str
    """Absolute path to the templates directory."""

    environment: jinja2.Environment = field(init=False, default=None)
    """The Jinja2 :class:`~jinja2.Environment` used to load and render templates."""

    def __post_init__(self):
        self.environment = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_directory),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **context: object) -> str:
        """Load *template_name* and render it with *context*.

        :param template_name: The ``.py.jinja`` file name (e.g. ``"rdr_module.py.jinja"``).
        :param context: Keyword arguments passed to the template as variables.
        :returns: The rendered source string.
        """
        template = self.environment.get_template(template_name)
        return template.render(**context)
