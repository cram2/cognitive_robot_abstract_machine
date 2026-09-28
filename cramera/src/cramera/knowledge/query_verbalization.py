"""
Reading an entity query language expression back as English, coloured by role.

The panel shows this as the asked question, so a query says what it asks rather than
only what came back — with class and attribute words linking to what explains them:
their published documentation, or their source in the repository.
"""

from __future__ import annotations

import html
import inspect
import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from krrood.entity_query_language.verbalization.exceptions import (
    UnverbalizableExpressionError,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)
from krrood.entity_query_language.verbalization.rendering.formatter import HTMLFormatter
from krrood.entity_query_language.verbalization.rendering.renderer import (
    ParagraphRenderer,
)
from krrood.entity_query_language.verbalization.rendering.source_link_resolver import (
    AutoAPIResolver,
)
from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
from typing_extensions import Any, Dict, FrozenSet, Optional

from cramera import paths

SOURCE_SITE_VARIABLE = "CRAMERA_SOURCE_SITE"
"""
Environment variable overriding where the repository's files are read online.
"""

SOURCE_REPOSITORY = "https://github.com/cram2/cognitive_robot_abstract_machine"
"""
The repository the workspace's packages are read from.
"""

DEFAULT_SOURCE_REVISION = "main"
"""
The revision a file is read at when the checkout does not say which one it is on.
"""

DOCUMENTED_PACKAGES: FrozenSet[str] = frozenset(
    {
        "coraplex",
        "giskardpy",
        "krrood",
        "probabilistic_model",
        "random_events",
        "semantic_digital_twin",
    }
)
"""
The workspace packages whose AutoAPI documentation the docs site publishes.

Words naming classes of any other package are linked to their source instead
(:class:`RepositorySourceResolver`).
"""

DOCUMENTATION_SITE_VARIABLE = "CRAMERA_DOCUMENTATION_SITE"
"""
Environment variable overriding the documentation site the verbalized words link to.
"""

DEFAULT_DOCUMENTATION_SITE = "https://cram2.github.io/cognitive_robot_abstract_machine"
"""
The published aggregate docs site, hosting each package's docs under its own name.
"""


@dataclass(frozen=True)
class PublishedDocumentationResolver:
    """
    Resolves a verbalized word's source reference to its published AutoAPI page.

    The docs site hosts one Sphinx build per package (``{site}/{package}/autoapi/…``),
    so the reference's own top-level package picks the build its link points into.
    """

    site: str = DEFAULT_DOCUMENTATION_SITE
    """
    Root URL of the aggregate documentation site.
    """

    def resolve(self, reference: SourceReference) -> Optional[str]:
        """
        The AutoAPI page URL a reference documents itself at, or None when its package
        publishes no documentation.

        :param reference: Source reference of the class or attribute a word names.
        """
        if not isinstance(reference.owner_type, type):
            return None
        package = reference.owner_type.__module__.split(".", 1)[0]
        if package not in DOCUMENTED_PACKAGES:
            return None
        return AutoAPIResolver(
            base_url="%s/%s" % (self.site.rstrip("/"), package)
        ).resolve(reference)

    @classmethod
    def of_environment(cls) -> PublishedDocumentationResolver:
        """
        A resolver against the configured docs site (:data:`DOCUMENTATION_SITE_VARIABLE`),
        or the published one.
        """
        return cls(
            site=os.environ.get(DOCUMENTATION_SITE_VARIABLE)
            or DEFAULT_DOCUMENTATION_SITE
        )


@dataclass(frozen=True)
class RepositorySourceResolver:
    """
    Resolves a verbalized word to the source the class it names is written in.

    This answers for the packages the docs site does not publish, which is every package
    a scene's own entities are defined in. The link points at the class's declaration;
    an attribute has no place of its own outside the published documentation, so its
    word points at the class holding it.
    """

    site: str
    """
    Where a repository-relative path is read at, the revision it is read at included.
    """

    root: Path = field(default_factory=paths.repository_root)
    """
    The checkout the running classes are read from, whose paths the links are of.
    """

    def resolve(self, reference: SourceReference) -> Optional[str]:
        """
        The URL the class a word names is written at, or None when it is not written in
        this repository.

        :param reference: Source reference of the class or attribute a word names.
        """
        if not isinstance(reference.owner_type, type):
            return None
        source = self.source_file(reference.owner_type)
        if source is None:
            return None
        page = "%s/%s" % (
            self.site.rstrip("/"),
            source.relative_to(self.root).as_posix(),
        )
        line = declaration_line(source, reference.owner_type.__name__)
        return page if line is None else "%s#L%d" % (page, line)

    def source_file(self, owner_type: type) -> Optional[Path]:
        """
        The Python file a class is written in inside this repository, or None when it is
        written elsewhere or has no source at all.

        :param owner_type: The class to locate.
        """
        module = inspect.getmodule(owner_type)
        if module is None or module.__spec__ is None or module.__spec__.origin is None:
            return None
        source = Path(module.__spec__.origin).resolve()
        if source.suffix != ".py" or not source.is_relative_to(self.root):
            return None
        return source

    @classmethod
    def of_environment(cls) -> RepositorySourceResolver:
        """
        A resolver against the configured source site (:data:`SOURCE_SITE_VARIABLE`), or
        the repository the running checkout is read from.
        """
        root = paths.repository_root()
        return cls(
            site=os.environ.get(SOURCE_SITE_VARIABLE) or source_site(root), root=root
        )


def source_site(root: Path) -> str:
    """
    Where one checkout's files are read online: the repository at the commit the checkout
    is on, so a link shows the code that is running rather than whatever the default
    branch holds. GitHub serves any commit pushed to the repository or a fork of it.

    :param root: The checkout the links are built for.
    """
    return "%s/blob/%s" % (
        SOURCE_REPOSITORY,
        checked_out_commit(root) or DEFAULT_SOURCE_REVISION,
    )


@lru_cache(maxsize=None)
def checked_out_commit(root: Path) -> Optional[str]:
    """
    The commit a checkout is on, or None when it is not one git can read.

    Remembered per directory: a running viewer answers from the checkout it was started
    from, and asking git per verbalized word would cost a process apiece.

    :param root: The directory to read the commit of.
    """
    read = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return read.stdout.strip() if read.returncode == 0 else None


def declaration_line(source: Path, class_name: str) -> Optional[int]:
    """
    The line a class is declared on in a source file, or None when the file declares no
    such class of its own.

    :param source: The Python file to read.
    :param class_name: Name of the class to find the declaration of.
    """
    declarations = ("class %s(" % class_name, "class %s:" % class_name)
    for number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.startswith(declarations):
            return number
    return None


@dataclass(frozen=True)
class WordLinkResolver:
    """
    Resolves a verbalized word to what explains the class it names.

    Documentation first and source second: a published AutoAPI page reads better than
    the source it was generated from, and every class a query names has source.
    """

    documentation: PublishedDocumentationResolver
    """
    Where a word goes when its package's documentation is published.
    """

    source: RepositorySourceResolver
    """
    Where every other word goes.
    """

    def resolve(self, reference: SourceReference) -> Optional[str]:
        """
        The URL a word links to, or None when neither knows the class it names.

        :param reference: Source reference of the class or attribute a word names.
        """
        return self.documentation.resolve(reference) or self.source.resolve(reference)

    @classmethod
    def of_environment(cls) -> WordLinkResolver:
        """
        A resolver against the configured documentation and source sites.
        """
        return cls(
            documentation=PublishedDocumentationResolver.of_environment(),
            source=RepositorySourceResolver.of_environment(),
        )


@dataclass
class EscapedHtmlFormatter(HTMLFormatter):
    """
    Krrood's HTML colour markup, with the display text escaped.

    The rendered sentence is inserted into the page as markup, and a query's literals
    are whatever a viewer typed, so they arrive as text rather than as tags.
    """

    def colorize(self, text: str, role: SemanticRole) -> str:
        """
        Colour one already-escaped span of display text.

        :param text: Plain display text to escape and colour.
        :param role: The semantic role deciding the colour.
        """
        return super().colorize(html.escape(text), role)


@dataclass(frozen=True)
class QueryVerbalization:
    """
    One query read back as English, in both the renderings the viewer needs.
    """

    text: str
    """
    The sentence as plain prose, for logs and for anything that cannot show markup.
    """

    html: str
    """
    The same sentence as ``<span>`` markup, coloured by semantic role.
    """

    @classmethod
    def of_expression(cls, expression: Any) -> Optional[QueryVerbalization]:
        """
        Read one entity query language expression back as English.

        Building the sentence leaves the expression evaluable, so the caller can word a
        query and then answer it.

        :param expression: The expression to word.
        :return: Both renderings, or None for anything krrood cannot word — a sentence
            is a nicety, and failing to build one must not cost the caller its answer.
        """
        try:
            fragment = EQLVerbalizer().build(expression)
        except UnverbalizableExpressionError:
            return None
        return cls(
            text=ParagraphRenderer().render(fragment),
            html=ParagraphRenderer(
                EscapedHtmlFormatter(),
                link_resolver=WordLinkResolver.of_environment(),
            ).render(fragment),
        )

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the answer panel reads.
        """
        return {"text": self.text, "html": self.html}
