from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Optional, Protocol

from krrood.entity_query_language.verbalization.fragments.source_reference import (
    SourceReference,
)

_log = logging.getLogger(__name__)


class SourceLinkResolver(Protocol):
    """
    Protocol: maps a source reference to a URL string, or ``None`` when the class or
    attribute cannot be located.
    """

    def resolve(self, reference: SourceReference) -> Optional[str]:
        """
        Resolve *reference* to a URL string.

        :param reference: Source reference to resolve.
        :return: URL string, or ``None`` when the reference cannot be resolved.
        """
        ...


@dataclass
class AutoAPIResolver:
    """
    Resolves source references to Sphinx AutoAPI documentation pages.
    """

    base_url: str
    """Root URL of the generated docs site, e.g. ``"https://myproject.readthedocs.io/en/latest"``
    or ``"http://localhost:63342/project/doc/_build/html"``."""

    html_root: Optional[Path] = None
    """
    Optional local path to the Sphinx HTML output directory.

    When set, resolution verifies that the AutoAPI page exists on disk and logs a
    warning if it does not.
    """

    def resolve(self, reference: SourceReference) -> Optional[str]:
        """
        Resolve *reference* to a Sphinx AutoAPI page URL.

        Constructs the URL as::

            {base_url}/autoapi/{module/path}/index.html#{module.QualName[.attribute]}

        When ``html_root`` is set and the page does not exist on disk, logs a warning suggesting
        the docs be rebuilt.

        :param reference: Source reference to resolve.
        :return: AutoAPI page URL, or ``None`` when *reference.owner_type* has no ``__module__``.
        """
        if not isinstance(reference.owner_type, type):
            return None
        module = reference.owner_type.__module__
        qualname = reference.owner_type.__qualname__
        module_path = module.replace(".", "/")
        anchor = f"{module}.{qualname}"
        if reference.attribute is not None:
            anchor = f"{anchor}.{reference.attribute}"
        base = self.base_url.rstrip("/")
        url = f"{base}/autoapi/{module_path}/index.html#{anchor}"
        if self.html_root is not None:
            page = self.html_root / "autoapi" / module_path / "index.html"
            if not page.exists():
                _log.warning(
                    "AutoAPI page for %s.%s does not exist at %s — "
                    "the class may be missing from the docs; "
                    "try rebuilding: sphinx-build doc doc/_build/html",
                    module,
                    qualname,
                    page,
                )
        return url

    @classmethod
    def for_in_site_docs(cls, levels_up: int = 2) -> AutoAPIResolver:
        """
        Build a resolver for links rendered *inside* the documentation site itself.

        The verbalization output is embedded in a docs page (e.g. ``eql/user/verbalization.html``),
        and the AutoAPI tree is a sibling at the site root (``autoapi/…``). Emitting links relative
        to the current page therefore resolves them correctly in *any* build or host — a standalone
        ``_build/html``, a Pages preview, or the published aggregate — without baking in an absolute
        URL. Prefer this over :meth:`for_package` (which targets a specific local IDE server) for
        links that ship in the built docs.

        :param levels_up: The current page's directory depth below the site root, i.e. how many
            ``..`` segments reach the root. The verbalization pages live at ``eql/<section>/`` → 2.
        :return: A resolver whose links are relative to a page *levels_up* directories deep.

        >>> AutoAPIResolver.for_in_site_docs().base_url
        '../..'
        """
        return cls(base_url="/".join([".."] * levels_up) or ".")

    @classmethod
    def for_package(cls, package_name: str, port: int = 63342) -> AutoAPIResolver:
        """
        Build a resolver for *package_name*'s locally built Sphinx docs, with the base
        URL targeting the JetBrains IDE built-in HTTP server.

        .. note::
            This targets a live ``localhost`` IDE server, so it is for the local
            preview-while-editing workflow only. For links that ship inside the built docs, use
            :meth:`for_in_site_docs` instead (relative links that resolve on any host).

        :param package_name: The installed package whose docs to resolve against.
        :param port: Port of the JetBrains IDE built-in HTTP server.
        :return: A resolver pointing at the package's local Sphinx HTML output.
        :raises ImportError: if *package_name* cannot be imported.
        :raises FileNotFoundError: if ``doc/_build/html`` does not exist —
            build the docs first with ``sphinx-build doc doc/_build/html``.
        """
        try:
            pkg = importlib.import_module(package_name)
        except ImportError as exc:
            raise ImportError(f"Cannot import package {package_name!r}: {exc}") from exc

        pkg_file = Path(inspect.getfile(pkg)).resolve()

        package_root: Optional[Path] = None
        for parent in pkg_file.parents:
            if (parent / "pyproject.toml").exists():
                package_root = parent
                break
        if package_root is None:
            raise FileNotFoundError(
                f"No pyproject.toml found in any parent of {pkg_file}"
            )

        html_root = package_root / "doc" / "_build" / "html"
        if not html_root.exists():
            raise FileNotFoundError(
                f"Sphinx HTML output not found at {html_root}. "
                f"Build the docs first: sphinx-build doc doc/_build/html"
            )

        git_root: Optional[Path] = None
        for parent in [package_root, *package_root.parents]:
            if (parent / ".git").exists():
                git_root = parent
                break
        if git_root is None:
            git_root = package_root.parent

        rel_html = html_root.relative_to(git_root)
        return cls(
            base_url=f"http://localhost:{port}/{git_root.name}/{rel_html}",
            html_root=html_root,
        )
