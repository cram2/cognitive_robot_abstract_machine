"""
Interactive :class:`ExpertInterface` backed by an embedded IPython shell.

The expert is shown the case rendered as a table with the instructions printed *below* it
(nearest the prompt), then authors a **live EQL condition expression** over ``case_variable``
and assigns it to ``conditions`` (and a ``conclusion`` when no ground-truth target is known).

Pressing Ctrl-D *submits*: the assignment is validated and, if it is invalid or missing, the
error is printed inline and the **same shell stays open** rather than bailing out. Calling
``exit()`` (or ``quit()``) cancels the session unconditionally, raising
:class:`~krrood.entity_query_language.rdr.interface.ExpertAbort`.

All on-screen text is composed as plain prose and coloured through a single
:class:`Palette`, and the header is assembled from small section builders so styling lives
in one place. Line magics — ``%help``, ``%show_tree`` and (when aids are configured) ``%aid``
— keep the standing header short while staying discoverable.

The actual shell launch is injectable (``shell_runner``) so tests can play the expert's
part without a real terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Callable, Dict, List, Optional

from colorama import Fore, Style

from krrood.entity_query_language.rdr.case_table import (
    DEFAULT_MIN_COLUMN_WIDTH,
    render_case_table,
)
from krrood.entity_query_language.rdr.interface import (
    CASE_INSTANCE_NAME,
    CASE_VARIABLE_NAME,
    EXIT_NAME,
    _ABORT_FLAG,
    AnswerRequest,
    CaseContext,
    ExpertInterface,
)
from krrood.entity_query_language.rdr.expert import ANSWER_NAME, CONCLUSION_NAME
from krrood.entity_query_language.rdr.magics import (
    BACKWARD_MAGIC,
    CONDITIONS_MAGIC,
    CONCLUSION_MAGIC,
    _KNOWLEDGE_KEY,
    _make_assign_exit_magic,
    _make_knowledge_magic,
)
from krrood.entity_query_language.rdr.prompt_sections import (  # noqa: F401
    AID_MAGIC,
    HELP_MAGIC,
    PROMPT_SECTIONS,
    RenderContext,
    SHOW_TREE_MAGIC,
)
from krrood.entity_query_language.rdr.rule_tree_view import render_rule_tree

#: A shell runner takes ``(namespace, header)`` and must leave the expert's assignments
#: (and any ``exit()`` flag) visible in ``namespace`` when it returns.
ShellRunner = Callable[[Dict[str, Any], str], None]

# AID_MAGIC, HELP_MAGIC, SHOW_TREE_MAGIC are defined in prompt_sections and re-exported above.

#: ANSI escape sequence: erase display and return cursor to top-left.
#: Emitted before each new embedded shell so each case starts on a clean screen.
CLEAR_SCREEN = "\033[2J\033[H"

#: Private namespace key holding the zero-arg rule-tree renderer for the ``%show_tree`` magic.
_TREE_RENDER_KEY = "__rule_tree_render__"

#: Private namespace key holding the zero-arg help-text builder for the ``%help`` magic.
_HELP_TEXT_KEY = "__expert_help__"

#: Private namespace key holding the zero-arg aid-text renderer for the ``%aid`` magic.
_AID_TEXT_KEY = "__expert_aid__"


@dataclass
class Palette:
    """Maps semantic roles to ANSI styling behind a single ``use_color`` switch.

    Keeps colour out of the message text: callers write plain prose and wrap each fragment
    in the role that fits it, so one line can mix roles without leaking or losing codes.
    """

    use_color: bool = True

    def _paint(self, text: str, *codes: str) -> str:
        if not self.use_color:
            return text
        return f"{''.join(codes)}{text}{Style.RESET_ALL}"

    def label(self, text: str) -> str:
        """Ordinary instruction prose."""
        return self._paint(text, Fore.MAGENTA)

    def good(self, text: str) -> str:
        """A correct / target value."""
        return self._paint(text, Fore.GREEN)

    def wrong(self, text: str) -> str:
        """A value that is currently wrong (no emphasis)."""
        return self._paint(text, Fore.RED)

    def strong_wrong(self, text: str) -> str:
        """A wrong value the expert must steer away from (emphasised)."""
        return self._paint(text, Fore.RED, Style.BRIGHT)

    def neutral(self, text: str) -> str:
        """A value with no good/bad judgement attached."""
        return self._paint(text, Fore.WHITE)

    def code(self, text: str) -> str:
        """A name or expression the expert can type."""
        return self._paint(text, Fore.CYAN)

    def keyword(self, text: str) -> str:
        """An emphasised term within prose (e.g. the word ``condition``)."""
        return self._paint(text, Fore.CYAN, Style.BRIGHT)

    def hint(self, text: str) -> str:
        """A low-key pointer to a command."""
        return self._paint(text, Fore.YELLOW)

    def error(self, text: str) -> str:
        """A validation error."""
        return self._paint(text, Fore.RED)


@dataclass
class IPythonInterface(ExpertInterface):
    """An :class:`~krrood.entity_query_language.rdr.interface.ExpertInterface` backed by an
    embedded IPython shell.

    Assembles a namespace from the case definition scope, renders a colour-coded header
    via :data:`~krrood.entity_query_language.rdr.prompt_sections.PROMPT_SECTIONS`, and
    drives the validate-re-prompt loop.  The actual shell launch is injectable via
    :attr:`shell_runner` so tests can run without a real terminal.
    """

    shell_runner: Optional[ShellRunner] = None
    """Injectable launcher; defaults to a real embedded IPython shell. Tests pass a stub."""

    min_column_width: int = DEFAULT_MIN_COLUMN_WIDTH
    """Smallest width a case-table pair column may take; sets how many fit per row."""

    use_color: bool = True
    """Whether the header, framing and magics emit ANSI colour."""

    rdr: Optional[Any] = None
    """The :class:`~krrood.entity_query_language.rdr.single_class.EQLSingleClassRDR` being fit,
    exposed for the ``%knows`` backward-inference magic. ``None`` when the interface is used
    without an RDR (no magic is registered)."""

    _aid_cache: Optional[str] = field(init=False, default=None)
    """Memoized aid ``present()`` output for the current ``interact()`` call (set once per
    question in :meth:`_build_namespace`, reused by the header and the ``%aid`` magic)."""

    @property
    def palette(self) -> Palette:
        """The styling used for every piece of on-screen text."""
        return Palette(self.use_color)

    # ----------------------------------------------------------------- header

    def _render_header(
        self,
        context: CaseContext,
        requests: List[AnswerRequest],
        errors: Dict[str, str],
    ) -> str:
        ctx = RenderContext(case=context, requests=requests, palette=self.palette)
        parts: List[str] = ["", self._case_table(context), ""]
        if self._aid_cache:
            parts.extend([self._aid_cache, ""])
        for section in PROMPT_SECTIONS:
            if section.applicable(ctx):
                parts.extend(section.lines(ctx))
        if errors:
            parts.append(self._format_errors(errors))
        parts.append("")
        return "\n".join(parts)

    def _case_table(self, context: CaseContext) -> str:
        return render_case_table(
            context.case_instance, self.min_column_width, use_color=self.use_color
        )

    def _format_errors(self, errors: Dict[str, str]) -> str:
        """:return: A red, one-line-per-error block (empty mapping -> empty string)."""
        p = self.palette
        return "\n".join(
            p.error(f"[error] {name}: {message}") for name, message in errors.items()
        )

    # --------------------------------------------------------------- guidance

    def _render_tree(self, context: CaseContext) -> Optional[str]:
        """:return: The coloured rule-tree text for this case, or ``None`` if unavailable."""
        trace = context.trace
        if trace is None or trace.rule_tree_root is None:
            return None
        return render_rule_tree(trace, use_color=self.use_color)

    def _aid_text(self, context: CaseContext) -> Optional[str]:
        """:return: The combined ``present()`` output of every aid, or ``None`` if none speak.

        Called once per question (memoized in :meth:`_build_namespace`) so a heavy aid does not
        re-run on each re-prompt cycle.
        """
        fragments = [text for aid in context.aids if (text := aid.present(context))]
        return "\n".join(fragments) if fragments else None

    def _help_text(self, context: CaseContext, requests: List[AnswerRequest]) -> str:
        """The how-to-answer guidance printed by ``%help`` — plain prose, one accent colour."""
        p = self.palette
        lines = [
            p.label("How to answer:"),
            f"  Inspect the case with {p.code(CASE_INSTANCE_NAME)} "
            f"(e.g. {p.code(f'{CASE_INSTANCE_NAME}.milk')}).",
            f"  Build your answer over {p.code(CASE_VARIABLE_NAME)} and assign it:",
        ]
        lines.extend(f"      {p.code(request.example)}" for request in requests)
        lines.append("  Or use the shorthand magic (assigns and submits in one step):")
        for request in requests:
            magic = (
                CONCLUSION_MAGIC
                if request.name == CONCLUSION_NAME
                else CONDITIONS_MAGIC
            )
            lines.append(f"      {p.code(f'%{magic} <value-or-expression>')}")
        lines.append(
            f"  Submit with {p.code('Ctrl-D')}; cancel with {p.code(f'{EXIT_NAME}()')}."
        )
        lines.append(f"  Show the rule tree with {p.code(f'%{SHOW_TREE_MAGIC}')}.")
        if context.aids:
            lines.append(f"  Show the task aid with {p.code(f'%{AID_MAGIC}')}.")
        if self.rdr is not None:
            lines.append(
                f"  Query backward inference with {p.code(f'%{BACKWARD_MAGIC} <value>')}."
            )
        lines.append(f"  Show this help again with {p.code(f'%{HELP_MAGIC}')}.")
        return "\n".join(lines)

    def _build_namespace(
        self, context: CaseContext, requests: List[AnswerRequest]
    ) -> Dict[str, Any]:
        namespace = super()._build_namespace(context, requests)
        namespace[_TREE_RENDER_KEY] = lambda: self._render_tree(context)
        namespace[_HELP_TEXT_KEY] = lambda: self._help_text(context, requests)
        # Run each aid's present() once for this question; reuse the result in the header and
        # the %aid magic so a heavy aid does not re-run on every re-prompt cycle.
        self._aid_cache = self._aid_text(context)
        if context.aids:
            namespace[_AID_TEXT_KEY] = lambda: self._aid_cache
        if self.rdr is not None:
            namespace[_KNOWLEDGE_KEY] = self.rdr
        if context.conclusion_domain is not None:
            for name, value in context.conclusion_domain.namespace_bindings().items():
                namespace.setdefault(name, value)
        return namespace

    # ------------------------------------------------------------------- shell

    def _run(
        self,
        namespace: Dict[str, Any],
        header: str,
        validate: Callable[[], Dict[str, str]],
    ) -> None:
        if self.shell_runner is not None:
            self.shell_runner(namespace, header)
        else:
            self._default_run_shell(namespace, header, validate)

    def _default_run_shell(
        self,
        namespace: Dict[str, Any],
        header: str,
        validate: Callable[[], Dict[str, str]],
    ) -> None:
        from IPython.terminal.embed import InteractiveShellEmbed

        interface = self

        class _ValidatingEmbeddedShell(InteractiveShellEmbed):
            """Vetoes a Ctrl-D exit while the answer is invalid; ``exit()`` forces the leave."""

            def ask_exit(self) -> None:
                if getattr(self, "_force_exit", False):
                    super().ask_exit()
                    return
                errors = validate()
                if errors:
                    print(interface._format_errors(errors))
                    return
                super().ask_exit()

        if self.use_color:
            print(CLEAR_SCREEN, end="", flush=True)

        shell = _ValidatingEmbeddedShell(banner1=header, user_ns=namespace)
        shell.auto_match = True
        shell.confirm_exit = False
        shell._force_exit = False

        self._register_namespace_magic(
            shell, namespace, SHOW_TREE_MAGIC, _TREE_RENDER_KEY
        )
        self._register_namespace_magic(shell, namespace, HELP_MAGIC, _HELP_TEXT_KEY)
        self._register_namespace_magic(shell, namespace, AID_MAGIC, _AID_TEXT_KEY)

        # Register action magics for the answers that are in flight this session.
        # The answer names are pre-seeded into namespace by _build_namespace, so
        # their presence is a reliable signal for which magics to register.
        if CONCLUSION_NAME in namespace:
            shell.register_magic_function(
                _make_assign_exit_magic(
                    CONCLUSION_NAME, shell, namespace, validate, self.palette
                ),
                magic_kind="line",
                magic_name=CONCLUSION_MAGIC,
            )
        if ANSWER_NAME in namespace:
            shell.register_magic_function(
                _make_assign_exit_magic(
                    ANSWER_NAME, shell, namespace, validate, self.palette
                ),
                magic_kind="line",
                magic_name=CONDITIONS_MAGIC,
            )

        # Register the backward-inference read-only magic when the RDR is available.
        if _KNOWLEDGE_KEY in namespace:
            shell.register_magic_function(
                _make_knowledge_magic(namespace, self.palette),
                magic_kind="line",
                magic_name=BACKWARD_MAGIC,
            )

        def _cancel() -> None:
            shell._force_exit = True
            namespace[_ABORT_FLAG] = True
            shell.ask_exit()

        namespace[EXIT_NAME] = _cancel
        namespace["quit"] = _cancel
        # The shell shares ``namespace``, so the expert's assignments are already visible
        # to the caller when it returns.
        shell()

    @staticmethod
    def _register_namespace_magic(
        shell: Any, namespace: Dict[str, Any], magic_name: str, render_key: str
    ) -> None:
        """Register a line magic that prints whatever the zero-arg renderer at ``render_key`` returns."""
        render = namespace.get(render_key)
        if render is None:
            return

        def magic(line: str) -> None:
            text = render()
            print(
                text
                if text
                else f"{Fore.LIGHTBLACK_EX}(nothing to show){Style.RESET_ALL}"
            )

        shell.register_magic_function(magic, magic_kind="line", magic_name=magic_name)
