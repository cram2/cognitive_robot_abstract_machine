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
in one place. Two line magics — ``%help`` and ``%show_tree`` — keep the standing header
short while staying discoverable.

The actual shell launch is injectable (``shell_runner``) so tests can play the expert's
part without a real terminal.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from krrood.entity_query_language.rdr.rule_tree_view import (
    format_condition,
    render_rule_tree,
)

#: A shell runner takes ``(namespace, header)`` and must leave the expert's assignments
#: (and any ``exit()`` flag) visible in ``namespace`` when it returns.
ShellRunner = Callable[[Dict[str, Any], str], None]

#: The IPython line magic the expert types to redisplay the rule tree.
SHOW_TREE_MAGIC = "show_tree"

#: The IPython line magic the expert types to redisplay the how-to-answer guidance.
HELP_MAGIC = "help"

#: Private namespace key holding the zero-arg rule-tree renderer for the ``%show_tree`` magic.
_TREE_RENDER_KEY = "__rule_tree_render__"

#: Private namespace key holding the zero-arg help-text builder for the ``%help`` magic.
_HELP_TEXT_KEY = "__expert_help__"


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
    """An interface based on an embedded IPython shell, that mediates the interaction with the expert"""

    shell_runner: Optional[ShellRunner] = None
    """Injectable launcher; defaults to a real embedded IPython shell. Tests pass a stub."""

    min_column_width: int = DEFAULT_MIN_COLUMN_WIDTH
    """Smallest width a case-table pair column may take; sets how many fit per row."""

    use_color: bool = True
    """Whether the header, framing and magics emit ANSI colour."""

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
        parts: List[str] = ["", self._case_table(context), ""]
        parts.extend(self._framing_lines(context))
        parts.append(self._hint_line())
        if errors:
            parts.append(self._format_errors(errors))
        parts.append("")
        return "\n".join(parts)

    def _case_table(self, context: CaseContext) -> str:
        return render_case_table(
            context.case_instance, self.min_column_width, use_color=self.use_color
        )

    def _framing_lines(self, context: CaseContext) -> List[str]:
        """State the (wrong) conclusion and what condition would resolve it."""
        p = self.palette
        lines: List[str] = []
        if context.has_target:
            lines.append(
                p.label("Ground-truth conclusion: ")
                + p.good(repr(context.target_conclusion))
            )
        lines.append(self._current_conclusion_line(context, p))
        lines.extend(self._resolution_lines(context, p))
        return lines

    def _current_conclusion_line(self, context: CaseContext, p: Palette) -> str:
        value = repr(context.current_conclusion)
        if not context.has_target:
            styled = p.neutral(value)
        elif context.current_conclusion == context.target_conclusion:
            styled = p.good(value)
        else:
            styled = p.wrong(value)
        return p.label("Current conclusion: ") + styled

    def _resolution_lines(self, context: CaseContext, p: Palette) -> List[str]:
        """The call to action: which condition to write, and why."""
        if not context.has_current_conclusion:
            lines = [p.label("No rule fired for this case.")]
            if context.has_target:
                lines.append(
                    p.label("Write a ")
                    + p.keyword("condition")
                    + p.label(" that fires for it.")
                )
            return lines
        if not (
            context.has_target
            and context.current_conclusion != context.target_conclusion
        ):
            return []
        lines = []
        if context.trace is not None:
            lines.append(
                p.label("Apparently, the condition ")
                + p.code(format_condition(context.trace.firing_anchor))
                + p.label(" satisfies both ")
                + p.good(repr(context.target_conclusion))
                + p.label(" and ")
                + p.strong_wrong(repr(context.current_conclusion))
                + p.label(".")
            )
        lines.append(
            p.label("Write a ")
            + p.keyword("condition")
            + p.label(" that satisfies ")
            + p.good(repr(context.target_conclusion))
            + p.label(" and does not satisfy ")
            + p.strong_wrong(repr(context.current_conclusion))
        )
        return lines

    def _hint_line(self) -> str:
        """A single standing pointer to the fuller guidance behind ``%help``."""
        return self.palette.hint(
            f"Type %{HELP_MAGIC} for how to answer this case."
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
        lines.append(
            f"  Submit with {p.code('Ctrl-D')}; cancel with {p.code(f'{EXIT_NAME}()')}."
        )
        lines.append(f"  Show the rule tree with {p.code(f'%{SHOW_TREE_MAGIC}')}.")
        lines.append(f"  Show this help again with {p.code(f'%{HELP_MAGIC}')}.")
        return "\n".join(lines)

    def _build_namespace(
        self, context: CaseContext, requests: List[AnswerRequest]
    ) -> Dict[str, Any]:
        namespace = super()._build_namespace(context, requests)
        namespace[_TREE_RENDER_KEY] = lambda: self._render_tree(context)
        namespace[_HELP_TEXT_KEY] = lambda: self._help_text(context, requests)
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

        shell = _ValidatingEmbeddedShell(banner1=header, user_ns=namespace)
        shell.auto_match = True
        shell.confirm_exit = False
        shell._force_exit = False

        self._register_namespace_magic(shell, namespace, SHOW_TREE_MAGIC, _TREE_RENDER_KEY)
        self._register_namespace_magic(shell, namespace, HELP_MAGIC, _HELP_TEXT_KEY)

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
            print(text if text else f"{Fore.LIGHTBLACK_EX}(nothing to show){Style.RESET_ALL}")

        shell.register_magic_function(magic, magic_kind="line", magic_name=magic_name)
