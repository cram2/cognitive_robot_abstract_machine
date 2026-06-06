"""IPython line-magic factory for the EQL-RDR interactive expert shell.

:func:`_make_assign_exit_magic` creates action magics (``%conclusion``,
``%conditions``) that evaluate an expression, assign it to the named answer
variable, validate, and exit the embedded shell on success — all in one step.
:func:`_make_knowledge_magic` creates a read-only magic (``%knows``) that
displays backward-inference results.
"""

from __future__ import annotations

from typing_extensions import Any, Callable, Dict

from krrood.entity_query_language.rdr.rule_tree_view import format_condition

#: Magic name for setting the conclusion answer variable.
CONCLUSION_MAGIC = "conclusion"

#: Magic name for setting the conditions answer variable.
CONDITIONS_MAGIC = "conditions"

#: Magic name for backward-inference query.
BACKWARD_MAGIC = "knows"

#: Namespace key holding the RDR for the ``%knows`` magic.
_KNOWLEDGE_KEY = "__backward_knowledge__"


def _make_assign_exit_magic(
    target_name: str,
    shell: Any,
    namespace: Dict[str, Any],
    validate: Callable[[], Dict[str, str]],
    palette: Any,
) -> Callable[[str], None]:
    """Build a line-magic function that assigns, validates, and exits in one step.

    The returned callable is registered as an IPython line magic.  When the
    expert types ``%conclusion Species.mammal``, the shell calls
    ``magic("Species.mammal")``, which:

    1. ``eval``-uates the expression in the live namespace.
    2. Assigns the result to ``namespace[target_name]``.
    3. Calls ``validate()`` (returns ``{name: error_message}`` for failures).
    4. If ``target_name`` has an error: prints it and returns (shell stays open).
    5. If valid: sets ``shell._force_exit = True`` and calls ``shell.ask_exit()``.

    The plain-assignment path (``conclusion = value`` then Ctrl-D) still works
    unchanged — this magic is an optional shorthand.

    :param target_name: The namespace variable name to assign (e.g. ``"conclusion"``).
    :param shell: The :class:`~IPython.terminal.embed.InteractiveShellEmbed` instance.
    :param namespace: The shared namespace dict (mutated in place).
    :param validate: Zero-arg callable returning ``{name: error}`` for each failing answer.
    :param palette: A :class:`~krrood.entity_query_language.rdr.interactive.Palette` for
        colouring error messages.
    :return: A line-magic function ``(line: str) -> None``.
    """

    def magic(line: str) -> None:
        try:
            value = eval(line.strip(), namespace)
        except Exception as exc:
            print(palette.error(f"[error] {target_name}: {exc}"))
            return
        namespace[target_name] = value
        errors = validate()
        if target_name in errors:
            print(palette.error(f"[error] {target_name}: {errors[target_name]}"))
            return
        shell._force_exit = True
        shell.ask_exit()

    return magic


def _make_knowledge_magic(
    namespace: Dict[str, Any],
    palette: Any,
) -> Callable[[str], None]:
    """Build a line magic (``%knows <value>``) that queries backward inference.

    Reads the RDR reference from the namespace at :data:`_KNOWLEDGE_KEY`,
    evaluates the line argument in the namespace, calls
    ``rdr.what_do_we_know_about(value)``, and prints the result as a
    human-readable list of sufficient condition sets.

    :param namespace: The shell's namespace (mutated in place).
    :param palette: A :class:`~krrood.entity_query_language.rdr.interactive.Palette` for
        colouring output.
    :return: A line-magic function ``(line: str) -> None``.
    """

    def magic(line: str) -> None:
        rdr = namespace.get(_KNOWLEDGE_KEY)
        if rdr is None:
            print(palette.error("[error] No rule tree available in this session."))
            return

        if not line.strip():
            print(palette.hint("Usage: %knows <conclusion_value>"))
            return

        try:
            value = eval(line.strip(), namespace)
        except Exception as exc:
            print(palette.error(f"[error] Cannot evaluate argument: {exc}"))
            return

        knowledge = rdr.what_do_we_know_about(value)
        p = palette

        if not knowledge.is_satisfiable():
            print(
                p.label("→ ")
                + p.neutral("No rule path concludes ")
                + p.code(repr(value))
                + p.label(".")
            )
            return

        label = repr(value)
        sets = knowledge.sufficient_condition_sets
        print(
            p.label("→ ")
            + str(len(sets))
            + p.label(" sufficient condition set(s) for ")
            + p.good(label)
            + p.label(":")
        )
        for i, scs in enumerate(sets, 1):
            print()
            print(f"  {p.keyword(f'{i}.')}")
            for gc in scs.conditions:
                expr_str = format_condition(gc.expression)
                display = f"not {expr_str}" if gc.negated else expr_str
                print(f"    {p.code(display)}")

    return magic
