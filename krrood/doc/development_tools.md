# Development Tools

krrood ships the tools its own repository is developed with, so any Python repository
can use the same conventions:

- `krrood format FILE...` formats Python files with `ruff format`, then applies
  docformatter's docstring changes only where a second ruff pass leaves them unchanged.
  docformatter can disagree with ruff about the blank lines after an attribute docstring
  that precedes a decorated definition; for such a file ruff's output is kept, so the two
  tools never oscillate.
- `krrood agents-md [--check]` generates the repository's `AGENTS.md` from the rules
  krrood ships and the repository's own, replacing whatever the file held.
  With `--check` it writes nothing and exits non-zero when the file differs; run that in
  CI.
- `krrood skills [--check]` generates the `.claude/skills/<name>/SKILL.md` of every
  skill the repository declares, replacing whatever those files held. `--check` works
  as for `agents-md`.
- `krrood issue-forms [--check]` generates the `.github/ISSUE_TEMPLATE/<name>.yml` of
  every GitHub issue form krrood ships and the repository declares, replacing whatever those files held.
  `--check` works as for `agents-md`.
- `krrood issue-answer --form NAME --field IDENTIFIER` reads the body of an issue opened
  with an issue form from standard input and prints its answer to one field, code
  without its fence; it exits with 1 when the issue leaves the field empty.
- `krrood check [--all-lines | --since REVISION] FILE...` reports the code that breaks a
  rule a checker enforces; `AGENTS.md` marks such a rule "(checked by `krrood check`)".
  Only the lines the staged change (`git diff --cached`) adds or changes are checked,
  because existing code has many violations; `--all-lines` checks every line.
  `--since REVISION` checks the lines the working tree changes since `REVISION`, staged
  or not; without files it checks every tracked Python file changed since then.

`format`, `agents-md`, `skills` and `issue-forms` exit with 1 when they modified a file, and `check` when it
found a violation, following pre-commit's convention for hooks.

## Rules

The rules are code. A section is an enum subclassing `Section`, named for one topic
followed by `Rules`. Its members are its rules: each names one rule, and its value is
the rule's statement, or the statement followed by the details that spell it out.
Every member is an `AgentRule`, so `ExceptionModellingRules.NEVER_TRY_EXCEPT.statement`
is one of the rules on modelling exceptions. A rule can carry a checker, a `CodeRuleChecker` that finds the code breaking it; give it
as the third element of the member's value. A rule that lint rules of ruff enforce takes a
`RuffRuleChecker` naming them, with the settings they need and whether the rule covers
tests, the code they test or both; it runs ruff isolated from any configuration file, so
every repository checks alike. When a script can check only part of a
rule, split the rule so the checkable part carries the checker. A nested section names the section it is
part of by overriding the `parent` classproperty. A section's heading is its topic in
words, in sentence case (`ExternalServicesAndCIRules` becomes "External services and
CI"). A rule that names code builds that name from the imported object with
`CodeReference`, so the rule follows a rename:

```python
from krrood.development_tools.agents.rule import CodeReference, Section


class ExceptionModellingRules(Section):
    @classproperty
    def parent(cls) -> type[Section]:
        return DesignPrinciplesRules

    NEVER_TRY_EXCEPT = "Never use try-except; ..."
    SUBCLASS_THE_KRROOD_BASE_CLASS = (
        f"Custom exceptions subclass {CodeReference(DataclassException).path}, ..."
    )
```

The shipped rules live in the package `krrood.development_tools.agents.shipped_rules`,
one module per top-level section, holding that section and the sections nested in it.
`krrood agents-md` finds every section defined in a module of that package; a section
a module only imports, or one defined anywhere else, is never listed. Top-level
sections appear in alphabetical order of their headings, and nested sections in the
order they are written.

## Core boundary

A distribution can declare a light core that a plain `pip install` must be able to load,
in its `pyproject.toml`:

```toml
[tool.krrood.core-boundary]
# The packages and modules of core; every other module of the distribution is not core.
core = ["my_package.core"]
# The distributions core may load, with everything they require; without it, core may
# load only the standard library.
dependencies = ["typing-extensions"]
```

`krrood check` then reports every import that a core module runs when it is loaded
(outside functions and `if TYPE_CHECKING:`) and that loads a module outside core, or a
distribution outside `dependencies` and what they require. It reads the whole
distribution's import graph, so for a module outside core it also names the
distributions that module loads in turn, however long the chain. It maps an imported
package to its distribution through the installed packages, so it runs in the
distribution's development environment. Core may import anything inside a function.

Like every checker, it checks only the lines a commit changes, so existing crossings do
not block a commit and a new one cannot slip in; `krrood check --all-lines` lists all of
them.

## Excluding files

A repository keeps files out of `krrood check` in its `pyproject.toml`, wherever the
check runs:

```toml
[tool.krrood.check]
# Regular expressions, searched for in each path relative to the repository root.
exclude = ['(^|/)generated_module\.py$', '^test/dataset/']
```

## Installation

The development tools are the `krrood-development-tools` distribution, which pins the
formatters so every repository formats alike. They need Python 3.11 or later:

```bash
pip install krrood-development-tools
```

## Use with pre-commit

The hooks are published in
[`.pre-commit-hooks.yaml`](https://github.com/cram2/cognitive_robot_abstract_machine/blob/main/.pre-commit-hooks.yaml)
at the root of cognitive_robot_abstract_machine: `krrood-format`, `krrood-agents-md`,
`krrood-skills`, `krrood-issue-forms` and `krrood-check`. Add that repository to your
`.pre-commit-config.yaml` and list the hook ids you want.

The hooks run the `krrood` command already installed in your development environment
(`language: system`), because pre-commit would otherwise install that whole repository
rather than krrood. `krrood-check` needs that environment anyway: the core boundary check
reads the installed packages.

The same check runs in CI too: `krrood check --since` with the base of a pull request
checks the lines it changes.

To use rules that are not released yet, install krrood from a commit of
cognitive_robot_abstract_machine:
`pip install "krrood-development-tools @ git+https://github.com/cram2/cognitive_robot_abstract_machine@<commit>#subdirectory=krrood/krrood-development-tools"`.

## Project rules

A repository's own rules are code too. Declare them in sections in an `agent_rules.py`
module at the repository root, or in the modules of an `agent_rules` package there.
`krrood agents-md` imports it and lists the sections defined in it after the shipped
ones, found and ordered the same way:

```python
from krrood.development_tools.agents.rule import Section


class DeploymentRules(Section):
    DEPLOY_FROM_TAGS = "Deploy only tagged commits"
    TAG_RELEASES = "Tag every release", ("Use semantic versions", "Sign the tag")
```

## Skills

A skill is code as well: a class subclassing `Skill`, named for the skill followed by
`Skill`, in an `agent_skills.py` module at the repository root or in the modules of an
`agent_skills` package there. `krrood skills` writes each skill defined in it to the
path its `path` classproperty gives, relative to the repository root, usually
`.claude/skills/<name>/SKILL.md`. The name is the class name's words in kebab case
(`CRAMLiveSessionSkill` becomes `cram-live-session`). The skill's other classproperties
give the front matter (`description`, `allowed_tools`), the paragraphs below its title
(`introduction`) and its sections, which are `Section`s like the rules and are written
in the order `sections` lists them, one heading level below the title:

```python
from pathlib import Path

from krrood.development_tools.agents.rule import Section
from krrood.development_tools.agents.skill import AgentTool, Skill


class LoopRules(Section):
    ONE_STEP_PER_INSTRUCTION = "Do what the developer asked, then stop and wait"


class LiveSessionSkill(Skill):
    @classproperty
    def path(cls) -> Path:
        return Path(".claude/skills/live-session/SKILL.md")

    @classproperty
    def description(cls) -> str:
        return "Drive a simulated robot step by step. Use when ..."

    @classproperty
    def allowed_tools(cls) -> list[AgentTool]:
        return [AgentTool.BASH, AgentTool.READ]

    @classproperty
    def introduction(cls) -> str:
        return "The developer says what the robot does next. ..."

    @classproperty
    def sections(cls) -> list[type[Section]]:
        return [LoopRules]
```

Never edit a generated `SKILL.md`; the comment below its front matter names the class
and module it comes from. Skills written by hand in `.claude/skills` are left alone.

## Issue forms

A GitHub issue form is a class subclassing `IssueForm`, named for the form followed by
`IssueForm`. `krrood issue-forms` writes every form krrood ships, and every form defined
in an `issue_forms.py` module at the repository root or in the modules of an
`issue_forms` package there, to `.github/ISSUE_TEMPLATE/<name>.yml`, the name being the
class name's words in kebab case. The form's title is those words in sentence case; its
classproperties give the `description` and the `IssueType` set on every issue opened
with it, which the organisation has to define, and each classproperty returning a
`TextArea` is a field of the form, in the order they are defined. The bug report krrood
ships,
`BugReportIssueForm` in `krrood/development_tools/issue_forms/bug_report.py`, shows
each of them.

`krrood issue-answer` finds a form by its `name`, the file name without `.yml`, and a
field by its `identifier`, so a workflow can act on an answer, for example run the
script of a bug report.

Never edit a generated issue form; the comment at its top names the class and module it
comes from. Issue forms written by hand in `.github/ISSUE_TEMPLATE` are left alone.

## Changing a rule

Never edit `AGENTS.md`: the comment above each section's rules names the class and
module they come from. Find the member by the rule's text and change it there.

- A repository's own rule is in its `agent_rules.py`, or in a module of its
  `agent_rules` package.
- A shared rule lives in the module for its top-level section in
  `krrood/development_tools/agents/shipped_rules/` (in `krrood/krrood-development-tools/src` of
  cognitive_robot_abstract_machine).
- A section is named for its topic, broadly enough that later rules on that topic fit
  in it (`SerializationRules`, not `JSONSerializationRules`). Each topic is one section, so
  sections never share a title. Add a rule to the section whose topic it is; when no
  section fits, declare one, as a new module for a new top-level section.

After changing a rule, run `krrood agents-md` (the pre-commit hook does this) and commit
the regenerated `AGENTS.md` with the change; CI fails when the file is stale or was
edited by hand. A shared rule reaches other repositories with the next krrood release.

To adopt the tools in a repository with an `AGENTS.md`, first move its rules into
`agent_rules.py`, then run `krrood agents-md`, which replaces the file.
