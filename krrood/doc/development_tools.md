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

Both commands exit with 1 when they modified a file, following pre-commit's convention
for formatting hooks.

## Rules

The rules are code. A section is an enum subclassing `Section`, named for one topic
followed by `Rules`. Its members are its rules: each names one rule, and its value is
the rule's statement, or the statement followed by the details that spell it out.
Every member is an `AgentRule`, so `ExceptionModellingRules.NEVER_TRY_EXCEPT.statement`
is one of the rules on modelling exceptions. A nested section names the section it is
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

## Installation

The formatting tools are an optional dependency, pinned so every repository formats
alike:

```bash
pip install "krrood[development-tools]"
```

## Use with pre-commit

pre-commit installs krrood into the hook's own environment:

```yaml
repos:
  - repo: local
    hooks:
      - id: krrood-format
        name: Format Python (ruff, then docformatter)
        entry: krrood format
        language: python
        additional_dependencies: ["krrood[development-tools]==<version>"]
        types: [python]
      - id: krrood-agents-md
        name: Update the rules in AGENTS.md
        entry: krrood agents-md
        language: python
        additional_dependencies: ["krrood[development-tools]==<version>"]
        pass_filenames: false
        always_run: true
```

To use rules that are not released yet, replace the requirement with a commit of
cognitive_robot_abstract_machine:
`krrood[development-tools] @ git+https://github.com/cram2/cognitive_robot_abstract_machine@<commit>#subdirectory=krrood`.

The same hooks are published in the `.pre-commit-hooks.yaml` at the root of
[cognitive_robot_abstract_machine](https://github.com/cram2/cognitive_robot_abstract_machine).
They run the `krrood` command already installed in your development environment
(`language: system`), because pre-commit would otherwise install that whole repository
rather than krrood.

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

## Changing a rule

Never edit `AGENTS.md`: the comment above each section's rules names the class and
module they come from. Find the member by the rule's text and change it there.

- A repository's own rule is in its `agent_rules.py`, or in a module of its
  `agent_rules` package.
- A shared rule lives in the module for its top-level section in
  `krrood/development_tools/agents/shipped_rules/` (in `krrood/src` of
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
