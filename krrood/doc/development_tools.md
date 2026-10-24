# Development Tools

krrood ships the tools its own repository is developed with, so any Python repository
can use the same conventions:

- `krrood format FILE...` formats Python files with `ruff format`, then applies
  docformatter's docstring changes only where a second ruff pass leaves them unchanged.
  docformatter can disagree with ruff about the blank lines after an attribute docstring
  that precedes a decorated definition; for such a file ruff's output is kept, so the two
  tools never oscillate.
- `krrood agents-md [--rule-sets SET ...] [--check]` generates the repository's
  `AGENTS.md` from the rules of the chosen rule sets, replacing whatever the file held.
  With `--check` it writes nothing and exits non-zero when the file differs; run that in
  CI.

Both commands exit with 1 when they modified a file, following pre-commit's convention
for formatting hooks.

## Rule sets

The rules are code. A rule set is a subclass of `RuleSet` and becomes a top-level heading
of `AGENTS.md`. A section is an enum subclassing `Section`, whose members are its rules:
each member's value is the rule's statement, or a tuple of the statement and the details
that spell it out. Every member is an `AgentRule`, so `Errors.NEVER_TRY_EXCEPT.statement`
is one of the Python rules on errors. A section names the section or rule set it is
nested under with a `parent` classproperty, so `Errors.parent` is `DesignPrinciples`,
whose `parent` is `PythonRules`. A section's heading is its class name in words, in
sentence case (`ExternalServicesAndCI` becomes "External services and CI"). A rule that
names code builds that name from the imported object with `CodeReference`, so the rule
follows a rename.

`--rule-sets` chooses the rule sets `AGENTS.md` lists (all of them by default):

- `python`: rules for any Python repository, the sections of `PythonRules`.
- `krrood`: conventions for a repository built on krrood, naming the krrood API each one
  uses (`DataclassException`, `classproperty`, `PathEnumeration`, the JSON serializer,
  `SubClassSafeGeneric`), the sections of `KrroodRules`.
- `project`: the repository's own rules, the sections of `ProjectRules` (see
  [Project rules](#project-rules)).

A repository that does not use krrood generates its file with
`--rule-sets python project`, in its pre-commit hook and in its CI check alike.

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

A repository's own rules are code too. Declare them as sections of `ProjectRules` in
`agent_rules.py` at the repository root, or in modules that one imports; `krrood
agents-md` imports that module, and every section whose `parent` is `ProjectRules`
becomes a section under "Project rules":

```python
from krrood.development_tools.agent_rule import ProjectRules, Section
from krrood.ormatic.utils import classproperty


class Deployment(Section):
    @classproperty
    def parent(cls) -> type[Section]:
        return ProjectRules

    DEPLOY_FROM_TAGS = "Deploy only tagged commits"
    TAG_RELEASES = "Tag every release", ("Use semantic versions", "Sign the tag")
```

## Changing a rule

Never edit `AGENTS.md`: the comment above each section's rules names the class and
module they come from. Find the member by the rule's text and change it there.

- A repository's own rule is in a `ProjectRules` section in its `agent_rules.py` or a
  module it imports.
- A shared rule lives under `krrood/development_tools/rules/` (in `krrood/src` of
  cognitive_robot_abstract_machine). A rule that names krrood's API is in a
  `KrroodRules` section in `krrood_rules.py`; every other rule is in a `PythonRules`
  section in `python_rules.py`, in the order they appear.

After changing a rule, run `krrood agents-md` (the pre-commit hook does this) and commit
the regenerated `AGENTS.md` with the change; CI fails when the file is stale or was
edited by hand. A shared rule reaches other repositories with the next krrood release.

To adopt the tools in a repository with an `AGENTS.md`, first move its rules into
`agent_rules.py` as `ProjectRules` sections, then run `krrood agents-md`, which replaces
the file.
