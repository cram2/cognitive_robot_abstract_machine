# basstler

The workflow tooling for this repository: the Python behind the stacked-pull-request
tooling, the plan dashboards, the personal-notes hooks and the upstream review reader.

The name is the German word for someone who builds things themselves, and shares its
first letters with the surname of the person who wrote it, Bassiouny.

## Importing it

`basstler` is a plain top-level directory on the repository root, so it imports from there
with no installation step - which is what a cloud session running on a fresh clone with
no `pip` step needs.

```python
from basstler.stack import load_configuration
```

The `pyproject.toml` beside this file is for installing it somewhere that is not such a
checkout:

```bash
pip install ./basstler                # the modules that need nothing but the standard library
pip install './basstler[rendering]'   # plus the dashboard build's Jinja2/markdown/nh3
```

## Running it

Every entry point is run as a module rather than by its file path:

```bash
python3 -m basstler.stack configuration
python3 -m basstler.maintenance run-report --json
python3 -m basstler.build_dashboard --help
```

By path, the interpreter puts *this* directory on `sys.path` in place of the repository
root, and a module's imports of its siblings stop resolving. The shell entry points under
`.claude/hooks/` and the skills that call them all use `python3 -m "${SOME_MODULE}"`, with
the module named once in `.claude/hooks/resolve-personal-notes-config.sh`.

## The dependencies install themselves

**Every session start installs whatever of this package's declared dependencies your
clone is missing, without asking.** They are declared once, in `basstler/pyproject.toml`'s
`[project] dependencies`, the way every package in this repository declares them, and
`basstler.dependencies` is what reads that list back. `.claude/hooks/session-start.sh` does
the installing, so no module ever has to work around a dependency that is not there, and
nobody has to notice a missing one and run `pip` by hand.

Three things bound it, and they are worth knowing because it writes to your Python
environment:

- **Only for someone who has already set the tooling up.** The hook stops before this
  point when it cannot reach your personal-notes branch, so a clone that has never run
  `/setup-personal-notes` installs nothing at all.
- **Only what is missing.** The usual start looks the declaration up and runs no
  installer, which is what makes doing it every time affordable.
- **Never fatal.** No `pip`, no network, or an externally managed environment that refuses
  the write: the summary's `dependencies:` line says so, naming what to install, and the
  rest of the run carries on. `pip install ./basstler` installs them all when that happens.

An Actions runner reaches no session hook, so a workflow that runs a module installs the
dependencies in a step of its own -
`test/basstler_test/test_package_contract.py` finds every such workflow and checks that it
does.

## What stays outside the package

`SKILL.md` files, `settings.json`, the bash entry points under `.claude/hooks/` and the
plan-dashboard `example/` all stay under `.claude/`, because Claude Code and its readers
find them by path.
