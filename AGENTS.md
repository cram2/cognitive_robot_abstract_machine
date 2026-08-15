# Code Quality Rules

## Avoid Behaviour
- Avoid using global variables
- Avoid accessing any ormatic_interface.py files. if there are issues regarding the ormatic interface run the script `scripts/regenerate_all_orm.py`. If it does not fix the issue, consider consulting the developer.
- ormatic_interface.py files are committed empty on purpose: CI regenerates them for test runs only, and the `empty-ormatic-interface` pre-commit hook always truncates them back to empty before a commit is created. Never stage real generated content in them or work around the hook. `scripts/regenerate_all_orm.py` also marks the files it regenerates with git's skip-worktree bit (see `scripts/protect_generated_orm_interfaces.py`) so a locally regenerated copy is never proposed for staging in the first place; never clear that bit to force one into a commit.
- Avoid using mutable objects as default arguments
- If you are unsure why something was done or why specific numbers were chosen, ask the developer instead of inventing the reason and writing it as a comment.
- Never comment on or modify pull requests on the upstream `cram2/cognitive_robot_abstract_machine` repository. You may only do so when working in a fork and the user has explicitly allowed it - either through existing personal notes/instructions, or by asking the user first and having them accept.

## Testing
- If you need to run tests, execute them with pytest
- Reuse existing fixtures found in conftest.py
- Always use a test-driven development approach. For example for bugs, always prove a bug by adding a meaningful, failing test first, before then fixing it
- When fixing failing tests, never modify the test itself
- All new features and fixes must be covered by tests
- Name test classes (and the mimic classes used by tests) after the pattern or behaviour they exercise, not after the concrete external class they happen to stand in for
- Make assertions as specific as possible: when the correct expected value can be determined, assert equality to that value rather than only a weaker check such as not-None or not-empty
- Assert against the definition rather than a copy of it: compare to the enum member, the named constant, or the value read from the fixture the code under test consumed. Where a type distinguishes the case, assert the type - a distinct exception class or enum member - instead of matching on message text. A literal retyped into the test is a second copy of the thing the test exists to check, and it keeps passing when the original changes
- Keep each test focused on the one behaviour it names: assert exactly the values that behaviour determines, and do not also pin down incidental output a change unrelated to that behaviour could alter (for example, an unrelated wording tweak to an error message a test isn't about). Prefer deriving an expected value from the same production code that computes it (e.g. by calling the lower-level function under test and reusing its result) over hardcoding a second literal copy of output another test already asserts exactly — a hardcoded copy duplicates coverage and turns one wording change into two unrelated test failures. Tests should be separable and independent, each failing only for its own reason.
- CI safety: All added tests must be part of the CI suite, but only need to execute there if they can run without live external calls or missing credentials — tests requiring unavailable credentials must be skipped (or removed if new), not left to break the pipeline.
- Credentials: Any test requiring credentials to run in CI must be pre-approved by the user and have those credentials available in CI; otherwise it must be skipped there.
- No inline snippets: Code snippets must live in separate files with the correct file type, imported or read into the test rather than embedded as strings.
- Mock over live calls: Tests for code depending on external APIs should mock those APIs instead of calling them live, except when an API call is needed to download a dataset required for other tests to run.
- Live API tests: You may add tests that hit external APIs directly, but they must have skip conditions so they don't run in CI, and must be paired with an equivalent mock-based test that does run in CI.

## Code Style
- Divide a file into logical sections with `# %% <short description>` comment headers (e.g. `# %% same-noun disambiguation`), not decorative box-drawing dividers. Applies to source files as well as test files
- Create classes instead of using too many primitives. If a return type is always repeated, consider whether a dedicated class or type alias would convey more meaningful information
- Minimize duplication of code. Avoid placing methods in catch-all files like `utils.py`: prefer moving them onto a sensible class that owns the behaviour
- Comments must be meaningful and adhere to DRY; remove redundant or restating comments
- Do not wrap attribute access in try-except blocks
- Always access attributes via ".", never via getattr
- Use existing packages whenever possible
- Always use dataclasses

### Naming
- Names must be technically correct, simple and descriptive, in that order. Correct first: a name that describes the thing inaccurately is worse than a vague one, because a reader who trusts it stops reading. Then the simplest wording that stays correct
- Minimize jargon. Prefer the plain word every reader already knows over the specialist, metaphorical or in-house one, and reserve a technical term for where it is genuinely the precise word - not as shorthand between the people who happen to have been in the discussion. Jargon is a lookup the reader has to perform, and it is only worth it when the plain wording would be wrong
- Do not use abbreviations in variable names, methods, classes, or any other identifiers
- Use short but descriptive names: a name says *what* a thing is or does, never *how* it does it or *when* it runs
- Name the thing, not the layer or mechanism it is built on
- Avoid generic words that would fit anything. Use the plain technical word for what the thing actually is
- A name whose meaning has to be looked up elsewhere is the wrong name. Do not adopt another system's vocabulary as an identifier of ours; name the thing for what it is here, and if the foreign shape still needs explaining, explain it in the docstring rather than encoding it in the name
- Methods are verb phrases for what they do; classes and attributes are noun phrases for what they are. A field is named for its subject, not for the shape of the value it happens to hold
- One operation, one name, throughout a module: a second name for the same operation reads as a second operation. Where callers depend on that shared name, formalize it - a base class or a protocol declaring the method - rather than leaving it a convention every class is trusted to have followed; a convention breaks only once something happens to call the one class that spelled it differently
- Name an enum member for the situation it means, not for the function it dispatches to or the wording it renders - the implementation moves and the member should not have to
- Do not repeat the enclosing type's name in its members, and do not repeat the same word twice within one name
- Where the domain or file format already has a word for something, use that word
- Never take an identifier the language or something already in scope binds - `Enum` reserves `name`, a parameter called `field` shadows `dataclasses.field`, and a method is shadowed by a field of the same name. These fail at runtime or silently, not at import
- A rename is finished only when every reader of the old name reads the new one, docstrings and comments included, and the tests pass. A mechanical rename across a file is exactly where a method and a field converge on one name
- When no honest specific name exists, suspect the code rather than your vocabulary. A thing that can only be described vaguely usually has no single subject - it is a container holding whatever its caller passed, or a function doing two jobs - and the fix is to remove it, not to keep hunting for a better word

## Imports
- Imports should always be absolute
- Exception: within tests, importing another test module (for example a shared mimic or fixture from the test datasets) must use a relative import
- Imports should always be global (top of module), except in very special cases (for example ORM interface imports)
- Use stdlib type hints where possible, and for others use typing_extensions instead of typing
- Whenever you would wrap types in strings for deferred resolution, use `from __future__ import annotations` instead.
- use TYPE_CHECKING guard for type-only imports
- `krrood` must stay self-contained: never import from another workspace package into `krrood` (this includes its tests under `test/krrood_test`). The only permitted exceptions are `random_events` and `probabilistic_model`, because `krrood`'s own source already depends on them. In particular, do not import from `coraplex`, `semantic_digital_twin`, `giskardpy`, `physics_simulators`, `robokudo`, or `experiments` inside `krrood`.
- When a `krrood` test needs to exercise behaviour that another package triggers, mimic the relevant classes and patterns inside the `krrood` test datasets (`test/krrood_test/dataset`) and test against those mimics. Keep the test in `krrood`; do not move it to another package and do not depend on another package to reproduce the scenario.
- Mimic classes in the `krrood` test datasets must never import directly from another workspace package either; the only packages they may import from are the ones `krrood`'s source already imports (`random_events`, `probabilistic_model`) plus `krrood` itself.

## Design Principles
- Focus on strictly object oriented design
- Always apply the SOLID principles of object-oriented programming
  - Single Responsibility: Each class/method does one thing. If a method has cyclomatic complexity in the hundreds, refactor.
  - Open/Closed: The code should be open for extension, but closed for modification.
  - Liskov Substitution: Subtypes must be substitutable for their base types without breaking behaviour.
  - Interface Segregation: Prefer many small interfaces over one large one.
  - Dependency Inversion: Depend on abstractions, not concrete implementations.
- Code should be modular and decoupled
- Create meaningful custom exceptions
- Eliminate YAGNI smells
- Make interfaces hard to misuse
- Reduce nesting and reduce complexity:
  - The main branch of a function should hold the main output with the biggest compute; alternative outputs should be realized via guard clauses beforehand
  - When dealing with nested if statements and branching methods, use guard clauses to reduce nesting by inverting conditions and returning early
- Dont use try except blocks, programs in illegal states should raise appropriate exceptions.
- Prefer structured data over bare strings, hardcoded values, and meaningless numbers. This is the default, not a preference to weigh: reach for the structured form first and justify the literal, never the other way round.
  - Never hardcode a string that names a fixed thing - a payload key, a state, a label, a filename, an environment variable, a command flag, a status. Give it a `StrEnum` member and use that. A value spelled in two places has no single source to rename, and nothing fails when the two drift apart.
  - Replace a magic number with a named constant or an enum member. A bare literal that carries meaning is unreadable where it is used and unsearchable everywhere else.
  - For JSON our own classes round-trip, reuse `krrood.adapters.json_serializer.SubclassJSONSerializer` rather than hand-writing `to_json`/`from_json` - it already resolves the concrete subclass from the stored type name.
  - For data whose shape someone else controls - an API response, a configuration file - that serializer does not apply, since the payload carries no type of ours. Mirror the structure in dataclasses instead and parse into them the same way, with a `from_json` classmethod doing the reading, so the field names and the access path into the payload are written once rather than at every use site.
  - Replace a tuple whose positions carry meaning with a dataclass, or with an enum when the positions are a fixed set of alternatives rather than fields, so the parts are named rather than counted.
  - Keep a long literal document - a query, a template, a schema - in a file of its own type and read it in, rather than embedding it as a string.
- If there are methods that are never used outside of tests, consult the developer if they can be removed.

## Type Hints
- Classes and methods should always have accurate type hints (including `Any`) where applicable

## Documentation
- Classes and methods should always have meaningful, non-trivial documentation
- Every field/attribute must be documented with its own docstring placed directly below the field, not described in the class docstring
- Write docstrings in ReStructuredText format
- Write docstrings that explain what the function does and not how it does it
- Keep docstrings short and concise
- Use Sphinx directives (for example `..note::`, `..warning::`, and `:func:`) where appropriate
- Do not use all-caps words for emphasis in docstrings or comments; use RST emphasis (`*word*`) if emphasis is genuinely needed
- Do not create type information for docstrings (type hints already convey this)
- Do not name a function/class's current callers or consumers in its own docstring (e.g. "used by
  X and Y"); document what it does and its contract, not who happens to use it today — that
  reference goes stale the moment a caller changes and misleads a future reader into thinking the
  list is exhaustive or load-bearing
- Docstrings must be short and to the point: state what the code does, not a conversation about
  it. Do not compare against a rejected/alternative design, narrate the review or implementation
  history, or explain what would happen under a hypothetical design that was not chosen
- Do not use ALL-CAPS words for emphasis in docstrings or comments; use RST emphasis (`*word*`)
  instead. This does not apply to genuine identifiers, acronyms, or enum/constant names (e.g.
  `UUID`, `WHERE`, `Definiteness.DEFINITE`)
- Always run `scripts/format_docstrings.py` (black + docformatter) on modified files

## Domain-Specific Conventions
- When dealing with spatial types and connections, adhere to the style guide documented in `semantic_digital_twin/doc/style_guide.md`

## Version Control
- Commits must be authored in the name of the human user running the tool, using their own configured git `user.name` and `user.email`. Never author or amend a commit as an assistant/agent identity.
- Do not attribute authorship or co-authorship to an assistant: no `Co-Authored-By:` trailer for Claude or any assistant, and no `noreply@anthropic.com` (or similar) as author or committer. The commit's authorship reflects the person responsible for it.
- It is fine — and encouraged — to acknowledge assistant help in the commit message body with a short plain line, for example `Made with the help of Claude`. Keep it a note, not an author/co-author trailer.
- This applies to every contributor and every tool.

## Misc
- If you find a package that could be replaced by a more powerful one, let us know
- Always use the Python interpreter that is set as the current project interpreter for running tests and commands
