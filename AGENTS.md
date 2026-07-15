# Code Quality Rules

## Avoid Behaviour
- Avoid using global variables
- Avoid accessing any ormatic_interface.py files. if there are issues regarding the ormatic interface run the script `scripts/regenerate_all_orm.py`. If it does not fix the issue, consider consulting the developer.
- Avoid using mutable objects as default arguments
- If you are unsure why something was done or why specific numbers were chosen, ask the developer instead of inventing the reason and writing it as a comment.

## Testing
- If you need to run tests, execute them with pytest
- Reuse existing fixtures found in conftest.py
- Always use a test-driven development approach. For example for bugs, always prove a bug by adding a meaningful, failing test first, before then fixing it
- When fixing failing tests, never modify the test itself
- All new features and fixes must be covered by tests
- Name test classes (and the mimic classes used by tests) after the pattern or behaviour they exercise, not after the concrete external class they happen to stand in for

## Code Style
- Do not use abbreviations in variable names, methods, classes, or any other identifiers
- Method and class names should be concise and descriptive: they should tell *what* they do, not *how* they do it
- Create classes instead of using too many primitives. If a return type is always repeated, consider whether a dedicated class or type alias would convey more meaningful information
- Minimize duplication of code. Avoid placing methods in catch-all files like `utils.py`: prefer moving them onto a sensible class that owns the behaviour
- Comments must be meaningful and adhere to DRY; remove redundant or restating comments
- Do not wrap attribute access in try-except blocks
- Always access attributes via ".", never via getattr
- Use existing packages whenever possible
- Always use dataclasses
- Use short but descriptive names

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
- Instead of passing around strings, use enums instead
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
- Do not create type information for docstrings (type hints already convey this)
- Always run `docformatter` on modified files

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
