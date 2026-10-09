# Code Quality Rules

## Applying These Rules
- Existing code that breaks these rules stays as it is unless a change touches it or the user asks for the cleanup. When you modify a function or class, bring that function or class in line with these rules as part of the change, and leave code the change does not otherwise modify alone. If the cleanup renames a method and so changes the API, ask the user first
- When a rule seems wrong for the case at hand - a failing test that is itself wrong, an exception that has to be caught, a value that genuinely has to live at module level - ask the user before departing from it; never decide the exception on your own

## Ask the Developer
- When you are unsure why something was done or why a specific number was chosen, ask; never invent a reason and write it down as a comment
- When a method is never used outside of tests, ask whether it can be removed
- When regenerating the ORM interfaces does not fix an ORM problem, ask
- Before adding a new dependency, ask

## Generated ORM Interfaces
- `ormatic_interface.py` files are generated, never written, and git-ignored (see `.gitignore`). Never edit them, and never read them to debug an ORM problem; regenerate them with `scripts/regenerate_all_orm.py` instead
- The test suite builds them for its runs; a local checkout builds them with `scripts/regenerate_all_orm.py`
- Never track one again: git refuses to overwrite a tracked path a checkout has generated its own copy of, which makes every branch switch fail

## Testing

### Writing Tests
- Write tests with pytest: plain test functions or classes, pytest fixtures, `pytest.raises` and plain `assert`; never `unittest.TestCase` or its assertion methods
- Reuse the existing fixtures in `conftest.py`
- Work test-driven: prove a bug with a meaningful, failing test before fixing it. Every new feature and fix is covered by tests
- When fixing a failing test, never modify the test itself
- Name test classes, and the mimic classes tests use, after the pattern or behaviour they exercise, not after the external class they stand in for
- Keep code snippets in separate files of the correct type and import or read them into the test; never embed them as strings

### Assertions
- Assert equality to the expected value whenever it can be determined, never only a weaker check such as not-None or not-empty
- Assert against the definition, not a retyped copy of it: the enum member, the `classproperty`, or the value from the fixture the code under test consumed. Where a type distinguishes the case - a distinct exception class, an enum member - assert the type instead of matching message text. A retyped literal keeps passing when the original changes
- Each test checks the one behaviour it names and fails only for its own reason: assert exactly the values that behaviour determines, not incidental output such as the wording of an error message the test is not about. Where another test already asserts a value exactly, derive the expected value from the production code that computes it instead of hardcoding a second copy. Tests are independent of each other

### External Services and CI
- Every added test is part of the CI suite. A test that needs live external calls or credentials CI does not have is skipped there (or removed, if new) so it cannot break the pipeline
- A test that needs credentials in CI must be approved by the user and have those credentials available in CI; otherwise it is skipped there
- Mock external APIs; call one live only to download a dataset other tests need
- A test that calls an external API live must have a skip condition so it does not run in CI, and must be paired with an equivalent mocked test that does

### Running Tests
- Run tests with pytest
- Never run the whole test suite in one invocation: run one package or a few test directories at a time, and wait for each run to finish before starting the next
- Make sure a test run can never exhaust the machine's working memory: check the free memory before starting, cap the run's memory so the run is killed rather than the machine, and account for other test runs already going on the same machine
- Pass `--orm-build never`, and regenerate the ORM interfaces explicitly with `scripts/regenerate_all_orm.py` when a change needs it

## Code Style
- Always use dataclasses
- Divide every file, source and test, into sections with `# %% <short description>` headers (e.g. `# %% same-noun disambiguation`), never decorative box-drawing dividers
- A group of primitives that travels together, or a return type that keeps being repeated, becomes a dataclass
- Never duplicate code. Never put methods in a catch-all module such as `utils.py`; move them onto the class that owns the behaviour
- Access attributes with `.`, never with `getattr`, and never wrap attribute access in try-except
- Never use mutable objects as default arguments
- Never reimplement what the codebase or an existing dependency already provides. When searching for it, never cut off the output of grep or other searches; narrow the search instead

### Naming
- Names are technically correct, simple and descriptive, in that order. An inaccurate name is worse than a vague one, because a reader who trusts it stops reading
- A name stays correct and understandable when read on its own, without its class or a keyword argument next to it - after `value = instance.attribute`, in a log line, in a traceback, or when passed on positionally. A `Pipe` field `size` says nothing once it leaves the class; `inner_diameter` does. A generic word that would fit anything never passes this test
- Use the plain word every reader already knows. Use a specialist, metaphorical or in-house term only where it is genuinely the precise word, never as shorthand between the people who were in the discussion
- Where the domain or file format already has a word for something, use that word
- Never abbreviate an identifier
- A name says *what* a thing is or does - never *how* it does it, *when* it runs, or the layer or mechanism it is built on. Keep it short
- A name whose meaning has to be looked up elsewhere is wrong. Never adopt another system's vocabulary as an identifier of ours: name the thing for what it is here, and explain a foreign shape in the docstring
- Methods are verb phrases for what they do; classes and attributes are noun phrases for what they are. Name a field for its subject, not for the shape of its value
- One operation has one name throughout a module. Where callers depend on that name, declare it in a base class instead of leaving it a convention each class is trusted to follow
- Name an enum member for the situation it means, not for the function it dispatches to or the text it renders
- Never repeat the enclosing type's name in its members, or the same word twice within one name
- Never take an identifier the language or something in scope already binds: `Enum` reserves `name`, a parameter `field` shadows `dataclasses.field`, a field shadows a method of the same name. These fail at runtime or silently, not at import
- A rename is finished only when every reader of the old name, docstrings and comments included, uses the new one and the tests pass. A mechanical rename across a file is exactly where a method and a field converge on one name
- When no honest specific name exists, suspect the code, not your vocabulary: a thing that can only be described vaguely usually has no single subject (a container for whatever its caller passed, a function doing two jobs). Remove it instead of hunting for a better word

## Imports
- Imports are absolute. Exception: a test imports another test module (a shared mimic or fixture from the test datasets) relatively
- Imports go at the top of the module; fix an import cycle by restructuring the modules, never with a local import. If an ORM import creates a cycle, ask the user if a local import is fine.
- Guard type-only imports with `TYPE_CHECKING`

### krrood Isolation
- `krrood`, including its tests under `test/krrood_test`, never imports another workspace package - in particular not `coraplex`, `semantic_digital_twin`, `giskardpy`, `physics_simulators`, `robokudo` or `experiments`. The only exceptions are `random_events` and `probabilistic_model`, which `krrood`'s source already depends on
- When a `krrood` test needs behaviour another package triggers, mimic the relevant classes and patterns in `test/krrood_test/dataset` and test against the mimics. Keep the test in `krrood` and do not depend on another package to reproduce the scenario
- Those mimic classes import only `krrood`, `random_events` and `probabilistic_model`

## Design Principles
- Use strictly object-oriented design and always apply the SOLID principles:
  - Single Responsibility: each class and method does one thing. If a method has cyclomatic complexity in the hundreds, refactor
  - Open/Closed: open for extension, closed for modification
  - Liskov Substitution: subtypes are substitutable for their base types without breaking behaviour
  - Interface Segregation: many small interfaces over one large one. An interface is an explicit superclass or mixin, never a `Protocol`
  - Dependency Inversion: depend on abstractions, not concrete implementations
- Keep code modular and decoupled
- Eliminate YAGNI smells
- Make interfaces hard to misuse

### Methods
- Every method hides what actually runs from the reader, so extract one only when its abstraction is worth more than reading the executed code in place
- Keep every statement in a method on the same level of abstraction: a method either strings together named steps or carries out one step's detail, never both
- Never write a method that only forwards to or renames another call. A one-line method is allowed only when it implements an abstract method or a `classproperty`, caches its result, or removes duplication - and in the last case, first try to abstract the duplicated code itself
- A method with a single caller must earn its place under these rules; otherwise inline it
- Reduce nesting and complexity with guard clauses: handle alternative outputs first by inverting conditions and returning early, so the main branch holds the main output and the biggest compute. A single level of abstraction never justifies deep nesting; split the method instead

### Errors
- Never use try-except; a program in an illegal state raises an appropriate exception
- Create meaningful custom exceptions as dataclasses subclassing `krrood.exceptions.DataclassException`, implementing its `error_message` and `suggest_correction`
- Put each exception in the `exceptions.py` closest to the code that raises it:
  - An exception raised only within one subpackage goes in that subpackage's `exceptions.py`; create the file if the subpackage has none
  - An exception raised across several subpackages goes in the `exceptions.py` of their nearest common package
- Never use `assert` outside tests: Python drops it when run with `-O`, so the check silently disappears. Raise a custom exception instead

### Constants and Class-Level Values
- Never use global variables, module-level constants or `ClassVar`. Instead:
  - A fixed set of related values is a module-level enum
  - A default a caller may want to change is a parameter of the method that uses it if only that method does, otherwise a dataclass field with that default. For example, a `Heater` takes `target_temperature: float = 20.0` as a field instead of declaring `DEFAULT_TARGET_TEMPERATURE: ClassVar[float] = 20.0`
  - A constant that belongs to a class is a `classproperty` (use the one `krrood` provides)
  - `ClassVar` is allowed only for state that explicitly has to be shared and mutable across every instance of a class. This almost never applies

### Structured Data
- Structured data is the default over bare strings, hardcoded values and meaningless numbers: reach for the structured form first and justify the literal, never the other way round
- A string that names a fixed thing - a payload key, a state, a label, a filename, an environment variable, a command flag, a status - is a `StrEnum` member. A value spelled in two places has no single source to rename, and nothing fails when the two drift apart
- When the values are more than text - paths, numbers - give the enum values of that type, mixing the type in where Python supports it (`IntEnum`, `StrEnum`). `Path` cannot be mixed in, because pathlib builds every derived path through the enum's own member lookup, so a path enum is a plain `Enum` whose values are `Path`s
- A magic number becomes an enum member, a field default or a `classproperty`; a bare literal that carries meaning is unreadable where it is used and unsearchable everywhere else
- JSON that our own classes round-trip goes through `krrood.adapters.json_serializer`: use `DataclassJSONSerializer` wherever it can (de)serialize the class; where a class needs more, subclass `SubclassJSONSerializer` and build on `DataclassJSONSerializer.to_json`/`from_json`. Never hand-write field-by-field `to_json`/`from_json`
- Data whose shape someone else controls - an API response, a configuration file - is mirrored in dataclasses and parsed by a `from_json` classmethod, so the field names and the access path into the payload are written once
- A tuple whose positions carry meaning becomes a dataclass, or an enum when the positions are a fixed set of alternatives rather than fields
- A long literal document - a query, a template, a schema - lives in a file of its own type and is read in, never embedded as a string

## Type Hints
- Every parameter and return value has an accurate type hint, `Any` and `-> None` included
- Use builtin generics and `X | None` (`list[int]`, `Pose | None`), never `typing.List`, `Dict`, `Optional` or `Union`; import every other typing construct from `typing_extensions`, never from `typing`
- Use `from __future__ import annotations` instead of wrapping types in strings
- Generic classes inherit `Generic[...]` and then `krrood.patterns.subclass_safe_generic.SubClassSafeGeneric`, which narrows field types when a subclass binds a parameter. Read a bound type with `get_type_of_generic_parameter`, never via `__orig_bases__`. Such classes cannot be frozen dataclasses
- When each class in a family declares the type it handles, carry that type as a bound generic parameter, not a `ClassVar`, so it is part of the signature and cannot disagree with a separate attribute: bind it in each member (`class MemberOfFamily(Family[ConcreteType])`)

## Documentation
- Every class and method has meaningful, non-trivial documentation in reStructuredText. Every field has its own docstring directly below it, not a description in the class docstring
- An override that keeps its base contract has no docstring and inherits the base one. One that changes the contract documents only the difference and references the overridden method with `:meth:`
- A docstring states what the code does and its contract, short and to the point. It never contains:
  - how the code does it
  - the ways a caller can supply a value (for example, that it may also be given as a query instead of a concrete object)
  - the current callers or consumers ("used by X and Y") - that goes stale and reads as exhaustive
  - justification of design decisions, comparison with rejected or hypothetical designs, or review and implementation history
  - type information - the type hints carry it
- Keep docstrings and comments short and never write walls of text. Comments are meaningful and DRY; remove comments that restate the code
- Reference code with Sphinx roles (`:func:`, `:class:`, `:attr:`, `:meth:`), never with plain backticks, and use directives such as `.. note::` and `.. warning::` for notes and warnings
- Never use all-caps words for emphasis in docstrings or comments; use RST emphasis (`*word*`). Genuine identifiers, acronyms and enum/constant names (`UUID`, `WHERE`, `Definiteness.DEFINITE`) are exempt
- Always run `scripts/format_docstrings.py` (black + docformatter) on modified files

## Domain-Specific Conventions
- For spatial types and connections, follow `semantic_digital_twin/doc/style_guide.md`

## Version Control
- Author commits as the human user running the tool, with their own configured git `user.name` and `user.email`. Never author or amend a commit as an assistant, and never credit one as author or co-author: no `Co-Authored-By:` trailer for an assistant, no `noreply@anthropic.com` or similar as author or committer
- A short plain line in the commit message body acknowledging assistant help, such as `Made with AI assistance`, is encouraged. Never name the model or the AI service
- These rules apply to every contributor and every tool
- Never push to the upstream `cram2/cognitive_robot_abstract_machine` repository - no branch, no tag, whatever the reason. Push only to your fork; work reaches upstream only through a pull request opened from the fork. Before any push, read the full, untruncated `git remote -v` output and confirm the target remote is the fork
- Never comment on or modify pull requests on the upstream repository, unless you work in a fork and the user has explicitly allowed it - through existing personal notes or instructions, or by accepting when you ask

## Misc
- Always use the project's configured Python interpreter for tests and commands
- If a package could be replaced by a more powerful one, tell us
