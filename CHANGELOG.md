# Changelog

All notable changes to the packages of the Cognitive Robot Abstract Machine are recorded
here. Every package shares one version, kept in the root `VERSION` file, in the form
`YY.MM.PATCH`: the year and month of the release, then a counter of the releases made in
that month. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Each package section lists the changes since that package's previous release on PyPI.
Packages that are not published to PyPI are summarised since the 26.07.0 release.

## [26.10.0] - 2026-10-31

### Upgrade notes

The changes below can break existing code. Each one is described in full in its package
section.

- **krrood**: An EQL match no longer has public attributes of its own: `expression`,
  `variable`, `matches_with_variables`, `update_fields`, `create_or_update_variable` and
  `has_cause_attributes` are gone. Write attributes and query modifiers directly on the
  match instead ([#662](https://github.com/cram2/cognitive_robot_abstract_machine/pull/662)).
- **krrood**: `AggregationStatistic` is now a `Role`. Read the domain object as
  `role_taker` instead of `instance`, and key `@aggregation_statistic` by a symbolic
  attribute ([#729](https://github.com/cram2/cognitive_robot_abstract_machine/pull/729)).
- **krrood**: `DontPrintStackTrace` is removed. Use
  `DataclassException.print_stack_trace` instead ([#701](https://github.com/cram2/cognitive_robot_abstract_machine/pull/701)).
- **krrood**: `DataclassJSONSerializer` stores numpy arrays as raw bytes. Arrays of Python
  objects and structured arrays are rejected with a dedicated exception ([#824](https://github.com/cram2/cognitive_robot_abstract_machine/pull/824)).
- **probabilistic_model**: Queries return result classes instead of tuples ([#793](https://github.com/cram2/cognitive_robot_abstract_machine/pull/793)).
  `*WithFiniteSupport` base classes are renamed `*WithConstrainedSupport` ([#798](https://github.com/cram2/cognitive_robot_abstract_machine/pull/798)).
- **semantic_digital_twin**: The `Hinge`, `Slider` and `ScrewMechanism` bodies are
  replaced by plain connections ([#699](https://github.com/cram2/cognitive_robot_abstract_machine/pull/699)). `GraspDescription` is replaced by grasp poses
  that objects generate themselves ([#588](https://github.com/cram2/cognitive_robot_abstract_machine/pull/588)). `Pose2D.from_position_and_yaw` is replaced by
  the `Pose2D(position, yaw)` constructor and `Pose2D.from_xy_yaw` ([#746](https://github.com/cram2/cognitive_robot_abstract_machine/pull/746)).
- **coraplex**: A plan runs as one cramph statechart. The plan tree, `MotionDesignator`,
  the plan factories, action pre- and postconditions, `Attempt`/`RepeatOnStall` and the
  ROS 1 modules are removed ([#810](https://github.com/cram2/cognitive_robot_abstract_machine/pull/810)).
- **giskardpy**: `MotionStatechart`, `ExecutionType` and the `GiskardWrapper`
  executors are replaced by the cramph `Statechart` and `GiskardClientExecutor` ([#810](https://github.com/cram2/cognitive_robot_abstract_machine/pull/810)).
- **Workspace**: krrood, probabilistic_model and random_events now require each other at
  version 26.10.0 or newer.

### Repository

- Each package declares its dependencies only in its `pyproject.toml`, including the
  workspace packages it imports ([#569](https://github.com/cram2/cognitive_robot_abstract_machine/pull/569)).
- Generated `ormatic_interface.py` files are no longer tracked. Build them with
  `scripts/regenerate_all_orm.py` ([#543](https://github.com/cram2/cognitive_robot_abstract_machine/pull/543), [#555](https://github.com/cram2/cognitive_robot_abstract_machine/pull/555)).
- Type hints across every package use builtin generics, `X | None` and
  `typing_extensions` ([#807](https://github.com/cram2/cognitive_robot_abstract_machine/pull/807)).
- SegMind follows the repository's shared version ([#805](https://github.com/cram2/cognitive_robot_abstract_machine/pull/805)).
- krrood's dependencies install on Windows ([#831](https://github.com/cram2/cognitive_robot_abstract_machine/pull/831)).

### krrood

Changes since 26.7.0.

#### Highlight: development tools

krrood now ships the tools this repository is developed with, so any Python repository
can use the same conventions ([#806](https://github.com/cram2/cognitive_robot_abstract_machine/pull/806)). `AGENTS.md` is generated from rules declared as code
instead of being written by hand.

```bash
pip install "krrood[development-tools]"   # pins ruff and docformatter

krrood agents-md            # generate AGENTS.md from the shipped rules and agent_rules.py
krrood agents-md --check    # in CI: fail if AGENTS.md is missing, stale or edited by hand
krrood format FILE...       # ruff format, then docformatter wherever ruff keeps its result
```

Both commands exit with 1 when they modify a file, following pre-commit's convention for
hooks. They are published as pre-commit hooks:

```yaml
repos:
  - repo: local
    hooks:
      - id: krrood-format
        name: Format Python (ruff, then docformatter)
        entry: krrood format
        language: python
        additional_dependencies: ["krrood[development-tools]==26.10.0"]
        types: [python]
      - id: krrood-agents-md
        name: Update the rules in AGENTS.md
        entry: krrood agents-md
        language: python
        additional_dependencies: ["krrood[development-tools]==26.10.0"]
        pass_filenames: false
        always_run: true
```

- Rules are enum members grouped into `Section`s. krrood ships rules for any Python
  repository and for repositories built on krrood. A repository adds its own rules in an
  `agent_rules.py` module, or an `agent_rules` package, at its root.
- A rule that names code builds the name from the imported object with `CodeReference`,
  so the rule follows a rename.
- The generated file marks the class and module each section comes from, so every rule
  can be traced back to its source.
- See `krrood/doc/development_tools.md` for the full guide.

#### Added

- **Probabilistic queries in EQL**: `distribution_of(match, *variables)` returns the
  conditioned and truncated model a match describes, and `probability_of(condition)`
  returns its probability. Expectations reuse `average()` ([#575](https://github.com/cram2/cognitive_robot_abstract_machine/pull/575)). `probability_of`
  accepts `evidence=` for conditional probabilities, and raises
  `EvidenceHasZeroProbability` when the evidence is impossible ([#778](https://github.com/cram2/cognitive_robot_abstract_machine/pull/778)).
- **Interventional queries in EQL**: `cause()` marks a match field as an intervention
  target, and `Match.causes_effect(...)` declares the effect it should explain. Under
  `ProbabilisticBackend`, these queries are answered by backdoor adjustment ([#560](https://github.com/cram2/cognitive_robot_abstract_machine/pull/560)).
  Relational models can keep grounded values as causes, through a new `GroundingMode`
  ([#595](https://github.com/cram2/cognitive_robot_abstract_machine/pull/595), [#635](https://github.com/cram2/cognitive_robot_abstract_machine/pull/635), [#767](https://github.com/cram2/cognitive_robot_abstract_machine/pull/767)).
- **Ripple-down rules engine**, first slice, in `entity_query_language.rdr`: backward
  inference, condition suggestion and `ConclusionAid` for help with expert labelling
  ([#516](https://github.com/cram2/cognitive_robot_abstract_machine/pull/516), [#557](https://github.com/cram2/cognitive_robot_abstract_machine/pull/557), [#565](https://github.com/cram2/cognitive_robot_abstract_machine/pull/565)).
- **ORMatic default type mappings** for `date`, `Decimal`, `np.ndarray`, `complex`,
  `bytearray`, `range`, `slice` and timezone-aware `datetime`. Before, these fields were
  silently dropped or produced interfaces that could not be imported ([#688](https://github.com/cram2/cognitive_robot_abstract_machine/pull/688)).
- ORMatic imports classes that a dependency's interface already maps, instead of
  generating them again ([#583](https://github.com/cram2/cognitive_robot_abstract_machine/pull/583)). Building the class diagram shows a progress bar ([#555](https://github.com/cram2/cognitive_robot_abstract_machine/pull/555)).
- `DataclassJSONSerializer` supports `SortedSet` ([#632](https://github.com/cram2/cognitive_robot_abstract_machine/pull/632)), `timedelta`, `timezone`, `UUID`,
  paths, `range`, `slice`, compiled patterns, types and enums ([#615](https://github.com/cram2/cognitive_robot_abstract_machine/pull/615), [#824](https://github.com/cram2/cognitive_robot_abstract_machine/pull/824)).
- The deserialization tracker is generic, so references to tracked objects are written
  as identifiers. This makes motion statechart JSON much smaller ([#615](https://github.com/cram2/cognitive_robot_abstract_machine/pull/615), [#618](https://github.com/cram2/cognitive_robot_abstract_machine/pull/618)).
- Symbolic math expressions round-trip through JSON ([#650](https://github.com/cram2/cognitive_robot_abstract_machine/pull/650)).
- Relational sum-product networks are JSON serializable ([#469](https://github.com/cram2/cognitive_robot_abstract_machine/pull/469)).
- `BoundsInterface` unifies `Bounds` and axis-aligned regions ([#773](https://github.com/cram2/cognitive_robot_abstract_machine/pull/773)).
- `PathEnumeration`: an enum whose members derive plain `Path`s ([#771](https://github.com/cram2/cognitive_robot_abstract_machine/pull/771)).
- `DataclassException.print_stack_trace` controls whether an exception prints its stack
  trace ([#701](https://github.com/cram2/cognitive_robot_abstract_machine/pull/701)).

#### Changed

- **A match reads like an instance of the class it matches** ([#662](https://github.com/cram2/cognitive_robot_abstract_machine/pull/662)). The match's own
  state moves to underscore names (`_type_`, `_kwargs_`, ...), so `a(Drawer).body` is
  the drawer's `body`. A match implements `where`, `having`, `ordered_by`, `distinct`,
  `grouped_by` and `limit`, and returns itself. If the matched class's instances are
  callable, call them through a second pair of parentheses (`an(Adder)(offset=1)(2)`).
  Positional arguments in a pattern raise `PositionalArgumentsInMatchPattern`.
- Feature extraction reads features from domain objects instead of converting them to
  and from data access objects ([#694](https://github.com/cram2/cognitive_robot_abstract_machine/pull/694)).
- Generative backends build each answer from bindings and leave the user's pattern
  unchanged. Rows are weighted by their likelihood ([#777](https://github.com/cram2/cognitive_robot_abstract_machine/pull/777)).
- `AggregationStatistic` is a `Role` of the domain object it describes, and
  `@aggregation_statistic` is keyed by a symbolic attribute ([#729](https://github.com/cram2/cognitive_robot_abstract_machine/pull/729)).
- The mapping types say whether they reach a single value, and
  `apply_mapping_on_external_root` returns every value a chain reaches ([#564](https://github.com/cram2/cognitive_robot_abstract_machine/pull/564)).
- A comparison of a symbolic `Scalar` always returns a `Scalar` ([#616](https://github.com/cram2/cognitive_robot_abstract_machine/pull/616)).
- Symbolic math is safe to use from several threads. CasADi calls are serialised by
  `CasadiLock` ([#603](https://github.com/cram2/cognitive_robot_abstract_machine/pull/603)).
- `SubClassSafeGeneric` subclasses keep identity comparison and are hashable ([#797](https://github.com/cram2/cognitive_robot_abstract_machine/pull/797)).

#### Fixed

- EQL: conditions-root resolution ([#486](https://github.com/cram2/cognitive_robot_abstract_machine/pull/486)), truth of operation results ([#491](https://github.com/cram2/cognitive_robot_abstract_machine/pull/491)), splicing
  rules with `insert_at` ([#498](https://github.com/cram2/cognitive_robot_abstract_machine/pull/498), [#644](https://github.com/cram2/cognitive_robot_abstract_machine/pull/644)), shared nodes in the query graph ([#500](https://github.com/cram2/cognitive_robot_abstract_machine/pull/500)), the query
  root of a variable shared by two queries ([#532](https://github.com/cram2/cognitive_robot_abstract_machine/pull/532)), query-rooted attributes in a query's
  own conditions ([#563](https://github.com/cram2/cognitive_robot_abstract_machine/pull/563)), chains that reach no value ([#641](https://github.com/cram2/cognitive_robot_abstract_machine/pull/641)), the type of a single-variable
  query ([#648](https://github.com/cram2/cognitive_robot_abstract_machine/pull/648)), predicates over several attributes of a match ([#696](https://github.com/cram2/cognitive_robot_abstract_machine/pull/696)), and a query started
  during another query's evaluation ([#774](https://github.com/cram2/cognitive_robot_abstract_machine/pull/774)). A suspended query no longer leaves its
  evaluation context set ([#579](https://github.com/cram2/cognitive_robot_abstract_machine/pull/579)).
- ORMatic: tuple-valued many-to-many relationships ([#558](https://github.com/cram2/cognitive_robot_abstract_machine/pull/558)), name clashes with `_id`
  fields ([#619](https://github.com/cram2/cognitive_robot_abstract_machine/pull/619)), fields typed with unparameterised generics ([#626](https://github.com/cram2/cognitive_robot_abstract_machine/pull/626), [#631](https://github.com/cram2/cognitive_robot_abstract_machine/pull/631)), the conversion
  order of alternative mappings ([#639](https://github.com/cram2/cognitive_robot_abstract_machine/pull/639)), concrete enums in polymorphic enum columns
  ([#642](https://github.com/cram2/cognitive_robot_abstract_machine/pull/642)), DAO lookup for classes reloaded as `__main__` or `__mp_main__` ([#651](https://github.com/cram2/cognitive_robot_abstract_machine/pull/651)), and
  SQLAlchemy 2.1 compatibility ([#682](https://github.com/cram2/cognitive_robot_abstract_machine/pull/682)).
- RDR: models report why they failed to load and can be loaded from any path ([#645](https://github.com/cram2/cognitive_robot_abstract_machine/pull/645)).
  `FunctionCase` no longer generates an empty `if TYPE_CHECKING:` block ([#697](https://github.com/cram2/cognitive_robot_abstract_machine/pull/697)).
- `FloatVariableData` no longer leaves compiled functions pointing at a stale array
  ([#629](https://github.com/cram2/cognitive_robot_abstract_machine/pull/629)).

#### Performance

- Importing the symbol graph no longer loads matplotlib, which saves about 2 s on a cold
  import ([#803](https://github.com/cram2/cognitive_robot_abstract_machine/pull/803)).
- `get_type_of_generic_parameter` caches its result: about 920 ns down to 120 ns per
  lookup ([#800](https://github.com/cram2/cognitive_robot_abstract_machine/pull/800)).
- Building scalar and spatial symbolic expressions is faster. Loading the PR2 drops from
  0.81 s to 0.68 s ([#736](https://github.com/cram2/cognitive_robot_abstract_machine/pull/736), [#737](https://github.com/cram2/cognitive_robot_abstract_machine/pull/737)).
- The probabilistic backend builds its answers without per-sample lookups ([#712](https://github.com/cram2/cognitive_robot_abstract_machine/pull/712)).

### probabilistic_model

Changes since 26.7.0.

#### Added

- **Layered probabilistic circuits** (`probabilistic_circuit.tensorized`) store every
  level of a circuit as sparse parameter arrays, so queries run over whole layers at once
  ([#675](https://github.com/cram2/cognitive_robot_abstract_machine/pull/675), [#727](https://github.com/cram2/cognitive_robot_abstract_machine/pull/727)). They come with discrete layers ([#685](https://github.com/cram2/cognitive_robot_abstract_machine/pull/685)), Gaussian and truncated Gaussian
  layers ([#684](https://github.com/cram2/cognitive_robot_abstract_machine/pull/684)), multivariate Gaussian layers ([#693](https://github.com/cram2/cognitive_robot_abstract_machine/pull/693)) and benchmarks ([#698](https://github.com/cram2/cognitive_robot_abstract_machine/pull/698)). JAX circuits
  convert only through the layered NumPy circuits ([#705](https://github.com/cram2/cognitive_robot_abstract_machine/pull/705)).
- Multivariate Gaussian distributions ([#672](https://github.com/cram2/cognitive_robot_abstract_machine/pull/672)), and circuits learned as Gaussian mixtures
  ([#681](https://github.com/cram2/cognitive_robot_abstract_machine/pull/681)).
- Progressive probabilistic circuits, which learn tasks one after another without
  forgetting earlier ones ([#692](https://github.com/cram2/cognitive_robot_abstract_machine/pull/692)).
- `MarkovChainDistributionTemplate` for HMM-shaped relational circuits ([#590](https://github.com/cram2/cognitive_robot_abstract_machine/pull/590)).
- Mixtures and products can be composed in every circuit type ([#747](https://github.com/cram2/cognitive_robot_abstract_machine/pull/747)), and a circuit
  reports its mixtures in its own terms ([#749](https://github.com/cram2/cognitive_robot_abstract_machine/pull/749)).
- Relational circuits are grounded through one interface shared by every circuit type,
  and can be grounded as layered circuits, including nested exchangeable parts ([#720](https://github.com/cram2/cognitive_robot_abstract_machine/pull/720),
  [#721](https://github.com/cram2/cognitive_robot_abstract_machine/pull/721), [#731](https://github.com/cram2/cognitive_robot_abstract_machine/pull/731)).
- Causal circuits work on either circuit type ([#758](https://github.com/cram2/cognitive_robot_abstract_machine/pull/758), [#767](https://github.com/cram2/cognitive_robot_abstract_machine/pull/767)). How a circuit is fitted is a
  `LearningMethod`, and `StratifiedLearning` gives every value of a cause its own branch
  ([#635](https://github.com/cram2/cognitive_robot_abstract_machine/pull/635)). Relational causal reasoning is supported ([#595](https://github.com/cram2/cognitive_robot_abstract_machine/pull/595)).

#### Changed

- Queries return result classes from `probabilistic_model.query_results` instead of
  tuples ([#793](https://github.com/cram2/cognitive_robot_abstract_machine/pull/793)).
- `ContinuousDistributionWithFiniteSupport`, `ContinuousLayerWithFiniteSupport` and
  `DifferentiableContinuousLayerWithFiniteSupport` are renamed to
  `...WithConstrainedSupport` ([#798](https://github.com/cram2/cognitive_robot_abstract_machine/pull/798)).
- `TruncatedGaussianDistribution` delegates to `scipy.stats.truncnorm`. Its own rejection
  samplers and moment formulas are removed ([#677](https://github.com/cram2/cognitive_robot_abstract_machine/pull/677)).
- `ProbabilisticModel.variables` can be declared as a dataclass field ([#673](https://github.com/cram2/cognitive_robot_abstract_machine/pull/673)).
- `copy` and `deepcopy` behave consistently across every model ([#783](https://github.com/cram2/cognitive_robot_abstract_machine/pull/783)).
- Monte Carlo estimates and shallow circuits use the total variation distance ([#769](https://github.com/cram2/cognitive_robot_abstract_machine/pull/769)).

#### Fixed

- `JointProbabilityTree` splits on data with only continuous variables ([#680](https://github.com/cram2/cognitive_robot_abstract_machine/pull/680)).
- Truncating a truncated Gaussian intersects the two intervals ([#683](https://github.com/cram2/cognitive_robot_abstract_machine/pull/683), [#678](https://github.com/cram2/cognitive_robot_abstract_machine/pull/678)).
- Symbolic distributions no longer return a log-likelihood of `-inf` for categories that
  are present ([#584](https://github.com/cram2/cognitive_robot_abstract_machine/pull/584)).
- `NygaInduction` no longer widens extreme points into a sibling leaf ([#570](https://github.com/cram2/cognitive_robot_abstract_machine/pull/570)).
- `infer_variables_from_dataframe` handles the string dtype of pandas 3 ([#597](https://github.com/cram2/cognitive_robot_abstract_machine/pull/597)).
- Causal circuits: the support determinism check ([#638](https://github.com/cram2/cognitive_robot_abstract_machine/pull/638)), counting a cause range once
  ([#710](https://github.com/cram2/cognitive_robot_abstract_machine/pull/710)), and a fixed order for a cause's regions ([#745](https://github.com/cram2/cognitive_robot_abstract_machine/pull/745)).
- Layered circuits: closed bounds of Dirac delta layers and large symbolic hashes ([#722](https://github.com/cram2/cognitive_robot_abstract_machine/pull/722),
  [#741](https://github.com/cram2/cognitive_robot_abstract_machine/pull/741)), and truncating point masses within their tolerance ([#744](https://github.com/cram2/cognitive_robot_abstract_machine/pull/744)).
- The uniform measure weights pieces by width and pinned variables by point masses
  ([#730](https://github.com/cram2/cognitive_robot_abstract_machine/pull/730)).

#### Performance

- `NygaInduction` scores split candidates in one vectorised pass, and its root lookup is
  linear ([#578](https://github.com/cram2/cognitive_robot_abstract_machine/pull/578)).

### random_events

Changes since 26.7.0.

- The C++ extension builds on Windows ([#831](https://github.com/cram2/cognitive_robot_abstract_machine/pull/831)).
- Inner and outer box approximations of a `Polytope` are faster ([#601](https://github.com/cram2/cognitive_robot_abstract_machine/pull/601)).
- Plotting moved into `random_events.plotting`, with support for 2D and 3D events ([#568](https://github.com/cram2/cognitive_robot_abstract_machine/pull/568)).
- A string names one set element instead of the characters it iterates over ([#640](https://github.com/cram2/cognitive_robot_abstract_machine/pull/640)).

### semantic_digital_twin

Changes since 26.7.0.

#### Added

- **Specifications**: world-independent, reusable descriptions of bodies, regions,
  connections, annotations, robots and whole worlds, which an explicit step turns into
  world entities. Each one materialises a fresh world ([#402](https://github.com/cram2/cognitive_robot_abstract_machine/pull/402), [#525](https://github.com/cram2/cognitive_robot_abstract_machine/pull/525)).
- **World rollback**: every world modification can be undone, and
  `World.rollback_modification_blocks()` reverts recent blocks ([#594](https://github.com/cram2/cognitive_robot_abstract_machine/pull/594)).
- **Parsers and datasets**: a USD parser and the ArtVIP dataset ([#552](https://github.com/cram2/cognitive_robot_abstract_machine/pull/552)), a Gazebo parser
  ([#493](https://github.com/cram2/cognitive_robot_abstract_machine/pull/493)), a GraspClutter6D adapter ([#625](https://github.com/cram2/cognitive_robot_abstract_machine/pull/625)), and meshes read from a dataset server ([#637](https://github.com/cram2/cognitive_robot_abstract_machine/pull/637)).
- **Physical simulation in MuJoCo** for Tracy, the PR2, Tiago and the HSR, driven by
  Giskard ([#630](https://github.com/cram2/cognitive_robot_abstract_machine/pull/630), [#732](https://github.com/cram2/cognitive_robot_abstract_machine/pull/732), [#734](https://github.com/cram2/cognitive_robot_abstract_machine/pull/734), [#752](https://github.com/cram2/cognitive_robot_abstract_machine/pull/752), [#768](https://github.com/cram2/cognitive_robot_abstract_machine/pull/768), [#794](https://github.com/cram2/cognitive_robot_abstract_machine/pull/794)).
- **Grasping**: objects generate their own grasp poses ([#588](https://github.com/cram2/cognitive_robot_abstract_machine/pull/588)), and semantic annotations
  state where they may be grasped as an EQL statement over surface grasps ([#762](https://github.com/cram2/cognitive_robot_abstract_machine/pull/762)).
- **Perception**: camera models and observations shared with RoboKudo ([#761](https://github.com/cram2/cognitive_robot_abstract_machine/pull/761)), lidar
  annotations for the PR2, HSR, Tiago and Stretch ([#600](https://github.com/cram2/cognitive_robot_abstract_machine/pull/600)), rectangular resolutions in
  `RayTracer` ([#439](https://github.com/cram2/cognitive_robot_abstract_machine/pull/439)), and the surface finish of a body ([#660](https://github.com/cram2/cognitive_robot_abstract_machine/pull/660)).
- Predicates answer whether they hold, together with the measurement behind the answer
  ([#655](https://github.com/cram2/cognitive_robot_abstract_machine/pull/655)).
- Parts such as handles and rims are split off an object's shape ([#728](https://github.com/cram2/cognitive_robot_abstract_machine/pull/728)).
- Graphs of convex sets: Drake GCS with convex polytopes ([#503](https://github.com/cram2/cognitive_robot_abstract_machine/pull/503)), separate 2D and 3D
  classes with `constrain_to_free_space` ([#568](https://github.com/cram2/cognitive_robot_abstract_machine/pull/568)), and distance-weighted path finding
  ([#519](https://github.com/cram2/cognitive_robot_abstract_machine/pull/519)).
- `ScrewConnection` models a screw mechanism as a single degree of freedom ([#484](https://github.com/cram2/cognitive_robot_abstract_machine/pull/484)).
- The DAiSy robot ([#405](https://github.com/cram2/cognitive_robot_abstract_machine/pull/405)) and its gripper ([#687](https://github.com/cram2/cognitive_robot_abstract_machine/pull/687)).
- The annotations `TumbleDryer` and `WashingMachine`. `Chair` can annotate a body ([#822](https://github.com/cram2/cognitive_robot_abstract_machine/pull/822)).
- `OrthogonalProjection` relates `Point2` and `Point3` ([#764](https://github.com/cram2/cognitive_robot_abstract_machine/pull/764)).
- `PrefixPathPackageLocator` finds ROS packages without ROS tooling ([#520](https://github.com/cram2/cognitive_robot_abstract_machine/pull/520)).
- Unique, readable TF names ([#571](https://github.com/cram2/cognitive_robot_abstract_machine/pull/571)), and `combined_mesh` on semantic annotations ([#576](https://github.com/cram2/cognitive_robot_abstract_machine/pull/576)).

#### Changed

- `Hinge`, `Slider` and `ScrewMechanism` bodies are replaced by plain connections ([#699](https://github.com/cram2/cognitive_robot_abstract_machine/pull/699)).
- `Pose2D(position, yaw, reference_frame)` and `Pose2D.from_xy_yaw` replace
  `from_position_and_yaw` ([#746](https://github.com/cram2/cognitive_robot_abstract_machine/pull/746)).
- `Connection` is a `WorldEntityWithID` ([#618](https://github.com/cram2/cognitive_robot_abstract_machine/pull/618)).
- Meshes are serialised as binary GLB. A 13.2 M-face scene that ran out of memory at
  14 GB now builds in 9 s using 5.8 GB ([#824](https://github.com/cram2/cognitive_robot_abstract_machine/pull/824)).
- Markers refer to meshes by `package://` URI instead of absolute file paths ([#811](https://github.com/cram2/cognitive_robot_abstract_machine/pull/811)).

#### Fixed

- URDF: mesh colours from materials ([#801](https://github.com/cram2/cognitive_robot_abstract_machine/pull/801)), floating joints and non-unit joint axes
  ([#796](https://github.com/cram2/cognitive_robot_abstract_machine/pull/796)), and doors wired straight to their cabinet ([#550](https://github.com/cram2/cognitive_robot_abstract_machine/pull/550)).
- Collision checking: rule distances, equality and syncing ([#623](https://github.com/cram2/cognitive_robot_abstract_machine/pull/623)), held objects against
  the end effector ([#621](https://github.com/cram2/cognitive_robot_abstract_machine/pull/621)), and contacts exactly at the violated distance ([#756](https://github.com/cram2/cognitive_robot_abstract_machine/pull/756)).
- Synchronisation: the synchroniser blocks until it has started ([#540](https://github.com/cram2/cognitive_robot_abstract_machine/pull/540), [#611](https://github.com/cram2/cognitive_robot_abstract_machine/pull/611)), and
  world-to-simulation pose writes are no longer lost ([#559](https://github.com/cram2/cognitive_robot_abstract_machine/pull/559)).
- Meshes: convex decompositions are cached only when finished ([#819](https://github.com/cram2/cognitive_robot_abstract_machine/pull/819)), temporary files no
  longer leak ([#522](https://github.com/cram2/cognitive_robot_abstract_machine/pull/522)), and textures are no longer decoded needlessly ([#786](https://github.com/cram2/cognitive_robot_abstract_machine/pull/786), [#787](https://github.com/cram2/cognitive_robot_abstract_machine/pull/787), [#789](https://github.com/cram2/cognitive_robot_abstract_machine/pull/789),
  [#790](https://github.com/cram2/cognitive_robot_abstract_machine/pull/790)).
- MuJoCo: unique names for equally named entities ([#757](https://github.com/cram2/cognitive_robot_abstract_machine/pull/757)), spawn poses of free-jointed
  objects ([#553](https://github.com/cram2/cognitive_robot_abstract_machine/pull/553)), and the video recorder's frame rate ([#614](https://github.com/cram2/cognitive_robot_abstract_machine/pull/614)).
- A world's modification history stays replayable ([#527](https://github.com/cram2/cognitive_robot_abstract_machine/pull/527)). Soft connections serialise
  ([#606](https://github.com/cram2/cognitive_robot_abstract_machine/pull/606)). Supporting surfaces work for any origin ([#706](https://github.com/cram2/cognitive_robot_abstract_machine/pull/706)). The kitchen maps are corrected
  ([#593](https://github.com/cram2/cognitive_robot_abstract_machine/pull/593), [#607](https://github.com/cram2/cognitive_robot_abstract_machine/pull/607), [#760](https://github.com/cram2/cognitive_robot_abstract_machine/pull/760)).

#### Performance

- Moving a branch and forward kinematics track model revisions ([#572](https://github.com/cram2/cognitive_robot_abstract_machine/pull/572)).
- Building spatial types is faster ([#737](https://github.com/cram2/cognitive_robot_abstract_machine/pull/737)).

### coraplex

Changes since 26.7.0.

#### Changed

- **A plan runs as one cramph statechart**, and each action is a node that creates its
  own steps ([#810](https://github.com/cram2/cognitive_robot_abstract_machine/pull/810)). The plan tree, `MotionDesignator`, the plan factories, action pre-
  and postconditions, `Attempt`/`RepeatOnStall`, the legacy ROS 1 modules and unused
  external interfaces are removed.

#### Added

- Executors for kinematic simulation, physical simulation in MuJoCo, the real robot, and
  a Giskard client that chooses children while Giskard runs the goal. Each executor
  adapts the plan to its robot ([#718](https://github.com/cram2/cognitive_robot_abstract_machine/pull/718), [#810](https://github.com/cram2/cognitive_robot_abstract_machine/pull/810)).
- The actions `CarryAction`, `SearchAction` and `PickUpAction.from_above` ([#810](https://github.com/cram2/cognitive_robot_abstract_machine/pull/810)). A
  navigation action that plans with graphs of convex sets ([#592](https://github.com/cram2/cognitive_robot_abstract_machine/pull/592)), and navigation through
  elevators ([#566](https://github.com/cram2/cognitive_robot_abstract_machine/pull/566)).
- Plan transformations adapt an expanded plan to a scenario, for example detecting an
  object before grasping it ([#579](https://github.com/cram2/cognitive_robot_abstract_machine/pull/579)).
- Underspecified nodes try each candidate on a disposable copy of the world first ([#599](https://github.com/cram2/cognitive_robot_abstract_machine/pull/599)).
- Plan node details show designator parameters readably ([#545](https://github.com/cram2/cognitive_robot_abstract_machine/pull/545), [#814](https://github.com/cram2/cognitive_robot_abstract_machine/pull/814)).
- Motion thresholds ([#549](https://github.com/cram2/cognitive_robot_abstract_machine/pull/549)) and physics parameters for pick and place ([#492](https://github.com/cram2/cognitive_robot_abstract_machine/pull/492)).
- GARMI ([#713](https://github.com/cram2/cognitive_robot_abstract_machine/pull/713), [#815](https://github.com/cram2/cognitive_robot_abstract_machine/pull/815)), DAiSy ([#687](https://github.com/cram2/cognitive_robot_abstract_machine/pull/687)) and the Tracy Montessori demo ([#738](https://github.com/cram2/cognitive_robot_abstract_machine/pull/738)).
- cramera, an optional browser-based tool for live visualisation, replay, EQL queries and
  plan inspection ([#667](https://github.com/cram2/cognitive_robot_abstract_machine/pull/667)).

#### Fixed

- Collision avoidance in worlds with several robots ([#823](https://github.com/cram2/cognitive_robot_abstract_machine/pull/823)) and in motion charts ([#589](https://github.com/cram2/cognitive_robot_abstract_machine/pull/589)).
- Attachments held by the physics are reported as executed ([#813](https://github.com/cram2/cognitive_robot_abstract_machine/pull/813)).
- Location poses no longer assume the robot's parent is the world root ([#533](https://github.com/cram2/cognitive_robot_abstract_machine/pull/533)).
- A demonstration's scene is merged into the world fetched on real runs ([#779](https://github.com/cram2/cognitive_robot_abstract_machine/pull/779)).
- Costmap obstacle inflation no longer copies every window ([#791](https://github.com/cram2/cognitive_robot_abstract_machine/pull/791)).

### Packages not published to PyPI

These packages take the shared version but are not released on PyPI.

- **cramph** (new): a statechart engine for any kind of task, taken out of giskardpy's
  motion statechart. It supports node life cycles, transition conditions, composite
  nodes (sequence, parallel, retry), statecharts that grow at runtime, executor and
  context extensions, JSON and ORM round trips, plotting, and a ROS statechart inspector
  ([#810](https://github.com/cram2/cognitive_robot_abstract_machine/pull/810)).
- **giskardpy**: `MotionStatechart` is replaced by the cramph `Statechart` ([#810](https://github.com/cram2/cognitive_robot_abstract_machine/pull/810)). An
  explicit motion server replaces the behaviour tree ([#487](https://github.com/cram2/cognitive_robot_abstract_machine/pull/487)). Nodes end as `SUCCEEDED`,
  `FAILED` or `INTERRUPTED` ([#581](https://github.com/cram2/cognitive_robot_abstract_machine/pull/581)). New: stalling detection ([#517](https://github.com/cram2/cognitive_robot_abstract_machine/pull/517)), jerk limits derived
  from a braking time ([#679](https://github.com/cram2/cognitive_robot_abstract_machine/pull/679)), client presence and a watchdog ([#610](https://github.com/cram2/cognitive_robot_abstract_machine/pull/610), [#750](https://github.com/cram2/cognitive_robot_abstract_machine/pull/750)), graceful
  shutdown ([#586](https://github.com/cram2/cognitive_robot_abstract_machine/pull/586)), RPL semantics for `Success` and `TryInOrder` ([#605](https://github.com/cram2/cognitive_robot_abstract_machine/pull/605)), and ROS 2
  configurations for real robots ([#810](https://github.com/cram2/cognitive_robot_abstract_machine/pull/810)). Several infeasible QP cases are fixed ([#534](https://github.com/cram2/cognitive_robot_abstract_machine/pull/534),
  [#551](https://github.com/cram2/cognitive_robot_abstract_machine/pull/551), [#617](https://github.com/cram2/cognitive_robot_abstract_machine/pull/617), [#622](https://github.com/cram2/cognitive_robot_abstract_machine/pull/622), [#624](https://github.com/cram2/cognitive_robot_abstract_machine/pull/624), [#634](https://github.com/cram2/cognitive_robot_abstract_machine/pull/634), [#671](https://github.com/cram2/cognitive_robot_abstract_machine/pull/671), [#676](https://github.com/cram2/cognitive_robot_abstract_machine/pull/676)).
- **segmind**: detectors are cramph statecharts ([#810](https://github.com/cram2/cognitive_robot_abstract_machine/pull/810)), the unused ripple-down rule
  models are removed ([#820](https://github.com/cram2/cognitive_robot_abstract_machine/pull/820)), containment detection is fixed ([#717](https://github.com/cram2/cognitive_robot_abstract_machine/pull/717)), event loggers are
  released after their run ([#788](https://github.com/cram2/cognitive_robot_abstract_machine/pull/788)), and SegMind runs in the Bullet World demo ([#670](https://github.com/cram2/cognitive_robot_abstract_machine/pull/670)).
- **robokudo**: the `YoloAnnotator` ([#748](https://github.com/cram2/cognitive_robot_abstract_machine/pull/748)) and `HumanAndPoseAnnotator` ([#751](https://github.com/cram2/cognitive_robot_abstract_machine/pull/751)) moved into
  the core, `ClipAnnotator` is renamed `ZeroShotClassificationAnnotator` ([#740](https://github.com/cram2/cognitive_robot_abstract_machine/pull/740)), and an
  open-vocabulary annotator is added ([#719](https://github.com/cram2/cognitive_robot_abstract_machine/pull/719)). Camera models are shared with
  semantic_digital_twin ([#761](https://github.com/cram2/cognitive_robot_abstract_machine/pull/761)), and the ROS life cycle is improved ([#567](https://github.com/cram2/cognitive_robot_abstract_machine/pull/567), [#664](https://github.com/cram2/cognitive_robot_abstract_machine/pull/664)). Plane
  RANSAC is reproducible ([#652](https://github.com/cram2/cognitive_robot_abstract_machine/pull/652)), and empty clouds are handled ([#782](https://github.com/cram2/cognitive_robot_abstract_machine/pull/782)).
- **physics_simulators**: actuator-driven joints in MuJoCo ([#630](https://github.com/cram2/cognitive_robot_abstract_machine/pull/630), [#752](https://github.com/cram2/cognitive_robot_abstract_machine/pull/752)).
- **experiments**: experiments for causal circuits, layered circuits, graphs of convex
  sets and shelf scenes ([#509](https://github.com/cram2/cognitive_robot_abstract_machine/pull/509), [#604](https://github.com/cram2/cognitive_robot_abstract_machine/pull/604), [#636](https://github.com/cram2/cognitive_robot_abstract_machine/pull/636), [#698](https://github.com/cram2/cognitive_robot_abstract_machine/pull/698), [#711](https://github.com/cram2/cognitive_robot_abstract_machine/pull/711)).

[26.10.0]: https://github.com/cram2/cognitive_robot_abstract_machine/compare/52f4f745da...main
