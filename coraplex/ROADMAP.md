# Failure Handling Subsystem for coraplex

## Context

During plan execution, failures currently either abort the plan (`PlanNode.perform` re-raises) or are blindly swallowed (`UnderspecifiedExecutable` retries the next candidate). This branch introduces a proper failure-handling subsystem: a `FailureHandler` is invoked when a `PlanFailure` occurs, first *refines* the failure through an ensemble of detectors (each declared for an input error type, output error type, and required ParameterMixins of the failing action), then selects a *strategy* declared for the refined error class, which performs a *recovery sub-plan* (real robot actions) and signals how execution should continue. The existing skeleton under `coraplex/src/coraplex/failure_handling/` (all bodies `pass`) is WIP and may be freely reshaped.

**Confirmed design decisions:**
- Refinement applies the **single most-specific** applicable detector per step (ties raise a dedicated exception). A detector may *decline* by returning the failure it was given, which hands the same failure to the next-most-specific candidate (`FailureRefiner.confirmed_refinement`); an action can carry the parameters of several detectors, so the most specific one is not necessarily the one that recognises what went wrong.
- Strategies recover via **recovery sub-plans** (e.g. a `NavigateAction`), not by directly mutating world state — required for real robots.
- **Handled where it happened**: every failure carries the node it occurred at, and that node gets first refusal at handling it via `PlanNode.handle_failure`. A node that cannot deal with a failure escalates it to its parent *node* (`PlanNode.escalate`).
- **Escalation follows the plan tree, not the call stack.** *(Supersedes the original "single trigger point" and "complete separation from nodes" decisions — see below.)*
- **Failure-tolerant nodes terminate escalation.** `TryInOrderNode` and `TryAllNode` (mixin `ToleratesChildFailures`) re-raise a child's failure back into the failing child's perform frame instead of letting the walk continue, because their `notify` already deals with failed children — so tolerated failures neither mark ancestors `FAILED` nor trigger recovery strategies above them. `ParallelNode` does the same only while its children run in worker threads (no cross-thread ancestor mutation); the failure it re-raises after joining escalates normally. Failures raised *at* a tolerant node (its own `AllChildrenFailed`) escalate normally. The handler is still consulted once at the failing node, so recovery keeps first refusal.
- **Exactly one failure object carries the resolution**: `propagate` stamps it on the escalating (refined) failure and `TargetedResolution.apply` clears it at the target; `handle_failure` only consults, so a refined failure's resolution never sticks to the original.

> **Superseded (kept for context).** The subsystem was first built around a single chokepoint in `PlanNode.perform`, with resolutions interpreted against Python call frames: `apply` returning meant "re-run this frame", raising meant "hand to the parent frame", and no `PlanNode` subclass carried failure-handling logic. That model cannot work, because **the plan tree is complete while the Python stack is sparse**: `LanguageNode.notify` calls `child.notify()`, not `child.perform()`, and `Plan.perform` performs only the root. Only the plan root, `TryInOrder` children, parallel children and action-body subplan roots ever get a perform frame — children of `SequentialNode`, `RepeatNode`, `MonitorNode` and underspecified candidates never do, since their work is collapsed into one merged `GiskardExecutable`. A `TargetedResolution` therefore usually could not reach its target at all.

## Architecture

```
PlanFailure raised during execute(), carrying the node it happened at
  └─ caught in PlanNode.perform -> failure.node.handle_failure(failure)
       └─ context.failure_handler.handle(failure)   # once per failure object
            ├─ FailureRefiner.refine(failure)          # detector chain to fixpoint
            │    └─ most-specific FailureDetector.detect(failure) -> refined PlanFailure
            └─ most-specific FailureHandlingStrategy.resolve(refined)
                 ├─ performs a recovery sub-plan (robot actions in same Context)
                 └─ returns FailureResolution: Propagate | RetryNode(node) | Reparameterize(node)
       └─ resolution.apply(node) — no interpretation in the handling node:
            at the target  -> return; the work is run again
            otherwise      -> record status=FAILED and reason, then node.escalate(),
                              which applies the resolution at the parent node and
                              raises once escalation reaches the root
```

Motion failures are attributed through `GiskardExecutable.motion_mappings`: a per-motion watchdog cancels the chart with a `MotionDidNotFinish` that names the `MotionNode` whose motion did not finish, and the timeout path resolves a failed statechart node to its owning motion by walking `parent_node` to the top level before inverting the mapping.

`Context.failure_handler` is always present (`default_factory=FailureHandler.baseline` builds a *baseline* handler: empty detector ensemble + one `UnderspecifiedReparameterizationStrategy`); it lives in `coraplex/src/coraplex/datastructures/dataclasses.py`. A plan opts into recovery by assigning `default_failure_handler()` to its context. The baseline strategy reproduces the pre-subsystem semantics: if the failure occurred beneath an `UnderspecifiedNode` (walk `failure.node.path` — *strict* ancestors only, which is what lets `EmptyUnderspecified` terminate the iteration instead of re-targeting the exhausted node itself), return `Reparameterize(nearest UnderspecifiedNode)`; else `Propagate`. Running an `UnderspecifiedNode` again naturally advances to the next candidate, because its `parse().execute()` calls `advance()`.

**Key reuse:**
- `krrood/src/krrood/patterns/specificity_ranking.py` — `sole_maximum`, `mro_depth` for detector/strategy selection (coraplex→krrood import is allowed).
- `UnderspecifiedNode.advance()/stop_grounding()` (`coraplex/src/coraplex/plans/plan_node.py:490-513`) — docstring already promises reuse by failure handling.
- ParameterMixins (`coraplex/src/coraplex/robot_plans/parameter_mixins.py`) — plain `isinstance` on `action` works since actions multiply-inherit them.
- Never-raised specific failures (`RobotInCollision`, `NavigationGoalNotReachedError`, `BodyUnfetchable`, `EndEffectorDidNotReachTarget`) become the detectors' refined outputs. Every `PlanFailure` subclass now lives in `plans/failures.py` — `ConditionNotSatisfied` and `MotionDidNotFinish` were moved there out of `exceptions.py`.

## Work packages

Each WP is one coding-agent session: TDD (failing test first), dataclasses, field docstrings, absolute imports, typing_extensions, guard clauses, `# %%` headers, docformatter. WPs that change dataclass fields end with `scripts/regenerate_all_orm.py` (never hand-edit `orm/ormatic_interface.py`).

Each WP names a suggested agent model. Rule of thumb: *Sonnet* for well-specified, mechanical packages (precise file/line targets, formulaic patterns); *Opus* for design-sensitive abstractions or changes to core execution semantics; *Opus with extended thinking* (or Fable, if available) for the packages where subtle control-flow mistakes would corrupt plan-execution semantics repo-wide.

### WP0 — Fix latent raise-site bugs *(no deps)* — **done**

Two raise sites constructed `PlanFailure`s with missing required fields (TypeError instead of the intended failure) and blocked the whole subsystem:
- `ConditionNotSatisfied` — construction moved onto the node itself: `ConditionNode.not_satisfied_failure()` (`plans/condition_nodes.py:43`), mirroring `MotionNode.did_not_finish_failure`.
- `coraplex/src/coraplex/language.py` — `AllChildrenFailed(self)` was missing `language_node` (now `node=self, language_node=self`).

Tests: `test/coraplex_test/test_failure_handling/test_failure_construction.py` — every concrete failure constructs with its required kwargs (delivered late, during the review pass); a `TryInOrderNode` of failing leaves yields a well-formed `AllChildrenFailed`.

### WP1 — PlanFailure infrastructure *(no deps; unblocks WP2–WP5)* — **done**

On `PlanFailure` (`coraplex/src/coraplex/plans/failures.py`):
- Field `refined_from: Optional[PlanFailure]` (kw_only, default None) — provenance chain; detectors also set `__cause__`.
- Field `resolution: Optional[FailureResolution]` (kw_only, default None) — prevents double handling while a failure escalates (TYPE_CHECKING import).
- Property `action_node -> Optional[ActionNode]` — `self.node` if `ActionNode`, else `self.node.parent_action_node` (a `PlanNode` property since the review pass deduplicated the walk, `plan_node.py:305`).
- Property `context -> Context` — `self.node.plan.context`.

Tests: `test_failure_construction.py` — `action_node` resolution from nested nodes (build via action factories, no world), `None` for bare `CodeNode`, `refined_from` chain walk.

### WP2 — FailureDetector base + FailureRefiner.refine *(deps: WP1)* — **done**

Rewrite `coraplex/src/coraplex/failure_handling/failure_refiner.py` (drop PEP-695 generic and stray TypeVar):

```python
@dataclass
class FailureDetector(ABC):
    input_failure_type: ClassVar[Type[PlanFailure]] = PlanFailure
    output_failure_type: ClassVar[Type[PlanFailure]] = PlanFailure
    required_parameter_mixins: ClassVar[Tuple[Type, ...]] = ()
    def applies(self, failure: PlanFailure) -> bool: ...
    @abstractmethod
    def detect(self, failure: PlanFailure) -> PlanFailure: ...
```

`applies` = isinstance(failure, input_failure_type) AND failure.action_node is not None AND all mixin isinstance checks on `failure.action_node.action`. Instance-list ensemble stays (`FailureRefiner.failure_detectors`); selection = `sole_maximum` keyed `(mro_depth(input_failure_type), len(required_parameter_mixins))`, collision error `AmbiguousFailureDetector`.

`refine(failure) -> PlanFailure`: loop `confirmed_refinement` → repeat until no detector confirms (fixpoint). One refinement step asks the applicable detectors most-specific first; a detector *declines* by returning the failure it was given, which drops it from the step's candidate set and asks the next-most-specific one (`confirmed_refinement`, `failure_refiner.py:110`). Guards: seen-failure-type set → `FailureRefinementCycle` on repeat; stop on same object/type returned. Sets `refined_from` + `__cause__`; detectors pass `node=current.node` unless more precise.

New exceptions in `coraplex/src/coraplex/exceptions.py`: `AmbiguousFailureDetector`, `FailureRefinementCycle` (plain `DataclassException`s — programming errors, not plan failures).

Tests: `test/coraplex_test/test_failure_handling/test_failure_refiner.py` with stub failures/detectors (no world): fixpoint no-op, single hop, two-hop chain, mixin gating, subclass specificity wins, mixin-count tiebreak, ambiguity raises, cycle raises, `refined_from` set, decline hands over to the next-most-specific, all-decline returns the failure unchanged, a decline that leaves equally specific candidates is ambiguous.

### WP3 — Resolutions + FailureHandlingStrategy + FailureHandler.handle *(deps: WP2)* — **done**

`coraplex/src/coraplex/failure_handling/failure_handling_strategy.py` *(contract below is the tree contract WP4b settled on; the original frame-based wording is superseded)*:
- `FailureResolution` hierarchy (small dataclasses, open for extension) with a single abstract `apply(node: PlanNode) -> None` — the resolution interprets itself; nodes never branch on resolution types. Contract: `apply` *returning* means "the failure was dealt with at this node, run the work again"; the shared `propagate` records `FAILED`/`reason` on the node, stamps itself onto the escalating failure, and hands it to the parent via `node.escalate`, which raises once escalation reaches the root.
  - `Propagate(failure)`: `apply` always propagates. The `__cause__` chain comes from the refiner (refinement provenance) or, after a failed recovery, from the recovery plan's own failure.
  - `RetryNode(node)` / `Reparameterize(node: UnderspecifiedNode)` (shared concrete base `TargetedResolution`): `apply` returns when the applying node is the target (clearing `failure.resolution` so a later failure re-consults the handler), otherwise propagates so the walk continues towards the target.
- `FailureHandlingStrategy(ABC)` with `handled_failure_type: ClassVar[Type[PlanFailure]]`, `applies(failure)`, abstract `resolve(failure) -> FailureResolution`.
- `RecoveryPlanStrategy(FailureHandlingStrategy, ABC)` — the recovery-sub-plan base: abstract `recovery_plan(failure) -> Optional[ActionLike]`; `resolve` performs it as a separate plan in the failing plan's `Context` (same world/robot, `context.plan` restored afterwards), then returns the follow-up resolution (abstract `resolution_after_recovery(failure)`, typically `RetryNode(failure.action_node)`). If the recovery plan itself fails, `resolve` returns `Propagate` with the original refined failure and links the recovery failure as its `__cause__`. A `_recovering` re-entrancy guard makes a recovery that fails like the original propagate instead of recursing, because the recovery shares the failing context's handler. Tree-recorded recovery (via `mount_subplan`) stays follow-up work.

`coraplex/src/coraplex/failure_handling/failure_handler.py`: `strategies: List[FailureHandlingStrategy]`; `handle(failure) -> FailureResolution` = refine → most-specific strategy (`sole_maximum` by `mro_depth(handled_failure_type)`, collision `AmbiguousFailureHandlingStrategy` in exceptions.py) → `resolve`, defaulting to `Propagate(refined)` when none applies. Attempt bookkeeping (max retries) lives in strategy instances, not the handler.

Baseline pieces (used by WP4's Context default):
- `strategies/underspecified_reparameterization_strategy.py` — `UnderspecifiedReparameterizationStrategy` (handled type `PlanFailure`): walk `failure.node.path` for the nearest `UnderspecifiedNode` → `Reparameterize(it)`, else `Propagate(failure)`. This is today's blind candidate iteration, relocated into the failure-handling package.
- `failure_handling/factories.py` — `baseline_failure_handler() -> FailureHandler` (empty refiner + the baseline strategy).

Tests: `test_failure_handler.py` with stub strategies — most-specific wins, no match → `Propagate` carrying the refined failure, refiner runs before selection, ambiguity raises, attempt exhaustion → `Propagate`, a refined failure's resolution does not stick to the original; baseline strategy targets the nearest enclosing `UnderspecifiedNode` and propagates otherwise. `test_recovery_plan_strategy.py` with stub recovery strategies — no recovery plan → `Propagate`, recovery performs in the failing plan's context, failing recovery propagates with the recovery failure linked, re-entrant failure during recovery propagates without a second recovery, `context.plan` restored.

### WP4 — Integration in PlanNode.perform *(deps: WP3; first end-to-end value)* — **done**

- `Context.failure_handler: FailureHandler = field(default_factory=baseline_failure_handler)` (always present, never Optional; lives in `datastructures/dataclasses.py`).
- `PlanNode.perform` became a retry loop whose except block delegates to the failure handler.

Delivered as specified, but the stack-frame model it introduced was superseded immediately afterwards (see the Architecture note): wiring the handler in also made a latent attribution bug fatal, because `MotionDidNotFinish` carried a *giskard* statechart node in a field typed `PlanNode`, so the baseline strategy's `failure.node.path` raised `AttributeError` for every failing motion. Fixed by WP4a.

Also folded in WP0's two raise-site fixes (`AllChildrenFailed`, `ConditionNotSatisfied`), which had never been done.

Tests: `test_perform_integration.py` (originally written in the frame model; rewritten to tree vocabulary during the review pass — the two frame-specific tests were superseded by their tree-contract twins in `test_failure_escalation.py`).

### WP4a — Motion failures are attributed to their `MotionNode` *(deps: WP4)* — **done**

- `MotionNode.did_not_finish_failure(failed_motions)` builds a correctly attributed `MotionDidNotFinish`, mirroring `ConditionNode.not_satisfied_failure()`.
- `GiskardExecutable._add_motion_watchdogs` gives every motion a `CountControlCycles` budget plus a `CancelMotion` carrying that failure, so a stuck motion is reported as soon as it alone spends its budget.
- `GiskardExecutable.owning_motion_node` resolves a failed statechart node to its motion by walking `parent_node` to the top level before inverting `motion_mappings` (goal-returning motions such as `ReachMotion` add grandchildren during `compile`), falling back to the executable's first motion.
- `_execute_simulation` cleans up in a `finally`, which a raising `CancelMotion` previously skipped.

Tests: `test_motion_attribution.py` — one sabotaged pick-up performed by a module-scoped fixture, shared by every test; asserts the stuck pre-grasp reach motion itself is blamed, not just any `MotionNode`.

### WP4b — Node-local handling chain *(deps: WP4a)* — **done**

`PlanNode.handle_failure` / `PlanNode.escalate`, and `FailureResolution.propagate` ends in `node.escalate(...)` instead of `raise`. `perform`'s except block routes to `failure.node.handle_failure(failure)`.

The review pass added the tolerant-node refinement (see the design decision above): `ToleratesChildFailures` on `TryInOrderNode`/`TryAllNode`, and `ParallelNode.handle_failure` keeping child failures inside their worker threads while `_children_running` — which also removed the cross-thread ancestor mutation parallel escalation used to do.

Tests: `test_failure_escalation.py`; the frame-contract tests in `test_failure_handler.py` were rewritten for the tree contract.

### WP5 — UnderspecifiedExecutable stops swallowing failures *(deps: WP4b)* — **done**

`UnderspecifiedExecutable.execute()` lost its `except PlanFailure: continue` loop: it grounds and runs one candidate per execution, raising `EmptyUnderspecified` once the generator is exhausted. A candidate's failure escalates along the plan tree to the underspecified node, which resolves it by running again and thereby advancing — this now also works when the underspecified node is nested in a `sequential`, which the frame-based model could not do.

Tests: `test_underspecified_integration.py`.

### WP5a — Resume a merged chart from the failing motion *(deps: WP5)* — **open** 

**TODO** Is probably needed for recovering in pick and place actions, for example, if the part of a pick up fails running the whole action will fail since the pre-condition is not satisfied anymore.

When a motion in the middle of a merged chart fails and is recovered, the chart has already aborted, so the motions after it never ran; today the enclosing node re-runs its whole work, re-executing completed motions. `GiskardExecutable.execute` should instead resume from the failing motion.

Blocked on a contract decision: after `handle_failure` returns, `TargetedResolution.apply` has already cleared `failure.resolution`, so the executable cannot tell whether the target was one of its own motions (resume locally) or an ancestor (let the ancestor re-run). Resolving it means moving the "clear the resolution" responsibility from `apply` to whoever re-runs the work, so the target stays inspectable. Deferring is safe: without it behaviour is the pre-existing whole-chart retry, not a regression.

### WP6 — Concrete detectors *(deps: WP2; parallel to WP4/WP5)* — **done**

`coraplex/src/coraplex/failure_handling/detectors/motion_detectors.py` (exported in `detectors/__init__.py`), all input `MotionDidNotFinish`:
- `NavigationGoalDetector` — mixins `(TargetLocationMovedTo,)` → `NavigationGoalNotReachedError(current_pose=failure.context.robot.root.global_pose, goal_pose=action.target_location)`.
- `EndEffectorTargetDetector` — mixins `(UsedEndEffector, TargetPoseReached)` → `EndEffectorDidNotReachTarget(...)`.
- `BodyUnfetchableDetector` — mixins `(ObjectActedOn, UsedArm, UsedGraspDescription)` (one more than originally planned) → `BodyUnfetchable(body=action.target_object.root, arm=action.arm)`, deciding reachability via `IsObjectReachableBy` (exercises the mixin-count tiebreak).

Deviations from the original spec: every detector checks the world state and *declines* (returns the input failure) when it contradicts the detector's claim — e.g. the navigation detector declines when the robot does stand at its goal — relying on the WP2 decline protocol instead of refining unconditionally.

**Not yet wired into production:** no shipped handler carries these detectors; that wiring is WP7's `default_failure_handler()`.

Tests: `test_detectors.py` — build `NavigateAction`/`PickUpAction` nodes without executing, hand-craft `MotionDidNotFinish`, assert `applies`/`detect` payloads and decline behaviour; refiner-level test that the ensemble picks different detectors for navigation vs. manipulation actions.

### WP7 — Concrete recovery strategies + factory + end-to-end *(deps: WP4, WP5, WP6)* — **done**

`coraplex/src/coraplex/failure_handling/`:
- `attempt_budget.py` — `AttemptBudget(maximum_attempts=3)`, per-node attempt counts behind a lock (parallel children consult one strategy instance from several threads). Both new strategies bound themselves with it, which is what makes `PlanNode.perform`'s unbounded retry loop terminate against a deterministic failure. Deviation from the original spec: the bound is this object rather than a `maximum_attempts` field per strategy, so it is written once.
- `strategies/retry_strategy.py` — `RetryStrategy` (reusable base) plus `MotionRetryStrategy` (`MotionDidNotFinish`, a motion no detector recognised) and `EndEffectorRetryStrategy` (`EndEffectorDidNotReachTarget`). The base stays declared for `PlanFailure` and is therefore *not* registrable next to the baseline strategy — two strategies of equal specificity are ambiguous, so the shipped ones declare narrower types.
- `strategies/navigation_recovery_strategy.py` — `NavigationRecoveryStrategy` for `NavigationGoalNotReachedError`: recovery plan is a `NavigateAction` to a standing pose regenerated from `occupancy_location(failure.goal_pose, failure.context)`, skipping the goal the robot just failed to reach; then `RetryNode` on the failing action. The budget is consulted before any pose is generated, so giving up costs no world iteration.
- `FailureHandlingStrategy.retried_node(failure)` — `failure.action_node or failure.node`, shared by both strategies. Targeting the action node only became viable with tree-based escalation.
- `factories.py` — `default_failure_handler()` wires all three detectors and `NavigationRecoveryStrategy`, `MotionRetryStrategy`, `EndEffectorRetryStrategy`, `UnderspecifiedReparameterizationStrategy` (which keeps answering `BodyUnfetchable`).

Import-cycle constraint that shaped the layout: `Context` defaults to a handler, and the detectors (via `locations`) and `NavigateAction` import `Context`. So `baseline_failure_handler()` became `FailureHandler.baseline()` (import-light, used by `Context`), `factories.py` is no longer on `Context`'s import path, and **`failure_handling/strategies/__init__.py` must stay empty** — `failure_handler.py` imports a module from that package, which would execute re-exports of the navigation strategy and rebuild the cycle.

Also fixed here, because the per-node budget surfaced it: plan nodes hashed by identity (`PlanNode.__hash__`) but inherited a *generated* `__eq__` from `PlanEntity`, which compares entities by the plan managing them — so every node of a plan equalled every other one, and `CodeNode`/`SequentialNode`/`ParallelNode`/`ConditionNode` were unhashable on top (plain `@dataclass` drops the inherited `__hash__`). `PlanEntity` and those node classes are now `eq=False`.

That in turn exposed `Plan.nodes`, which promised depth-first order but returned `[root] + root.descendants` — and `descendants` is breadth-first. `test_depth_first_nodes_order` had been comparing lists of nodes that all compared equal, so it never checked anything. `PlanNode.subtree` now walks depth-first and `Plan.nodes` returns it, which also makes `PlanNode.previous_nodes` (documented as depth-first, used to find the action a later one follows) actually depth-first.

Open consideration left standing: a targeted resolution only re-runs its target when the target is at or below the nearest enclosing perform frame, so a retry targeting `failure.action_node` beneath a `TryInOrder`/parallel child re-runs that child's whole work.

Tests: `test_strategies.py` (budgets, retry targets, exhaustion, declared failure types, the shipped ensemble's selection per refined failure type, navigation recovery); `test_end_to_end.py` with `mutable_model_world` under `simulated_robot` — the first navigation chart raises the `MotionDidNotFinish` a stuck chart would raise, and with `default_failure_handler()` the plan ends `SUCCEEDED` with the robot at its goal, the blamed motion node recording the `NavigationGoalNotReachedError` refinement, and three charts run (failed attempt, recovery drive, retry). `test_plan.py::test_every_plan_node_is_identified_by_identity` covers the node-identity fix. `mutable_simple_pr2_world` was avoided: it returns the shared session world with a `Context` built on a throwaway copy, so `context.world` and `context.robot` belong to different worlds.

## Dependency graph

```
WP0 ─┐                                   done: WP0..WP5, WP6, WP7
WP1 ─┴→ WP2 → WP3 → WP4 → WP4a → WP4b → WP5 ─┬→ WP5a (open)
              WP2 → WP6 ──────────────────────┴→ WP7
```

End-to-end value from WP4 (refined failures surface to users); full recovery demo at WP7.

## Verification

- Per WP: new failing tests first, then implementation; run `pytest test/coraplex_test/test_failure_handling/`.
- Regression gates: `pytest test/coraplex_test/test_plan/` after WP4 and WP5 (perform/underspecified semantics untouched without a handler).
- After any dataclass field change: `python scripts/regenerate_all_orm.py`, then re-run tests.
- Final: full `pytest test/coraplex_test/` + `docformatter` on modified files.
