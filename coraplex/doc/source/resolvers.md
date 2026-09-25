# Resolving Designators

A designator description expresses intent and constraints; resolving (grounding) it turns it into a concrete,
executable instance based on the current world and robot state. In CoraPlex this happens automatically while a plan is
performed, so there is usually no need to resolve designators by hand.

## How resolution works

Resolution is deferred to execution time so that the query always sees the correct world state. When a plan reaches an
underspecified step, the {class}`~coraplex.plans.executables.UnderspecifiedExecutable` grounds it only after every
preceding executable has run and mutated the world. Candidates are tried in order until one executes without raising a
{class}`~coraplex.plans.failures.PlanFailure`; if none succeed, the step fails. This late grounding is what lets a plan
adapt to objects that moved, a torso that was already raised, or an object already held in the gripper.

## Location designators

Location designators are resolved into 6D poses by the locations in {mod}`coraplex.locations`. A
{class}`~coraplex.locations.base.Location` is drawn from a costmap (see {doc}`costmap`) for criteria such as reach
distance, visibility and occupancy. The locations in {mod}`coraplex.locations.locations`, such as
{class}`~coraplex.locations.locations.ReachabilityLocation`, build that costmap from the world as it is when they are
drawn from, and a target given relative to a body follows that body. Iterating a location draws candidate poses from
it; whether the robot can do its task from one of them is found out by trying the action that uses it.

## Customising resolution

To change how a particular kind of location is generated, provide or extend a
{class}`~coraplex.locations.base.Location` in {mod}`coraplex.locations` rather than adding a separate resolver module. Custom resolution logic should keep the same
interface as the designator it grounds so it stays a drop-in replacement.
