# Motion Statecharts

Motion Statecharts are a core concept in Giskard for composing complex robot motions. They provide a structured way to manage the transition between different motion goals and monitors, making it easier to build robust and reactive robot behaviors.

## The Problem

Traditional robot motion planning often involves a sequence of fixed waypoints or a single, monolithic trajectory. This approach faces several challenges:

- **Complex Sequencing**: Coordinating multiple movements (e.g., "move arm to pre-grasp," then "close gripper," then "lift arm") can become hard to manage as the number of steps increases.
- **Error Handling**: What happens if a collision is detected mid-motion? Or if the gripper fails to close? Handling these contingencies in a flat script often leads to "spaghetti code."
- **Reactivity**: Modern robots need to respond to their environment. A simple trajectory doesn't easily allow for behavior like "move until a certain force is felt" or "stop if a human enters the workspace."

## How Motion Statecharts Solve It

Motion Statecharts address these issues by using a state machine-based approach to motion composition. 

### Key Concepts

A Motion Statechart comprises multiple **nodes**, which may be one of the following
types:
- **Task**: A specific, single-purpose segment of the overall motion. These nodes add constraints to the motion problem and monitor their progress. For example, a Cartesian position task will monitor if the distance to the target is below a threshold.
- **Monitor**: Nodes that observe certain conditions or events without controlling motion. For example, monitoring the distance between the robot’s gripper and a goal point without actively controlling the motion.
- **Termination Node**: Nodes that signal the end of motion execution upon reaching a specific observation state. For example, a node that terminates the motion when a specific condition is met, such as reaching the final destination. There are two termination nodes, **EndMotion** and **CancelMotion**.
- **Goal**: Nodes that encapsulate reusable, parameterized designs for Motion Statechart patterns. For example, a combination of monitors and motion tasks to open a door can be encapsulated into a template for
reuse in different contexts.


Every node has two inner state machines:
- An observation FSM is used to represent whether an observed condition is currently True, False, or "?". For tasks, this state represents the constraint satisfaction status. An explicit "?" state is required for situations in which the state is unknown, for example when the node has never been active.
- A life cycle FSM is used to determine whether the node is active.
  - Possible states are :
    - Not Started: The initial state.
    - Running: The only active state. The Nodes can updates their observation state and Tasks will cause motions.
    - Paused: Indicates a temporary inactivity
    - Stopped: Indicates that the node purpose is done
  - Every node has life cycle transitions that based on observation state of itself or other nodes. Since "?" is a possible observation state, the transitions use three-values logic:
    - start: When True, Transition from Not Started -> Running
    - pause: When True, Transition from Running -> Pause, when not True, transition from Pause -> Running
    - end: When True, transition from Running or Paused -> Stopped
    - reset: When True, transition from any State to Not Started.
    - When multiple transitions are possible, the following priority is used: reset > end > pause > start

### Benefits


- **Modularity**: Individual motions and checks are self-contained nodes that can be reused across different tasks.
- **Clarity**: The statechart structure provides a clear visual and logical representation of the robot's behavior.
- **Robustness**: Error handling and environment reactivity are built directly into the motion's structure through monitors and transitions.
- **Constraint-Based**: The constraints of all currently active tasks influence the motion, ensuring the robot satisfies all requirements simultaneously (e.g., "reach for the cup while keeping the arm away from the table").


For practical examples of how to use Motion Statecharts, see the [Basic Motion](examples/basic_motion.md) and [Cartesian Goals](examples/cartesian_goals.md) tutorials.
