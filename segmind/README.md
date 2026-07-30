# Segmind

Segmind is a Python library for segmenting simulation episodes of robotic activities by detecting physical interactions and spatial events. It uses a statechart-based approach to monitor simulation data and trigger events like pick-ups, placings, and containment.

## Core Concepts

Segmind revolves around three main components: **Events**, **Detectors**, and **StateCharts**.

### Events

Events represent significant occurrences in the simulation. They are categorized into several types:

- **Atomic Events**: Basic physical changes.
    - `ContactEvent` / `LossOfContactEvent`: Changes in physical contact between bodies.
    - `TranslationEvent` / `StopTranslationEvent`: Start and end of linear motion.
    - `RotationEvent` / `StopRotationEvent`: Start and end of rotational motion.
- **Spatial Relation Events**: Changes in semantic spatial relations.
    - `SupportEvent` / `LossOfSupportEvent`: When an object starts/stops being supported by another.
    - `ContainmentEvent` / `LossOfContainmentEvent`: When an object enters/leaves a container.
- **Coarse (Interaction) Events**: Higher-level activities composed of atomic events.
    - `PickUpEvent`: Combination of a `TranslationEvent` and a `LossOfSupportEvent`.
    - `PlacingEvent`: Combination of a `StopMotionEvent` and a `SupportEvent`.
    - `InsertionEvent`: Detected when an object passes through a "hole" and becomes contained.

### Detectors

Detectors are the logic units responsible for identifying events. They process the simulation state (poses, contacts) at each tick.

- **Atomic Detectors**: `ContactDetector`, `MotionDetector` (Translation/Rotation).
- **Spatial Detectors**: `SupportDetector`, `ContainmentDetector`, `InsertionDetector`.
- **Coarse Detectors**: `PickUpDetector`, `PlacingDetector`.

### StateCharts

The `SegmindStatechart` orchestrates multiple detectors. It acts as a container that ticks all registered detectors against a shared `SegmindContext`.



## Example Usage

The following example demonstrates how to set up a `SegmindStatechart` to detect events in a simulation world.

```python
from segmind.detectors.base import SegmindContext
from segmind.event_logger import EventLogger
from segmind.statecharts.segmind_statechart import SegmindStatechart
from segmind.episode_segmenter import EpisodeSegmenterExecutor

# 1. Setup Context and Logger
logger = EventLogger()
context = SegmindContext(world=your_simulation_world, logger=logger)

# 2. Build Statechart
statechart_factory = SegmindStatechart()
statechart = statechart_factory.build_statechart(context)

# 3. Initialize Executor
executor = EpisodeSegmenterExecutor(context=context)
executor.compile(statechart)

# 4. Simulation Loop
while simulation_running:
    # Update your world state here
    # ...
    executor.tick()

# 5. Retrieve detected events
for event in logger.get_events():
    print(f"Detected {type(event).__name__} at {event.timestamp}")
```

Enjoy segmenting!

