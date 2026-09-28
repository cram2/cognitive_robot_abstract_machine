# Core scope

The core package observes an existing CRAM world and its plan execution, renders
live or recorded 3D scenes, saves recordings, and provides EQL and graph inspection.
It uses existing CRAM world/plan callbacks, native motion-state history observers,
and geometry serializers. Motion charts are published when their recorded state
changes. World updates capture the current chart alongside each pose, and plan
completion preserves the final chart observation. History subscriptions end with
their plan or visualization session.

## Deferred features

| Feature | Deferred implementation |
| --- | --- |
| Plan authoring and generated demos | `plan_builder.*`, Builder helpers, catalog/scaffold/save endpoints and generated Python programs |
| Manipulation from the browser | Object dragging, joint sliders, editable placement targets and constraint injection |
| Hand teleoperation | Sandbox page, teleop controller, MediaPipe runtime and hand model |
| Probabilistic model workbench | Models page, workbench API, Plotly and model-editing helpers |
| Guided offline presentation | Tour page, storyboard orchestration and presentation-specific startup tools |
| Multiple robot plans | Robot instance authoring, active-robot selection and multi-robot execution |
| Separate onboarding instrumentation | The old `cramera-onboard` CLI and demo monkey patches; live capture shares the retained bundle serializers |
| Hosting and sample distribution | Mirror/publishing workflows, scenes submodule and prepared demo recordings |

Generic `models[]` loading remains necessary to render a robot and its environment.
It does not expose authoring or selection of multiple executing robots.

The package ships procedural scene lighting and backgrounds. It does not include
lab photos, branding artwork, HDR photographs or camera-tracking model binaries.
