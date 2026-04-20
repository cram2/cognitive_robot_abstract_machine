import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1] / "src" / "pycram"
THESIS_MATH_DIR = ROOT / "robot_plans" / "actions" / "composite" / "thesis_math"


def _ensure_package(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, THESIS_MATH_DIR / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def thesis_math_modules():
    for pkg in [
        "pycram",
        "pycram.robot_plans",
        "pycram.robot_plans.actions",
        "pycram.robot_plans.actions.composite",
        "pycram.robot_plans.actions.composite.thesis_math",
        "semantic_digital_twin",
    ]:
        _ensure_package(pkg)

    spatial_types = types.ModuleType("semantic_digital_twin.spatial_types")

    class HomogeneousTransformationMatrix:
        def __init__(self, matrix):
            self._matrix = np.asarray(matrix, dtype=float)

        def to_np(self):
            return self._matrix

    spatial_types.HomogeneousTransformationMatrix = HomogeneousTransformationMatrix
    sys.modules["semantic_digital_twin.spatial_types"] = spatial_types

    profiles = _load_module(
        "pycram.robot_plans.actions.composite.thesis_math.motion_profiles",
        "motion_profiles.py",
    )
    models = _load_module(
        "pycram.robot_plans.actions.composite.thesis_math.motion_models",
        "motion_models.py",
    )

    world_utils = types.ModuleType(
        "pycram.robot_plans.actions.composite.thesis_math.world_utils"
    )
    world_utils.body_local_aabb = lambda body, **kwargs: body.local_aabb(**kwargs)
    sys.modules[world_utils.__name__] = world_utils

    presets = _load_module(
        "pycram.robot_plans.actions.composite.thesis_math.motion_presets",
        "motion_presets.py",
    )
    return profiles, models, presets


class DummyBody:
    def __init__(self, mins, maxs):
        self._mins = np.asarray(mins, dtype=float)
        self._maxs = np.asarray(maxs, dtype=float)

    def local_aabb(self, **kwargs):
        return self._mins.copy(), self._maxs.copy()


class DummyFrame:
    def __init__(self, matrix):
        self._matrix = np.asarray(matrix, dtype=float)

    def to_np(self):
        return self._matrix


def test_ramp_and_spiral_profiles(thesis_math_modules):
    profiles, _, _ = thesis_math_modules

    assert profiles.ramp(-0.1, tau_end=0.5, d_max=0.3) == 0.0
    assert profiles.ramp(0.25, tau_end=0.5, d_max=0.3) == pytest.approx(0.15)
    assert profiles.ramp(0.75, tau_end=0.5, d_max=0.3) == pytest.approx(0.3)

    spiral = profiles.planar_spiral_xy(0.5, r0=0.1, r1=0.5, cycles=1.0)
    assert spiral == pytest.approx(np.array([-0.3, 0.0, 0.0]))

    sweep = profiles.planar_sweep_x(0.25, length=0.2, cycles=1.0)
    assert sweep == pytest.approx(np.array([0.2, 0.0, 0.0]))


def test_raster_sampling_and_constraints(thesis_math_modules):
    profiles, _, _ = thesis_math_modules

    lane0 = profiles.planar_raster_xy(0.125, width=2.0, height=4.0, lanes=3)
    lane1 = profiles.planar_raster_xy(0.5, width=2.0, height=4.0, lanes=3)
    lane_last = profiles.planar_raster_xy(1.0, width=2.0, height=4.0, lanes=3)

    assert lane0 == pytest.approx(np.array([-0.25, -2.0, 0.0]))
    assert lane1 == pytest.approx(np.array([0.0, 0.0, 0.0]))
    assert lane_last == pytest.approx(np.array([1.0, 2.0, 0.0]))

    sampled = profiles.sample_local_curve(
        lambda tau: np.array([tau, tau**2, -tau], dtype=float),
        [0.0, 0.5, 1.0],
    )
    assert sampled == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [0.5, 0.25, -0.5], [1.0, 1.0, -1.0]])
    )

    clamped_box = profiles.clamp_to_aabb(
        np.array([2.0, -2.0, 0.1]),
        mins=np.array([-1.0, -1.0, -1.0]),
        maxs=np.array([1.0, 1.0, 1.0]),
        margin=0.2,
    )
    assert clamped_box == pytest.approx(np.array([0.8, -0.8, 0.1]))

    clamped_cyl = profiles.clamp_to_cylinder_xy(
        np.array([3.0, 4.0, 2.0]), radius=4.0, z_min=-1.0, z_max=1.0, margin=0.5
    )
    assert clamped_cyl == pytest.approx(np.array([2.1, 2.8, 0.5]))

    constrained = profiles.make_constrained_curve(
        lambda tau: np.array([tau, tau, tau], dtype=float),
        lambda q: q * 2.0,
    )
    assert constrained(0.25) == pytest.approx(np.array([0.5, 0.5, 0.5]))


def test_shear_profiles(thesis_math_modules):
    profiles, _, _ = thesis_math_modules

    local = profiles.oscillatory_shear_local_profiled(
        0.25,
        profiles.ShearProfile(
            depth_max=0.4,
            depth_ramp_end=0.5,
            shear_amp=0.3,
            shear_cycles=1.0,
        ),
    )
    xy = profiles.oscillatory_shear_xy_profiled(
        0.0, profiles.ShearXYProfile(shear_amp=0.2, shear_cycles=1.0)
    )

    assert local == pytest.approx(np.array([0.3, 0.0, -0.2]))
    assert xy == pytest.approx(np.array([0.0, 0.2, 0.0]))


def test_motion_segment_sampling_applies_frame_transform(thesis_math_modules):
    _, models, _ = thesis_math_modules

    segment = models.MotionSegment(
        name="line",
        duration_s=1.0,
        local_curve=lambda tau: np.array([tau, 2.0 * tau, 0.0], dtype=float),
    )
    frame = np.array(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    times, pts = segment.sample(frame, dt=0.5, t0=1.0)

    assert times == pytest.approx(np.array([1.0, 1.5, 2.0]))
    assert pts == pytest.approx(
        np.array([[10.0, 20.0, 30.0], [9.0, 20.5, 30.0], [8.0, 21.0, 30.0]])
    )


def test_motion_sequence_concatenates_without_duplicate_boundary(thesis_math_modules):
    _, models, _ = thesis_math_modules

    seq = models.MotionSequence(
        [
            models.MotionSegment(
                name="first",
                duration_s=1.0,
                local_curve=lambda tau: np.array([tau, 0.0, 0.0], dtype=float),
            ),
            models.MotionSegment(
                name="second",
                duration_s=1.0,
                local_curve=lambda tau: np.array([1.0, tau, 0.0], dtype=float),
            ),
        ]
    )

    times, pts, phase_ids = seq.sample(DummyFrame(np.eye(4)), dt=0.5, t0=2.0)

    assert seq.duration_s == pytest.approx(2.0)
    assert times == pytest.approx(np.array([2.0, 2.5, 3.0, 3.5, 4.0]))
    assert pts == pytest.approx(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    assert np.array_equal(phase_ids, np.array([0, 0, 0, 1, 1]))


def test_duration_scale_uses_aabb_diagonal_and_validates_reference(thesis_math_modules):
    _, _, presets = thesis_math_modules
    body = DummyBody(mins=[0.0, 0.0, 0.0], maxs=[3.0, 4.0, 0.0])

    scale = presets._duration_scale_from_body(body, reference_size=2.5)
    assert scale == pytest.approx(2.0)

    with pytest.raises(ValueError, match="reference_size must be positive"):
        presets._duration_scale_from_body(body, reference_size=0.0)


def test_build_default_sequence_returns_three_named_phases(thesis_math_modules):
    _, _, presets = thesis_math_modules

    seq = presets.build_default_sequence()

    assert [phase.name for phase in seq.phases] == [
        "planar_spiral",
        "oscillatory_shear",
        "planar_sweep",
    ]
    assert seq.duration_s == pytest.approx(5.0)


def test_build_container_sequence_supports_spiral_and_stir_patterns(
    thesis_math_modules,
):
    _, _, presets = thesis_math_modules
    bowl = DummyBody(mins=[-0.2, -0.3, 0.1], maxs=[0.2, 0.3, 0.5])

    spiral = presets.build_container_sequence(bowl, pattern="spiral")
    stir = presets.build_container_sequence(bowl, pattern="loop", mix_duration_s=5.5)

    assert [phase.name for phase in spiral.phases] == ["planar_spiral_bowl"]
    spiral_start = spiral.phases[0].local_curve(0.0)
    spiral_edge = spiral.phases[0].local_curve(1.0)
    assert spiral_start == pytest.approx(np.array([0.0, 0.0, 0.495]))
    assert np.linalg.norm(spiral_edge[:2]) <= 0.195 + 1e-9
    assert 0.105 <= spiral_edge[2] <= 0.495

    assert [phase.name for phase in stir.phases] == ["continuous_stir_bowl"]
    assert stir.phases[0].duration_s == pytest.approx(5.5)
    stir_point = stir.phases[0].local_curve(0.25)
    assert np.linalg.norm(stir_point[:2]) <= 0.195 + 1e-9
    assert stir_point[2] == pytest.approx(0.495)

    with pytest.raises(ValueError, match="Unknown pattern"):
        presets.build_container_sequence(bowl, pattern="unknown")


def test_build_surface_sequence_selects_expected_patterns(thesis_math_modules):
    _, _, presets = thesis_math_modules
    surface = DummyBody(mins=[-0.4, -0.2, 0.0], maxs=[0.6, 0.2, 0.1])

    spiral = presets.build_surface_sequence(surface, pattern="planar_spiral")
    shear = presets.build_surface_sequence(surface, pattern="shear")
    raster = presets.build_surface_sequence(surface, pattern="surface_cover")

    assert [phase.name for phase in spiral.phases] == ["planar_spiral_surface"]
    assert [phase.name for phase in shear.phases] == ["oscillatory_shear_surface"]
    assert [phase.name for phase in raster.phases] == ["planar_raster_surface"]

    center = np.array([0.1, 0.0, 0.115])
    assert spiral.phases[0].local_curve(0.0) == pytest.approx(center)
    assert shear.phases[0].local_curve(0.0) == pytest.approx(
        center + np.array([0.0, 0.063, 0.0])
    )
    assert raster.phases[0].local_curve(0.0) == pytest.approx(
        center + np.array([-0.45, -0.18, 0.0])
    )

    with pytest.raises(ValueError, match="Unknown pattern"):
        presets.build_surface_sequence(surface, pattern="zigzag")


def test_build_cutting_sequence_covers_slice_saw_and_halving(thesis_math_modules):
    _, _, presets = thesis_math_modules
    food = DummyBody(mins=[0.0, 0.0, 0.0], maxs=[0.3, 0.2, 0.1])

    slice_seq = presets.build_cutting_sequence(
        food, technique="slice", num_cuts_x=2, slice_thickness=0.04
    )
    saw_seq = presets.build_cutting_sequence(food, technique="saw")
    halving_seq = presets.build_cutting_sequence(food, technique="halving")

    assert [phase.name for phase in slice_seq.phases] == [
        "cut_approach_x0",
        "cut_descend_x0",
        "cut_retract_x0",
        "cut_approach_x1",
        "cut_descend_x1",
        "cut_retract_x1",
    ]
    assert len(saw_seq.phases) == 3
    assert [phase.name for phase in saw_seq.phases] == [
        "cut_approach_x0",
        "oscillatory_shear_x0",
        "cut_retract_x0",
    ]
    assert [phase.name for phase in halving_seq.phases] == [
        "cut_approach_x0",
        "cut_descend_x0",
        "cut_retract_x0",
    ]

    slice_first = slice_seq.phases[0].local_curve(0.0)
    slice_last = slice_seq.phases[3].local_curve(0.0)
    assert slice_first == pytest.approx(np.array([0.03, 0.1, 0.145]))
    assert slice_last == pytest.approx(np.array([0.27, 0.1, 0.145]))

    saw_mid = saw_seq.phases[1].local_curve(0.25)
    assert saw_mid == pytest.approx(np.array([0.025, 0.1, 0.06607142857142857]))

    halving_mid = halving_seq.phases[1].local_curve(1.0)
    assert halving_mid == pytest.approx(np.array([0.15, 0.1, 0.05]))

    with pytest.raises(ValueError, match="Unknown cutting technique"):
        presets.build_cutting_sequence(food, technique="dice")
