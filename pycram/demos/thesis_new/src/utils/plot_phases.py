import os

import numpy as np
import matplotlib

if os.environ.get("DISPLAY"):
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from demos.thesis.simulation_setup import add_box, BoxSpec
from pycram.robot_plans.actions.composite.thesis_math.frame_provider import (
    WorldTransformFrameProvider,
)
from demos.thesis_new.old.geometry_utils import aligned_plane_frame
from pycram.robot_plans.actions.composite.thesis_math.motion_models import (
    Pose,
    FixedFrameProvider,
)
from pycram.robot_plans.actions.composite.thesis_math.motion_presets import (
    build_default_sequence,
    build_container_sequence,
)
from pycram.robot_plans.actions.composite.thesis_math.motion_profiles import (
    ShearProfile,
    SpiralProfile,
    SweepProfile,
    oscillatory_shear_local_profiled,
    planar_spiral_xy,
    planar_sweep_x,
    sample_local_curve,
)
from pycram.robot_plans.actions.composite.thesis_math.world_utils import (
    try_get_body,
    make_identity_pose_stamped,
    Rp_from_spatial,
)
from pycram.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection


def _setup_world_with_bowl():
    """Create a world with a bowl object for plotting."""
    world = setup_world()
    bowl = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "../../..",
            "..",
            "resources",
            "objects",
            "bowl.stl",
        )
    ).parse()
    with world.modify_world():
        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.4, 2.2, 1, reference_frame=world.root
            ),
        )
    return world


def plot_profiles():
    """Plot local phase profiles for parameter sweeps."""
    taus = np.linspace(0.0, 1.0, 900)

    shear_profiles = [
        (
            "set A: small amp, high freq",
            ShearProfile(
                depth_max=0.05, depth_ramp_end=0.7, shear_amp=0.006, shear_cycles=9.0
            ),
        ),
        (
            "set B: medium amp, medium freq",
            ShearProfile(
                depth_max=0.05, depth_ramp_end=0.7, shear_amp=0.012, shear_cycles=5.0
            ),
        ),
        (
            "set C: large amp, low freq",
            ShearProfile(
                depth_max=0.05, depth_ramp_end=0.7, shear_amp=0.020, shear_cycles=2.5
            ),
        ),
    ]

    spiral_profiles = [
        ("set A: tight spiral", SpiralProfile(r0=0.00, r1=0.10, cycles=5.0)),
        ("set B: medium spiral", SpiralProfile(r0=0.00, r1=0.14, cycles=3.0)),
        ("set C: wide spiral", SpiralProfile(r0=0.00, r1=0.18, cycles=1.8)),
    ]

    sweep_profiles = [
        ("set A: short sweep, high freq", SweepProfile(length=0.06, cycles=4.0)),
        ("set B: medium sweep, medium freq", SweepProfile(length=0.10, cycles=2.0)),
        ("set C: long sweep, low freq", SweepProfile(length=0.14, cycles=1.2)),
    ]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for label, prof in shear_profiles:
        curve = lambda tau, prof=prof: oscillatory_shear_local_profiled(tau, prof)
        P = sample_local_curve(curve, taus)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], label=label)
    ax.set_xlabel("x (shear)")
    ax.set_ylabel("y")
    ax.set_zlabel("z (monotone)")
    ax.set_title("Same local phase: oscillatory shear (parameter variation)")
    ax.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    for label, prof in shear_profiles:
        curve = lambda tau, prof=prof: oscillatory_shear_local_profiled(tau, prof)
        P = sample_local_curve(curve, taus)
        ax.plot(taus, P[:, 0], label=label)
    ax.set_xlabel("tau")
    ax.set_ylabel("x")
    ax.set_title("Shear component x(tau) under different parameter sets")
    ax.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    for label, prof in shear_profiles:
        curve = lambda tau, prof=prof: oscillatory_shear_local_profiled(tau, prof)
        P = sample_local_curve(curve, taus)
        ax.plot(taus, P[:, 2], label=label)
    ax.set_xlabel("tau")
    ax.set_ylabel("z")
    ax.set_title("Monotone displacement z(tau) under different parameter sets")
    ax.legend()
    plt.show()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    for label, prof in spiral_profiles:
        curve = lambda tau, prof=prof: planar_spiral_xy(
            tau, prof.r0, prof.r1, prof.cycles
        )
        P = sample_local_curve(curve, taus)
        ax.plot(P[:, 0], P[:, 1], label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Same local phase: planar spiral (parameter variation)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    for label, prof in sweep_profiles:
        curve = lambda tau, prof=prof: planar_sweep_x(tau, prof.length, prof.cycles)
        P = sample_local_curve(curve, taus)
        ax.plot(taus, P[:, 0], label=label)
    ax.set_xlabel("tau")
    ax.set_ylabel("x")
    ax.set_title("Same local phase: planar sweep x(tau) (parameter variation)")
    ax.legend()
    plt.show()


def plot_sequence_in_frames():
    """Plot the same sequence under two different frames."""
    seq = build_default_sequence()
    R_A, p_A = aligned_plane_frame([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    R_B, p_B = aligned_plane_frame(
        [0.35, 0.10, 0.20], [0.3, 0.4, 0.85], [0.0, 1.0, 0.0]
    )

    prov_A = FixedFrameProvider(Pose(R_A, p_A))
    prov_B = FixedFrameProvider(Pose(R_B, p_B))

    dt = 0.01
    tA, PA, idA = seq.sample(prov_A, dt=dt)
    tB, PB, _ = seq.sample(prov_B, dt=dt)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(PA[:, 0], PA[:, 1], PA[:, 2], label="sequence + frame A")
    ax.plot(PB[:, 0], PB[:, 1], PB[:, 2], label="sequence + frame B")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("MotionSequence reused under two different frame alignments")
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(tA, PA[:, 2])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("z [m]")
    ax.set_title("Sequence timing visible in z(t) for frame A")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(tA, idA, where="post")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("phase index")
    ax.set_title("MotionSegment index over time (frame A)")
    plt.show()


def plot_sequence_in_world():
    """Plot a sequence aligned to a world-frame pose."""
    world = _setup_world_with_bowl()
    with world.modify_world():
        box = add_box(
            world, BoxSpec(name="muh_box", scale_xyz=(0.3, 0.1, 0.1)), tf_frame="/map"
        )
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.9, y=1.0, z=0.95
                ),
            )
        )

    rf = world.get_body_by_name("muh_box").global_pose
    R, p = Rp_from_spatial(rf.evaluate())
    F = Pose(R=R, p=p)
    prov = FixedFrameProvider(F)

    seq = build_default_sequence()
    dt = 0.01
    tA, PA, idA = seq.sample(prov, dt=dt)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(PA[:, 0], PA[:, 1], PA[:, 2], label="sequence in world (provider=F)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("MotionSequence under world frame F")
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(tA, PA[:, 2])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("z [m]")
    ax.set_title("z(t)")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(tA, idA, where="post")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("phase index")
    ax.set_title("phase index over time")
    plt.show()


def plot_bowl_sequence():
    """Plot the bowl-constrained sequence in the bowl frame."""
    world = _setup_world_with_bowl()
    bowl_body = try_get_body(world, "bowl.stl")
    if bowl_body is None:
        print("[info] body 'bowl.stl' not found, skipping bowl plot.")
        return

    seq_container = build_container_sequence(bowl_body)
    prov_container = WorldTransformFrameProvider(
        world=world,
        source_frame=bowl_body,
        root_frame=world.root,
        make_identity_spatial=make_identity_pose_stamped,
    )
    t_bowl, P_container, _ = seq_container.sample(prov_container, dt=0.01)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        P_container[:, 0],
        P_container[:, 1],
        P_container[:, 2],
        label="sequence in bowl",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("MotionSequence constrained to bowl (object frame)")
    ax.legend()
    plt.show()


def main():
    """Run all plotting demos."""
    plot_profiles()
    plot_sequence_in_frames()
    plot_sequence_in_world()
    plot_bowl_sequence()


if __name__ == "__main__":
    main()
