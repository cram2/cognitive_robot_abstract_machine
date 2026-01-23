import math
import threading
import time

import numpy as np
import pytest

from giskardpy.ros2_tools.force_torque_filter_node import (
    OffsetEstimator,
    LowPassButterworthFilter,
    DerivativeEstimator,
    FilterConfig,
    WrenchProcessor,
    ForceTorqueFilterNode,
)


def test_offset_estimator_tracks_constant_offset():
    est = OffsetEstimator(alpha=0.01)
    offset = 2.5
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.1, size=2000)
    xs = offset + noise
    last = None
    for x in xs:
        last = est.update(float(x))
    assert abs(last - offset) < 0.15


def test_offset_estimator_initializes_to_first_sample():
    est = OffsetEstimator(alpha=0.5)
    first = 42.0
    assert est.update(first) == pytest.approx(first)
    # Next update should move toward the new value
    y = est.update(50.0)
    assert first < y < 50.0


def test_lowpass_butter_reduces_high_freq():
    fs = 100.0
    filt = LowPassButterworthFilter(cutoff_hz=5.0, sampling_frequency=fs, order=3)
    t = np.arange(0, 2.0, 1.0 / fs)
    low = np.sin(2 * np.pi * 1 * t)
    high = 0.5 * np.sin(2 * np.pi * 30 * t)
    x = low + high
    y = np.array([filt.step(float(xi)) for xi in x])
    # Basic check: overall std should be reduced
    assert y.std() < x.std()


def test_derivative_of_linear_ramp_is_constant():
    der = DerivativeEstimator()
    dt = 0.01
    slope = 3.0
    xs = [slope * i * dt for i in range(200)]
    ys = [der.step(x, dt) for x in xs]
    ys = ys[1:]
    assert np.mean(ys) == pytest.approx(slope, abs=1e-2)


ess = np.finfo(float).eps


def test_wrench_processor_filters_and_derives():
    fs = 100.0
    cfg = FilterConfig(
        topic_in="/in",
        expected_rate_hz=fs,
        cutoff_main_hz=5.0,
        order_main=3,
        cutoff_diff_hz=3.0,
        order_diff=2,
        offset_mode="ewma",
        offset_alpha=0.01,
        warmup_samples=0,
    )
    proc = WrenchProcessor(cfg)

    from geometry_msgs.msg import Vector3, Wrench

    dt = 1.0 / fs
    offset = 10.0
    amp = 2.0
    freq = 1.0
    t = np.arange(0, 3.0, dt)
    x = offset + amp * np.sin(2 * np.pi * freq * t)

    filt_vals = []
    diff_vals = []
    for xi in x:
        w = Wrench(
            force=Vector3(x=float(xi), y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        wf, wd = proc.process(w, dt)
        filt_vals.append(wf.force.x)
        diff_vals.append(wd.force.x)

    filt_vals = np.array(filt_vals)
    diff_vals = np.array(diff_vals)

    # Offset should be largely removed
    assert abs(np.mean(filt_vals)) < 1.0

    # Derivative amplitude envelope roughly matches expected amplitude of derivative
    expected_amp = amp * 2 * np.pi * freq
    assert 0.3 * expected_amp < np.max(np.abs(diff_vals)) < 1.5 * expected_amp


@pytest.mark.timeout(30)
@pytest.mark.usefixtures("rclpy_node")
def test_node_publishes_after_warmup(rclpy_node):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from geometry_msgs.msg import WrenchStamped, Vector3, Wrench

    # Create and spin the filter node in its own executor thread
    node = ForceTorqueFilterNode()
    exec_filter = SingleThreadedExecutor()
    exec_filter.add_node(node)
    th = threading.Thread(target=exec_filter.spin, daemon=True)
    th.start()
    time.sleep(0.05)

    # Helper containers on the test node
    recv_filtered = []
    recv_diff = []

    # Subscriptions on the fixture node to the filter outputs
    rclpy_node.create_subscription(
        WrenchStamped,
        node.config.topic_filtered_out,
        lambda m: recv_filtered.append(m),
        10,
    )
    rclpy_node.create_subscription(
        WrenchStamped,
        node.config.topic_derivative_out,
        lambda m: recv_diff.append(m),
        10,
    )

    # Publisher on the fixture node to the filter input
    pub = rclpy_node.create_publisher(WrenchStamped, node.config.topic_in, 10)

    fs = node.config.expected_rate_hz
    dt = 1.0 / fs

    def publish_sequence(
        n_samples: int, offset: float = 5.0, amp: float = 1.0, freq: float = 1.0
    ):
        for i in range(n_samples):
            t = i * dt
            msg = WrenchStamped()
            # Set increasing timestamps
            sec = int(t)
            nanosec = int((t - sec) * 1e9)
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
            msg.wrench = Wrench(
                force=Vector3(
                    x=float(offset + amp * math.sin(2 * math.pi * freq * t)),
                    y=0.0,
                    z=0.0,
                ),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            )
            pub.publish(msg)
            # Let callbacks run in both executors
            rclpy.spin_once(rclpy_node, timeout_sec=0.0)
            time.sleep(0.002)

    try:
        # Warmup: publish exactly warmup_samples samples; no output expected
        publish_sequence(node.config.warmup_samples)
        # Allow processing
        for _ in range(10):
            rclpy.spin_once(rclpy_node, timeout_sec=0.01)
            time.sleep(0.005)
        assert len(recv_filtered) == 0
        assert len(recv_diff) == 0

        # Now publish additional samples beyond warmup
        publish_sequence(200)
        # Wait for messages to arrive
        for _ in range(50):
            rclpy.spin_once(rclpy_node, timeout_sec=0.01)
            time.sleep(0.005)

        assert len(recv_filtered) > 0
        assert len(recv_diff) > 0

        # Numerical sanity: filtered mean near zero (offset removed partly)
        filt_fx = np.array([m.wrench.force.x for m in recv_filtered])
        assert abs(np.mean(filt_fx)) < 1.0
    finally:
        exec_filter.shutdown()
        th.join(timeout=2.0)
        node.destroy_node()
