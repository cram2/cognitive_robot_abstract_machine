#!/usr/bin/env python3
"""
ROS 2 node and utilities to filter and differentiate force-torque signals.

Provides a configurable low-pass filtering pipeline with optional offset
removal and a smoothed first derivative for six-axis wrench data.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Vector3, Wrench
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from scipy.signal import butter, sosfilt


@dataclass
class FilterConfig:
    """
    Config file for the force torque filter node.
    """

    topic_in: str = "/in"
    """
    Name of the topic with the unfiltered force torque data.
    """
    topic_filtered_out_suffix: str = "filtered"
    """
    Suffix to append to the topic_in topic name to produce the filtered output topic.
    """
    topic_derivative_out_suffix: str = "filtered/derivative"
    """
    Suffix to append to the topic_in topic name to produce the first time derivative the filtered output topic.
    """
    expected_rate_hz: float = 100.0
    """
    Expected rate of the input topic, used to compute the filter cutoff frequencies.
    """
    cutoff_main_hz: float = 5.0
    """
    Cutoff frequency for the main low-pass filter.
    """
    order_main: int = 3
    """
    Order of the main low-pass filter.
    """
    cutoff_diff_hz: float = 3.0
    """
    Cutoff frequency for the derivative low-pass filter.
    """
    order_diff: int = 2
    """
    Order of the derivative low-pass filter.
    """
    offset_mode: str = "ewma"
    """
    Method for offset removal from the signal. Options:
    - "ewma": Use exponentially weighted moving average to estimate and remove offset
    - "none": Disable offset removal
    """

    offset_alpha: float = 0.005
    """
    Learning rate for the EWMA offset estimator (only used when offset_mode="ewma").
    Higher values adapt faster to offset changes but are more sensitive to noise.
    Valid range: (0.0, 1.0), typical values: 0.001 to 0.01
    """

    warmup_samples: int = 50
    """
    Number of initial samples to discard before publishing filtered output.
    This prevents startup transients from affecting the filtered signal.
    The node will process but not publish the first N samples.
    """

    qos_depth: int = 10
    """
    Quality of Service history depth for ROS 2 topics.
    Number of messages to keep in the publisher/subscriber queue.
    """

    qos_reliable: bool = True
    """
    Quality of Service reliability policy for ROS 2 topics.
    - True: Use RELIABLE delivery (guarantees message delivery)
    - False: Use BEST_EFFORT delivery (faster but may drop messages)
    """

    def __post_init__(self) -> None:
        """
        Run validation after dataclass initialization.

        Ensures that the configuration values are consistent before use.
        """
        self.sanity_check_params()

    def sanity_check_params(self):
        """
        Validate configuration values.

        Ensures that all numerical parameters are within acceptable ranges and
        that categorical parameters use supported values.
        """
        if self.offset_mode not in ("ewma", "none"):
            raise ValueError("Parameter 'offset_mode' must be 'ewma' or 'none'")
        if self.order_main <= 0 or self.order_diff <= 0:
            raise ValueError("Filter orders must be positive integers")
        if self.expected_rate_hz <= 0.0:
            raise ValueError("expected_rate_hz must be > 0")

    @property
    def topic_filtered_out(self) -> str:
        """
        Build the topic name for the filtered wrench output.

        Concatenates the input topic with the configured filtered suffix.

        :return: Full topic name for the filtered wrench output.
        """
        return (
            self.topic_in.rstrip("/") + "/" + self.topic_filtered_out_suffix.lstrip("/")
        )

    @property
    def topic_derivative_out(self) -> str:
        """
        Build the topic name for the derivative output.

        Concatenates the input topic with the configured derivative suffix.

        :return: Full topic name for the derivative wrench output.
        """
        return (
            self.topic_in.rstrip("/")
            + "/"
            + self.topic_derivative_out_suffix.lstrip("/")
        )

    @classmethod
    def from_ros2_params(cls, node: Node) -> FilterConfig:
        """
        Build a configuration from ROS 2 node parameters.

        Declares parameters with defaults on the provided node and
        then reads their current values into a new configuration.

        :param node: ROS 2 node used to declare and read parameters.
        :return: New configuration populated from the node parameters.
        """
        cls._declare_parameters_from_config(node)
        return cls._read_parameters_to_config(node)

    @classmethod
    def _declare_parameters_from_config(cls, node: Node) -> None:
        """
        Declare all configuration fields as ROS 2 parameters on the node.

        Uses type-appropriate neutral defaults for fields without explicit
        defaults so that parameters exist and can be overridden externally.

        :param node: Node on which parameters will be declared.
        """
        for f in dataclasses.fields(cls):
            # Compute a neutral default for required fields (no default provided)
            if f.default is not dataclasses.MISSING:
                default_val = f.default
            elif getattr(f, "default_factory", dataclasses.MISSING) is not dataclasses.MISSING:  # type: ignore[attr-defined]
                default_val = f.default_factory()  # type: ignore[misc]
            else:
                # Required without default: declare with neutral type-based default
                if f.type is str:
                    default_val = ""
                elif f.type is bool:
                    default_val = False
                elif f.type is int:
                    default_val = 0
                else:
                    # fallback float/others
                    default_val = 0.0
            node.declare_parameter(f.name, default_val)

    @classmethod
    def _read_parameters_to_config(cls, node: Node) -> FilterConfig:
        """
        Read ROS 2 parameters into a new configuration instance.

        Uses typed accessors to remain compatible across ROS 2 distributions.

        :param node: Node from which parameters are read.
        :return: Configuration populated from the node parameters.
        """

        # Typed getters for maximum ROS 2 compatibility across distros
        def get_str(name: str) -> str:
            return str(node.get_parameter(name).get_parameter_value().string_value)

        def get_bool(name: str) -> bool:
            return bool(node.get_parameter(name).get_parameter_value().bool_value)

        def get_int(name: str) -> int:
            return int(node.get_parameter(name).get_parameter_value().integer_value)

        def get_float(name: str) -> float:
            return float(node.get_parameter(name).get_parameter_value().double_value)

        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.type == "str":
                kwargs[f.name] = get_str(f.name)
            elif f.type == "bool":
                kwargs[f.name] = get_bool(f.name)
            elif f.type == "int":
                kwargs[f.name] = get_int(f.name)
            else:
                # Treat anything else as float in this config
                kwargs[f.name] = get_float(f.name)

        return cls(**kwargs)


@dataclass
class OffsetEstimator:
    """
    Estimates and removes a constant or slowly varying offset via EWMA.
    """

    alpha: float
    """
    Learning rate for the EWMA offset estimator (only used when offset_mode="ewma").
    Higher values adapt faster to offset changes but are more sensitive to noise.
    Valid range: (0.0, 1.0), typical values: 0.001 to 0.01
    """
    current_value: float = 0.0
    """
    Current estimated offset value.
    """
    initialized: bool = False
    """
    Indicates whether the estimator has received its first sample.
    """

    def update(self, x: float) -> float:
        """
        Update offset with new sample and return the offset estimate.

        :param x: New sample value.
        :return: Current estimated offset after the update.
        """
        if not self.initialized:
            self.current_value = float(x)
            self.initialized = True
            return self.current_value
        self.current_value = (1.0 - float(self.alpha)) * self.current_value + float(
            self.alpha
        ) * float(x)
        return self.current_value


@dataclass
class LowPassButterworthFilter:
    """
    Stateful Butterworth low-pass filter using Second-Order-Sections form for stability.
    """

    cutoff_hz: float
    """
    Cutoff frequency of the low-pass filter in hertz.
    """
    sampling_frequency: float
    """
    Sampling frequency of the input signal in hertz.
    """
    order: int
    """
    Order of the Butterworth filter.
    """
    second_order_sections: np.ndarray = field(init=False, repr=False)
    """
    Internal filter coefficients that define how the low-pass filter processes signals.

    This array contains the mathematical recipe for the Butterworth filter, broken down
    into smaller, more stable pieces called "second-order sections". Think of it like
    having a complex filter recipe split into simpler sub-recipes that work together.

    Why this approach?
    - **Stability**: Large filters can become numerically unstable when represented as 
      a single equation. Breaking them into smaller pieces prevents this.
    - **Accuracy**: Each smaller piece maintains better precision during calculations.

    Technical details:
    - Shape: (n_sections, 6) where n_sections ≈ filter_order/2
    - Each row contains 6 coefficients [b0, b1, b2, a0, a1, a2] for one filter section
    - Created automatically from filter parameters (cutoff frequency, order, sample rate)
    - Used internally by scipy.signal.sosfilt() for real-time filtering

    You don't need to modify this directly - it's automatically generated when the
    filter is initialized with your desired cutoff frequency and filter order.
    """

    filter_state: np.ndarray = field(init=False, repr=False)
    """
    Internal filter state for streaming processing.
    """

    def __post_init__(self) -> None:
        nyq = 0.5 * float(self.sampling_frequency)
        wn = float(self.cutoff_hz) / nyq
        self.second_order_sections = butter(
            int(self.order), wn, btype="low", output="sos"
        )
        # zi per SOS section: shape (n_sections, 2)
        self.filter_state = np.zeros((self.second_order_sections.shape[0], 2))

    def step(self, x: float) -> float:
        """
        Process a single sample through the filter.

        :param x: Input sample.
        :return: Filtered sample.
        """
        y, self.filter_state = sosfilt(
            self.second_order_sections, [x], zi=self.filter_state
        )
        return float(y[-1])


@dataclass
class DerivativeEstimator:
    """
    First derivative via sample-to-sample difference using provided dt.
    """

    previous: Optional[float] = None
    """
    Previous sample value used to compute the discrete derivative.
    """

    def step(self, x: float, dt: float) -> float:
        """
        Compute the first derivative using the previous sample and time step.

        Returns zero on the first call or when the provided time step is not
        positive.

        :param x: Current sample.
        :param dt: Time step since the previous sample in seconds.
        :return: Estimated derivative value.
        """
        if dt <= 0.0 or self.previous is None:
            self.previous = float(x)
            return 0.0
        dx = (float(x) - float(self.previous)) / float(dt)
        self.previous = float(x)
        return float(dx)


@dataclass
class WrenchProcessor:
    """
    Composes offset removal, main low-pass, derivative, and derivative smoothing for six axes.
    """

    config: FilterConfig
    """
    Configuration that defines filter and derivative behavior.
    """
    remove_offset: bool = field(init=False)
    """
    Whether offset removal is enabled based on the configuration.
    """
    offset: list = field(init=False, repr=False)
    """
    Per-axis offset estimators for the six wrench components.
    """
    main: list = field(init=False, repr=False)
    """
    Per-axis main low-pass filters for the six wrench components.
    """
    diff: list = field(init=False, repr=False)
    """
    Per-axis discrete derivative estimators for the six wrench components.
    """
    diff_smooth: list = field(init=False, repr=False)
    """
    Per-axis smoothing low-pass filters applied to the derivative signals.
    """

    def __post_init__(self) -> None:
        """
        Initialize per-axis processors according to the configuration.

        Creates optional offset estimators, main low-pass filters, derivative
        estimators, and derivative smoothing filters for each of the six axes.

        :return: None
        """
        fs = float(self.config.expected_rate_hz)
        self.remove_offset = self.config.offset_mode != "none"
        self.offset = [
            OffsetEstimator(self.config.offset_alpha) if self.remove_offset else None
            for _ in range(6)
        ]
        self.main = [
            LowPassButterworthFilter(
                self.config.cutoff_main_hz, fs, self.config.order_main
            )
            for _ in range(6)
        ]
        self.diff = [DerivativeEstimator() for _ in range(6)]
        self.diff_smooth = [
            LowPassButterworthFilter(
                self.config.cutoff_diff_hz, fs, self.config.order_diff
            )
            for _ in range(6)
        ]

    @staticmethod
    def _axes_from_wrench(w: Wrench) -> List[float]:
        """
        Extract the six scalar components from a wrench message.

        Returns force (x, y, z) followed by torque (x, y, z).

        :param w: Input wrench message.
        :return: List of six values: force (x, y, z) then torque (x, y, z).
        """
        return [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]

    @staticmethod
    def _wrench_from_axes(a: List[float]) -> Wrench:
        """
        Build a wrench message from six scalar components.

        Interprets the list as force (x, y, z) followed by torque (x, y, z).

        :param a: Six values: force (x, y, z) then torque (x, y, z).
        :return: Wrench message with forces and torques set.
        """
        f = Vector3(x=float(a[0]), y=float(a[1]), z=float(a[2]))
        t = Vector3(x=float(a[3]), y=float(a[4]), z=float(a[5]))
        return Wrench(force=f, torque=t)

    def process(self, w: Wrench, dt: float) -> Tuple[Wrench, Wrench]:
        """
        Filter a wrench and compute its smoothed first derivative.

        Applies optional offset removal, a main low-pass filter, a discrete
        derivative, and a secondary low-pass on the derivative for each axis.
        Returns a tuple of the filtered wrench and the derivative wrench.

        :param w: Input wrench to process.
        :param dt: Time step in seconds between samples.
        :return: Tuple of (filtered wrench, derivative wrench).
        """
        axes = self._axes_from_wrench(w)
        filtered = [0.0] * 6
        deriv = [0.0] * 6
        for i, x in enumerate(axes):
            x_in = float(x)
            if self.remove_offset and self.offset[i] is not None:
                x_in = x_in - self.offset[i].update(x_in)
            xf = self.main[i].step(x_in)
            dxf = self.diff[i].step(xf, dt)
            dxf_sm = self.diff_smooth[i].step(dxf)
            filtered[i] = xf
            deriv[i] = dxf_sm
        return self._wrench_from_axes(filtered), self._wrench_from_axes(deriv)


@dataclass(eq=False)
class ForceTorqueFilterNode(Node):
    """
    ROS 2 node: subscribes to a raw WrenchStamped and publishes filtered and derivative topics.

    It publishes two topics:
    - filtered: the denoised signal with offset removed
    - filtered/diff: the first derivative of the filtered signal with reduced noise
    """

    config: FilterConfig | None = None
    """
    Optional configuration to initialize the node. If not provided, parameters are read from the ROS 2 node.
    """
    processor: WrenchProcessor = field(init=False, repr=False)
    """
    Internal processor that performs filtering and derivative computation.
    """
    last_stamp_ns: Optional[int] = None
    """
    Timestamp of the last processed message in nanoseconds.
    """
    sample_count: int = 0
    """
    Count of received samples used for warm-up gating.
    """

    def __post_init__(self) -> None:
        """
        Set up the ROS 2 node, parameters, quality of service, and topics.

        Creates the processor pipeline, declares and reads parameters, and
        wires subscriptions and publishers for filtered and derivative output.

        :return: None
        """
        super().__init__("force_torque_filter")

        # Declare parameters with defaults, then read actual values
        if self.config is None:
            self.config = FilterConfig.from_ros2_params(self)

        qos = QoSProfile(
            reliability=(
                ReliabilityPolicy.RELIABLE
                if self.config.qos_reliable
                else ReliabilityPolicy.BEST_EFFORT
            ),
            history=HistoryPolicy.KEEP_LAST,
            depth=self.config.qos_depth,
        )

        self.processor = WrenchProcessor(self.config)

        self.sub = self.create_subscription(
            WrenchStamped, self.config.topic_in, self._on_msg, qos
        )
        self.pub_filtered = self.create_publisher(
            WrenchStamped, self.config.topic_filtered_out, qos
        )
        self.pub_diff = self.create_publisher(
            WrenchStamped, self.config.topic_derivative_out, qos
        )

        self.get_logger().info("ForceTorqueFilterNode started")

    def _on_msg(self, msg: WrenchStamped) -> None:
        """
        Callback that processes an incoming wrench sample.

        - Computes dt from the message header timestamp
        - Processes the signal to produce filtered and derivative outputs
        - Applies a warm-up gate to avoid startup transients

        :param msg: Incoming wrench message with header timestamp.
        :return: None
        """
        stamp = msg.header.stamp
        cur_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        dt = 0.0
        if self.last_stamp_ns is not None:
            dt = max(0.0, (cur_ns - self.last_stamp_ns) / 1e9)
        self.last_stamp_ns = cur_ns

        self.sample_count += 1
        dt_use = dt if dt > 0.0 else 1.0 / max(1e-6, self.config.expected_rate_hz)

        filt_wrench, diff_wrench = self.processor.process(msg.wrench, dt_use)

        # Warmup gate to avoid startup transients
        # Do not publish for the first `warmup_samples` messages, inclusive of the Nth sample
        # so that exactly publishing `warmup_samples` inputs produces zero outputs.
        if self.sample_count <= self.config.warmup_samples:
            return

        out_filtered = WrenchStamped()
        out_filtered.header = msg.header
        out_filtered.wrench = filt_wrench
        self.pub_filtered.publish(out_filtered)

        out_diff = WrenchStamped()
        out_diff.header = msg.header
        out_diff.wrench = diff_wrench
        self.pub_diff.publish(out_diff)


def main():
    """
    Run the force-torque filter as a ROS 2 node.

    Initializes rclpy, spins the node, and shuts down on exit.

    """
    rclpy.init()
    node = ForceTorqueFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
