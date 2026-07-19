from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import List, Optional, Union

from semantic_digital_twin.adapters.multi_sim import MujocoCamera
from semantic_digital_twin.adapters.mujoco_video_recording import (
    MujocoVideoRecorder,
    VideoResolution,
)

from coraplex.exceptions import PlanNotYetPerformedError
from coraplex.plans.plan import Plan


@dataclass
class ActionCaption:
    """
    The span, in seconds on the *encoded video's own timeline*, during which a single leaf
    :class:`~coraplex.plans.plan_node.PlanNode` was performed while recording a
    :class:`RenderedPlanVideo`. Directly comparable to
    :attr:`~semantic_digital_twin.adapters.mujoco_video_recording.RecordedVideo.frame_timestamps`,
    so a caption's span can be used to find the video frames it covers.
    """

    label: str
    """A human-readable description of the node, used as the caption text."""

    start_time: float
    """The video-timeline position at which the node started performing."""

    end_time: float
    """The video-timeline position at which the node finished performing."""


@dataclass
class RenderedPlanVideo:
    """
    The result of rendering a :class:`~coraplex.plans.plan.Plan`'s execution to a video file.
    """

    video_path: Path
    """The path of the encoded video file."""

    captions: List[ActionCaption]
    """One caption per leaf node performed, in performance order."""


@dataclass
class PlanVideoRecorder:
    """
    Renders a video of a previously performed :class:`~coraplex.plans.plan.Plan` by replaying
    it (see :meth:`~coraplex.plans.plan.Plan.prepare_for_replay`) while recording its world
    with a :class:`~semantic_digital_twin.adapters.mujoco_video_recording.MujocoVideoRecorder`.
    """

    plan: Plan
    """The plan to render. Must already have been performed at least once."""

    frames_per_second: int = 30
    """The simulated-time rate at which frames are kept; also the encoded video's frame rate."""

    resolution: VideoResolution = field(
        default_factory=lambda: VideoResolution(width=640, height=480)
    )
    """The pixel resolution of the captured frames."""

    camera: Optional[MujocoCamera] = None
    """
    An existing camera, already attached to :attr:`plan`'s world, to record from. If
    ``None``, a fixed overview camera framing the world's bounding box is attached
    automatically.
    """

    def record(self, output_path: Union[str, Path]) -> RenderedPlanVideo:
        """
        Replays :attr:`plan` from its initial world and writes a video of the replay.

        :param output_path: The file path the video is written to.
        :return: The rendered video's path together with one caption per leaf node.
        """
        if self.plan.initial_world is None:
            raise PlanNotYetPerformedError(plan=self.plan)

        self.plan.prepare_for_replay()

        recorder = MujocoVideoRecorder(
            world=self.plan.world,
            frames_per_second=self.frames_per_second,
            resolution=self.resolution,
            camera=self.camera,
        )
        recorder.start()
        try:
            captions = self._replay_with_captions(recorder)
        finally:
            recorded_video = recorder.stop()

        video_path = recorded_video.write(Path(output_path))
        return RenderedPlanVideo(video_path=video_path, captions=captions)

    def _replay_with_captions(
        self, recorder: MujocoVideoRecorder
    ) -> List[ActionCaption]:
        """
        Performs every leaf node of :attr:`plan`, mirroring :meth:`Plan.replay`'s traversal,
        while recording the video-timeline span of each node.

        :param recorder: The recorder :attr:`plan` is being replayed under, used to read how
            many frames were captured before/after each node.
        :return: One caption per leaf node performed, in performance order.
        """
        captions = []
        for node in self.plan.nodes:
            if not node.is_leaf:
                continue
            start_time = recorder.captured_frame_count / recorder.frames_per_second
            node.perform()
            end_time = recorder.captured_frame_count / recorder.frames_per_second
            captions.append(
                ActionCaption(label=str(node), start_time=start_time, end_time=end_time)
            )
        return captions
