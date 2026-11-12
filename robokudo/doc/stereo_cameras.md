# ROS stereo cameras

`RectifiedDepthCameraInterface` consumes a rectified left image, its camera info,
and aligned depth from an external stereo processor. Select it through
`CollectionReaderDescriptorFactory.create_descriptor` using `stereo`,
`foundation_stereo`, or `zed`. These configurations share one implementation.
FoundationStereo model dependencies stay in the inference environment.

## Integration and duplication

The implementation was adapted from `fork/foundation-stereo-integration`.
Despite its FoundationStereo configuration name, that branch's interface reads
`/ffs/depth` as **depth**, not disparity. It does not acquire a right image or run
stereo inference. This separation also permits other stereo processors.

| Concern | Existing Kinect-style ROS reader | Rectified depth reader |
|---|---|---|
| Inputs | Color, depth, camera info; raw/compressed transports | Raw rectified left image, aligned meter depth, per-frame left camera info |
| Image matching | Approximate timestamps | Exact color/depth acquisition timestamps |
| Calibration | Raw camera matrix `K` | Rectified projection `P`, with zero distortion |
| Work in callback | Image decoding and TF lookup | Retain latest matched acquisition |
| TF | Latest available transform | Transform at image acquisition time |
| Buffering | Existing synchronizer | Bounded synchronizers and one pending frame |
| Frame validity | Existing conversion behavior | Geometry, duplicate and ordering checks |
| Delivered data | Color/depth CAS views and camera observation | Same downstream representation |

The branch repeated image bridging, transform handling and CAS calibration setup.
This port reuses `CVBridgeWorkaround`, `ROSCameraInterface` and
`RosCameraModelAdapter`, and publishes the current semantic `CameraObservation`
instead of assigning obsolete CAS calibration/pose fields. Its separate frame
contract and synchronization retain the stereo-specific behavior without
changing existing RGB-D acquisition. The collection reader waits when a frame is
discarded, while existing interfaces returning `None` continue to report success.

## Input contract

* Color: raw `sensor_msgs/msg/Image`; bridge-supported color or mono encoding,
  converted to contiguous BGR. BGRA from camera drivers is supported.
* Depth: raw `sensor_msgs/msg/Image`, `32FC1`, **meters**, aligned to the left
  rectified pixel grid, with exactly the color acquisition timestamp.
* Calibration: per-frame `sensor_msgs/msg/CameraInfo`, describing the same full
  pixel grid. Intrinsics come from the left projection `P`, not raw `K`.
* All three inputs must have the same nonempty optical frame (`tf_from`) and
  dimensions. Cropping, nonidentity binning and nonstandard left projections
  are rejected. Upstream resizing must publish corresponding calibration.

Depth is copied to native contiguous float32 millimeters for RoboKudo.
Nonfinite, nonpositive and beyond-`max_depth` samples become zero. Source messages
are not modified. The color-to-depth ratio is identity. The image preprocessor
can build its normal point cloud; no external cloud is required.

Example inside an analysis-engine descriptor:

```python
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)

reader_descriptor = CollectionReaderDescriptorFactory.create_descriptor(
    "foundation_stereo",
    topic_color="/left/image_rect_color",
    topic_camera_info="/left/camera_info",
    topic_depth="/ffs/depth",
    tf_from="left_optical_frame",
    lookup_viewpoint=False,
)
```

Pass this descriptor to `CollectionReaderAnnotator` in the pipeline. The generic
`stereo` preset defaults to `/stereo/depth`; override topics and the optical frame
for the producer actually running. The FoundationStereo preset only changes
the depth default to `/ffs/depth`.

## ZED Mini and ZED 2i

Launch the wrapper with `camera_model:=zedm` or `camera_model:=zed2i`.
Topic prefix `~` means `/<camera_name>/<node_name>`; the preset uses
`/zed/zed_node`. The wrapper changed image names in 5.1:

| Stream | Wrapper 5.1+ suffix | Earlier suffix |
|---|---|---|
| Left rectified image | `/left/color/rect/image` | `/left/image_rect_color` |
| Left camera info | `/left/color/rect/camera_info` | `/left/camera_info` |
| Right rectified image | `/right/color/rect/image` | `/right/image_rect_color` |
| Right camera info | `/right/color/rect/camera_info` | `/right/camera_info` |
| Registered depth | `/depth/depth_registered` | `/depth/depth_registered` |

The SDK depth is registered to the left image. Set
`depth.openni_depth_mode:=false` for meter depth; enable depth calculation and
depth/image publication. The wrapper's `rgb` channel is equivalent to `left`.
These details and topic names are documented by
[Stereolabs](https://docs.stereolabs.com/docs/integrations/ros-2/zed-stereo-node).

```python
reader_descriptor = CollectionReaderDescriptorFactory.create_descriptor(
    "zed",
    lookup_viewpoint=False,
)
```

Verify the actual `header.frame_id` and configure `tf_from` accordingly. The
preset expects `zed_left_camera_optical_frame`. Override all three topic paths
and `tf_from` if the camera name/namespace differs; the camera model does not
require a different RoboKudo interface. Use matching image/depth resolutions.

For an external matcher, feed the rectified left/right images and both camera
infos to that processor. Feed its left-aligned metric output into this reader
by overriding `topic_depth`. Preserve acquisition stamps through inference.
Derive calibration and baseline from the actual published messages rather than
hardcoding a ZED Mini or ZED 2i baseline.

## Disparity producers

Convert disparity upstream to `32FC1` meter depth. The ROS
[`stereo_msgs/msg/DisparityImage` contract](https://raw.githubusercontent.com/ros2/common_interfaces/rolling/stereo_msgs/msg/DisparityImage.msg)
defines `Z = f * t / d`; its disparities already account for principal-point
offsets. Respect the valid window and disparity range, reject invalid values,
and preserve left-image geometry, optical frame and acquisition time. Plain
disparity images need explicitly known units, scaling and stereo calibration.

For the ZED wrapper, quantitative disparity is `/disparity/map` in 5.4+;
`/disparity/image` is a visualization. Earlier wrappers use
`/disparity/disparity_image`. Prefer registered depth when using SDK inference.
See the [published topic contract](https://docs.stereolabs.com/docs/integrations/ros-2/zed-stereo-node).

## Timing, QoS and pose

Color/depth matching is exact even when calibration matching is approximate.
`calibration_tolerance` defaults to 0.03 seconds; zero requires exact calibration
timestamps too. Calibration must be published per frame, not only at startup.
The synchronizers retain up to `synchronization_queue_size` messages per input
(default 30). Size this for input frequency multiplied by inference latency,
with headroom, and account for the memory cost of high-resolution images.
DDS histories have a separate `history_depth` (default 2).

Each stream has configurable reliability; defaults are reliable. If a publisher
uses best effort, set the corresponding `color_reliability`, `depth_reliability`
or `calibration_reliability` to `ReliabilityPolicy.BEST_EFFORT`.

Frames are accepted regardless of their age at reception or consumption.
Repeated and out-of-order acquisitions cannot replace newer frames.
Use the same clock and ROS domain
throughout; enable ROS simulated time for bag playback. Restart the reader after
a backward clock jump or a looping bag, since acquisition ordering is monotonic.

For moving cameras, leave `lookup_viewpoint=True`, set `tf_to`, and provide a TF
chain from the left optical frame at acquisition time. Retain TF history covering
inference latency. Missing transforms drop frames. For a fixed pose, disable
lookup and enable `static_camera_transform_enabled`, supplying
`static_world_frame` and `static_world_T_camera`. With both disabled, observations
have no world pose and support camera-frame processing.

`statistics` exposes matched, replaced, obsolete and missing-transform
counts. Filter-queue evictions are not included. When no frames match, check
topic types, QoS, exact acquisition stamps, calibration publication and buffering.

Automated tests cover message contracts, matching, conversion, calibration,
source selection and dropped-frame handling. Physical ZED Mini/2i acquisition,
depth accuracy and the installed wrapper configuration still need hardware
validation.
