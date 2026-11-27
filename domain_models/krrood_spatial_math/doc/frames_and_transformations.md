---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Frames, poses and transformations

This page explains how the package describes where one frame is relative to another, and
how such descriptions combine. It covers the homogeneous transformation matrix, the
points and vectors it acts on, chains of transformations and their inverses, and the
forms the package writes a rotation in. The names follow semantic_digital_twin's
[style guide](https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/style_guide.html).

```{code-cell} ipython3
:tags: [hide-input]

import json
import math
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import krrood.symbolic_math.symbolic_math as sm
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from krrood_spatial_math.cartesian_frame_3_dimensional import (
    CartesianFrame3Dimensional,
)
from krrood_spatial_math.embedded_plane import EmbeddedPlane
from krrood_spatial_math.space import EuclideanSpace
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix3Dimensional,
    Point3Dimensional,
    Quaternion,
    RotationMatrix,
    ScrewAxis,
    Vector3Dimensional,
)


def show(figure, script=None):
    """
    :param script: JavaScript to run once the figure is drawn, in which
        ``{plot_id}`` stands for the id of the figure's element.
    """
    return HTML(
        figure.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            include_mathjax=False,
            config=dict(displayModeBar=False),
            post_script=script,
        )
    )


def numbers(*values, digits=2):
    return (
        "("
        + ", ".join(
            f"{round(float(v), digits) + 0.0:.{digits}f}" for v in values
        )
        + ")"
    )


def slider(labels, prefix, active, states, traces):
    """
    A slider whose steps change, in place, the data of the given traces and some
    entries of the layout, so that nothing else of the figure, such as its axes,
    moves.

    :param states: For each step, the new traces and the changed layout entries.
    """

    def trace_changes(state_traces):
        changes = {}
        for key in ("x", "y", "z", "name"):
            values = [getattr(trace, key, None) for trace in state_traces]
            if all(value is not None for value in values):
                changes[key] = [
                    value
                    if isinstance(value, str)
                    else np.asarray(value).tolist()
                    for value in values
                ]
        return changes

    return dict(
        active=active,
        currentvalue=dict(prefix=prefix),
        pad=dict(t=30),
        steps=[
            dict(
                label=label,
                method="update",
                args=[trace_changes(state_traces), layout_changes, traces],
            )
            for label, (state_traces, layout_changes) in zip(labels, states)
        ],
    )


legend_box = dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.85)")
axis_colors = ("#d62728", "#2ca02c")


def arrow(start, end, color, name, width=4, dash=None):
    return go.Scatter(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        mode="lines+markers",
        line=dict(color=color, width=width, dash=dash),
        marker=dict(
            symbol="arrow", angleref="previous", size=[0, 14], color=color
        ),
        name=name,
    )


def frame_axes(transformation, label, length=0.4, width=4, dash=None):
    """
    The x and y axes of the frame a planar transformation places, as arrows.
    """
    matrix = np.asarray(transformation)
    origin = matrix[:2, 3]
    return [
        arrow(
            origin,
            origin + length * matrix[:2, column],
            axis_colors[column],
            f"{label}'s {'xy'[column]} axis",
            width=width,
            dash=dash,
        )
        for column in range(2)
    ]


def frame_label(transformation, label):
    """
    The name of a frame, written next to its origin.
    """
    matrix = np.asarray(transformation)
    return go.Scatter(
        x=[matrix[0, 3] - 0.08],
        y=[matrix[1, 3] - 0.08],
        mode="text",
        text=[label],
        textfont=dict(size=16),
        name=label,
        showlegend=False,
    )


def planar_figure(height, x_range, y_range):
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=45, r=5, t=5, b=0),
        legend=dict(**legend_box, font=dict(size=12)),
        xaxis=dict(range=x_range, title="x (m)"),
        yaxis=dict(range=y_range, title="y (m)", scaleanchor="x"),
    )
    return figure


def turning(angle, x=0.0, y=0.0, **frames):
    return HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        x=x, y=y, yaw=angle, **frames
    )
```

## Frames

A frame is an origin and three axes of unit length, perpendicular to each other and
right-handed, with lengths in metres. Anything that implements
{py:class}`~krrood_spatial_math.cartesian_frame_3_dimensional.CartesianFrame3Dimensional` is one. In
semantic_digital_twin, every body is a frame. On this page, a frame is just a name:

```{code-cell} ipython3
@dataclass(eq=False)
class Frame(CartesianFrame3Dimensional[str]):
    name: str
    space: EuclideanSpace | None = None
    id: UUID = field(default_factory=uuid4)

    def transform_into_own_frame(self, *, value):
        return value

    @property
    def z_axis(self):
        return Vector3Dimensional.Z(reference_frame=self)

    @property
    def embedded_plane(self):
        return EmbeddedPlane(frame_3_dimensional=self)


a, b, c = Frame("a"), Frame("b"), Frame("c")
```

## The transformation between two frames

The homogeneous transformation matrix $_aT_b$, written `a_T_b` in code, says where
frame $b$ is relative to frame $a$:

$$
{}_aT_b = \begin{bmatrix} R & p \\ 0\;0\;0 & 1 \end{bmatrix},
$$

where $R$ is a $3 \times 3$ rotation matrix and $p$ a column of three numbers. The
columns of $R$ are $b$'s x, y and z axes, and $p$ is $b$'s origin, all given in $a$'s
coordinates {cite}`lynch2017modern` (section 3.3.1). The same matrix is therefore also
the *pose* of $b$ in $a$.

It also turns coordinates given in $b$ into coordinates of the same point given in $a$.
A point $q$ with the coordinates $_bq = (x, y, z)$ in $b$ lies $x$ along $b$'s x axis,
$y$ along its y axis and $z$ along its z axis, starting from $b$'s origin. In $a$'s
coordinates, that is

$$
{}_aq = p + x\,R_{:,1} + y\,R_{:,2} + z\,R_{:,3} = R\,{}_bq + p,
$$

where $R_{:,k}$ is the $k$-th column of $R$. Writing the point with a fourth coordinate
1, as $(x, y, z, 1)$, turns this into a single product:
$_aq = {}_aT_b\;{}_bq$. This is why the matrix has its fourth row.

The figure turns frame $b$ about its origin, by the angle $\theta$ about the z axis,
which points out of the page. The point $q$ is fixed in $b$, at
$_bq = (0.6, 0.3, 0)$. Its coordinates in $b$ never change; its coordinates in $a$ do.
The box at the top left shows $_aT_b$ with its current values. Its first two columns
are $b$'s x and y axes, drawn as arrows, and its last column is $b$'s origin. Drag the
slider to turn $b$.

```{code-cell} ipython3
:tags: [hide-input]

b_origin = np.array([1.3, 0.3])
b_q = np.array([0.6, 0.3])
frame_angles = np.arange(-180, 181, 10)


def matrix_lines(matrix):
    return "<br>".join(
        "[" + "  ".join(f"{value + 0.0:5.2f}" for value in row) + "]"
        for row in np.asarray(matrix)
    )


def frame_state(index):
    angle = math.radians(frame_angles[index])
    a_T_b = turning(angle, *b_origin)
    a_q = a_T_b @ Point3Dimensional(*b_q)
    traces = [
        *frame_axes(a_T_b, "b", width=4),
        go.Scatter(
            x=[float(a_T_b.x), float(a_q.x)],
            y=[float(a_T_b.y), float(a_q.y)],
            mode="lines",
            line=dict(color="#9467bd", width=2, dash="dot"),
            name="from b's origin to q",
            showlegend=False,
        ),
        go.Scatter(
            x=[float(a_q.x)],
            y=[float(a_q.y)],
            mode="markers",
            marker=dict(color="#9467bd", size=12),
            name=f"q: in b {numbers(*b_q)}, in a {numbers(*a_q[:2])}",
        ),
        frame_label(a_T_b, "b"),
    ]
    title = (
        f"θ = {frame_angles[index]}°<br>"
        f"<sub>a</sub>T<sub>b</sub> =<br>{matrix_lines(a_T_b)}"
    )
    return traces, {"legend.title.text": title}


initial = int(np.argmin(np.abs(frame_angles - 30)))
frame_figure = planar_figure(560, [-0.6, 2.4], [-0.5, 2.1])
frame_figure.add_traces(frame_axes(turning(0), "a", width=6))
frame_figure.add_trace(frame_label(turning(0), "a"))
frame_figure.add_traces(frame_state(initial)[0])
frame_figure.update_layout(
    legend=dict(
        title=dict(
            text=frame_state(initial)[1]["legend.title.text"],
            font=dict(family="monospace", size=12),
        )
    ),
    sliders=[
        slider(
            [str(angle) for angle in frame_angles],
            "θ (°): ",
            active=initial,
            states=[frame_state(index) for index in range(len(frame_angles))],
            traces=[3, 4, 5, 6, 7],
        )
    ],
)
show(frame_figure)
```

In code, the frames travel with the matrix. Transforming a point given in $b$ gives the
same point, given in $a$:

```{code-cell} ipython3
a_T_b = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
    x=1.3, y=0.3, yaw=math.radians(30), reference_frame=a, child_frame=b
)
b_P_q = Point3Dimensional(0.6, 0.3, 0, reference_frame=b)
a_P_q = a_T_b @ b_P_q
print("q in a:", a_P_q[:3].to_list(), "given in", a_P_q.reference_frame.name)

R, p = a_T_b.rotation, a_T_b.position
assert np.allclose(a_P_q, p + R @ Vector3Dimensional(0.6, 0.3, 0))
assert np.allclose(a_T_b.pose.position, p)
```

## Points and vectors

A vector is a direction with a length, such as the step from one point to another. It
has no place, so moving a frame must not change it, only turning one. The package writes
a vector with the fourth coordinate 0, as $(x, y, z, 0)$, so the last column of the
matrix drops out of the product: $_aV = R\;{}_bV$. Two points subtract to a vector,
because $1 - 1 = 0$, and a point plus a vector is a point again.

```{code-cell} ipython3
b_V_step = Vector3Dimensional(0.6, 0.3, 0, reference_frame=b)
a_V_step = a_T_b @ b_V_step
assert np.allclose(a_V_step, R @ Vector3Dimensional(0.6, 0.3, 0))

b_P_origin = Point3Dimensional(0, 0, 0, reference_frame=b)
step = a_P_q - a_T_b @ b_P_origin
print(type(step).__name__, step[:3].to_list())
assert np.allclose(step, a_V_step)
```

Transforming a point with $_aT_b$ leaves the point where it is and gives its coordinates
in another frame. A rotation applied to a point's coordinates in the same frame, by
contrast, moves the point somewhere else. The style guide's
[section on rotating points](https://cram2.github.io/cognitive_robot_abstract_machine/semantic_digital_twin/style_guide.html#rotating-points)
covers the difference.

## Chains of transformations

Transformations chain the way their names suggest:

$$
{}_aT_c = {}_aT_b\;{}_bT_c .
$$

Given coordinates in $c$, $_bT_c$ turns them into coordinates in $b$, and $_aT_b$ turns
those into coordinates in $a$. The inner names, $b$ and $b$, must match and drop out,
which Lynch and Park call the *subscript cancellation rule* {cite}`lynch2017modern`
(section 3.3.1). A robot's arm is such a chain: each joint contributes a factor that
changes with the joint's position, and each link one that stays fixed.

The figure shows an arm whose shoulder is turned by 40°. Its elbow joint, at the end of
the 0.8 m upper link, turns by the angle $\theta$. The tool's tip sits 0.6 m along the
lower link. Only the middle factor of

$$
{}_{base}T_{tip} = {}_{base}T_{upper}\;{}_{upper}T_{lower}\;{}_{lower}T_{tip}
$$

depends on $\theta$. The box shows each factor as the angle it turns by and the origin it
places, with its current values, and the product's: in this planar case, that is all a
matrix holds. Drag the slider to bend the elbow.

```{code-cell} ipython3
:tags: [hide-input]

shoulder = math.radians(40)
elbow_angles = np.arange(-150, 151, 10)
base_T_upper = turning(shoulder)
lower_T_tip = turning(0, x=0.6)


def placement(transformation):
    matrix = np.asarray(transformation)
    angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    return f"turns by {angle + 0.0:.0f}°, origin at {numbers(*matrix[:2, 3])}"


def arm_state(index):
    upper_T_lower = turning(math.radians(elbow_angles[index]), x=0.8)
    base_T_lower = base_T_upper @ upper_T_lower
    base_T_tip = base_T_lower @ lower_T_tip
    joints = np.array(
        [np.asarray(t)[:2, 3] for t in (base_T_upper, base_T_lower, base_T_tip)]
    )
    traces = [
        go.Scatter(
            x=joints[:, 0],
            y=joints[:, 1],
            mode="lines+markers",
            line=dict(color="#7f7f7f", width=10),
            marker=dict(size=14, color="#333333"),
            name="the arm's links",
        ),
        *frame_axes(base_T_lower, "lower", length=0.3),
        *frame_axes(base_T_tip, "tip", length=0.3, dash="dot"),
    ]
    title = (
        f"θ = {elbow_angles[index]}°<br>"
        f"<sub>base</sub>T<sub>upper</sub>: {placement(base_T_upper)}<br>"
        f"<sub>upper</sub>T<sub>lower</sub>: {placement(upper_T_lower)}<br>"
        f"<sub>lower</sub>T<sub>tip</sub>: {placement(lower_T_tip)}<br>"
        f"<sub>base</sub>T<sub>tip</sub>: {placement(base_T_tip)}"
    )
    return traces, {"legend.title.text": title}


initial = int(np.argmin(np.abs(elbow_angles - 60)))
arm_figure = planar_figure(600, [-1.7, 1.4], [-0.6, 2.0])
arm_figure.add_traces(frame_axes(turning(0), "base", width=6))
arm_figure.add_traces(arm_state(initial)[0])
arm_figure.update_layout(
    legend=dict(title=dict(text=arm_state(initial)[1]["legend.title.text"])),
    sliders=[
        slider(
            [str(angle) for angle in elbow_angles],
            "θ (°): ",
            active=initial,
            states=[arm_state(index) for index in range(len(elbow_angles))],
            traces=[2, 3, 4, 5, 6],
        )
    ],
)
show(arm_figure)
```

The product of two transformations takes its reference frame from the left factor and
its child frame from the right one:

```{code-cell} ipython3
b_T_c = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
    x=0.8, yaw=math.radians(60), reference_frame=b, child_frame=c
)
a_T_c = a_T_b @ b_T_c
print(
    "a_T_c goes from", a_T_c.child_frame.name, "to", a_T_c.reference_frame.name
)
```

It does not compare the two inner frames: a product of $_aT_b$ and a transformation
from some frame $d$ to $c$ would be labelled as going from $d$ to $a$ all the same. This
is deliberate: products of transformations are performance critical, and a comparison
would cost time in every one of them. Keeping the names of the variables right is what
keeps a chain right.

## Order matters

Matrix products do not commute: $_aT_b\,M$ and $M\,_aT_b$ differ. Take a body at the
pose $_{map}T_{body}$ and a rotation $M$ by the angle $\alpha$ about the z axis.

- **Multiplied on the right**, $_{map}T_{body}\,M$ applies $M$ in the body's
  coordinates: the body turns about its own origin, which stays where it is.
- **Multiplied on the left**, $M\,{}_{map}T_{body}$ applies $M$ in the map's
  coordinates: the whole body, origin included, swings about the map's origin.

Lynch and Park put it as rotating about an axis of the body frame or of the fixed frame
{cite}`lynch2017modern` (section 3.2.1). The figure shows both for the body at
$(1.4, 0.6)$, turned by 30°. Drag the slider to change $\alpha$.

```{code-cell} ipython3
:tags: [hide-input]

order_angles = np.arange(0, 181, 5)
map_T_body = turning(math.radians(30), x=1.4, y=0.6)
outline = np.array(
    [[-0.25, -0.15], [0.25, -0.15], [0.25, 0.15], [-0.25, 0.15], [-0.25, -0.15]]
)


def body_outline(transformation, color, name, dash=None):
    matrix = np.asarray(transformation)
    corners = outline @ matrix[:2, :2].T + matrix[:2, 3]
    return go.Scatter(
        x=corners[:, 0],
        y=corners[:, 1],
        mode="lines",
        fill="toself",
        line=dict(color=color, width=2, dash=dash),
        name=name,
    )


def order_state(index):
    motion = turning(math.radians(order_angles[index]))
    in_body = map_T_body @ motion
    in_map = motion @ map_T_body
    swing = np.radians(np.linspace(0, order_angles[index], 60))
    radius_angle = math.atan2(0.6, 1.4)
    radius = math.hypot(1.4, 0.6)
    traces = [
        go.Scatter(
            x=radius * np.cos(radius_angle + swing),
            y=radius * np.sin(radius_angle + swing),
            mode="lines",
            line=dict(color="#ff7f0e", width=2, dash="dot"),
            name="path of the swung body's origin",
        ),
        body_outline(
            in_body,
            "#1f77b4",
            f"<sub>map</sub>T<sub>body</sub> M: origin at {numbers(*in_body[:2, 3])}",
        ),
        body_outline(
            in_map,
            "#ff7f0e",
            f"M <sub>map</sub>T<sub>body</sub>: origin at {numbers(*in_map[:2, 3])}",
        ),
    ]
    return traces, {"legend.title.text": f"α = {order_angles[index]}°"}


initial = int(np.argmin(np.abs(order_angles - 90)))
order_figure = planar_figure(600, [-2.0, 2.2], [-1.2, 2.0])
order_figure.add_traces(frame_axes(turning(0), "map", width=6))
order_figure.add_trace(
    body_outline(map_T_body, "#7f7f7f", "the body before", dash="dash")
)
order_figure.add_traces(order_state(initial)[0])
order_figure.update_layout(
    legend=dict(
        x=0.99,
        xanchor="right",
        y=0.01,
        yanchor="bottom",
        title=dict(text=order_state(initial)[1]["legend.title.text"]),
    ),
    sliders=[
        slider(
            [str(angle) for angle in order_angles],
            "α (°): ",
            active=initial,
            states=[order_state(index) for index in range(len(order_angles))],
            traces=[3, 4, 5],
        )
    ],
)
show(order_figure)
```

## Inverting a transformation

$_bT_a$, the transformation back from $a$ to $b$, is the inverse of $_aT_b$. Solving
$_aq = R\,{}_bq + p$ for $_bq$ gives $_bq = R^{-1}({}_aq - p)$. The columns of a
rotation matrix are perpendicular and of unit length, so $R^{-1} = R^\top$, the
transposed matrix, and

$$
{}_bT_a = {}_aT_b^{-1} =
\begin{bmatrix} R^\top & -R^\top p \\ 0\;0\;0 & 1 \end{bmatrix}
$$

{cite}`lynch2017modern` (section 3.3.1, proposition 3.15). The translation of the
inverse is not $-p$: it is $b$'s view of where $a$'s origin is, given along $b$'s axes.
The transpose of the whole $4 \times 4$ matrix is no transformation at all, since it
moves $p$ into the bottom row.

```{code-cell} ipython3
b_T_a = a_T_b.inverse()
print(
    "b_T_a goes from", b_T_a.child_frame.name, "to", b_T_a.reference_frame.name
)
assert np.allclose(b_T_a @ a_T_b, np.eye(4))
assert np.allclose(b_T_a.position.vector3, -(R.T @ p.vector3))

# for a plain 4x4 array of numbers, without building an expression:
numeric = HomogeneousTransformationMatrix3Dimensional.invert_numeric_matrix(
    np.asarray(a_T_b)
)
assert np.allclose(numeric, b_T_a)
```

## Rotations

The package writes a rotation in four forms. All of them convert into each other.

| Form | Numbers | Constraint | Useful for |
|---|---|---|---|
| `RotationMatrix` | 9, as the top left of a $4 \times 4$ matrix | columns perpendicular and of unit length | chaining with transformations and turning points and vectors |
| `Quaternion` | 4: $(x, y, z, w)$ | unit length | storing and interpolating rotations, with no singular angles |
| `AxisAngle` | 4: a unit axis and an angle | unit axis | reading what a rotation does |
| roll, pitch, yaw | 3 angles | none | entering a rotation by hand |

`Displacement.rotation_vector` is a fifth, related form: the axis scaled by the angle,
three numbers (see [Rates of motion](rates_of_motion.md)).

### Roll, pitch and yaw

`RotationMatrix.from_rpy(roll, pitch, yaw)` rotates first by `roll` about the x axis,
then by `pitch` about the y axis, then by `yaw` about the z axis, all three axes of the
fixed reference frame:

$$
R = R_z(\text{yaw})\,R_y(\text{pitch})\,R_x(\text{roll}),
$$

where $R_x(\gamma)$ is the rotation by the angle $\gamma$ about the x axis, and so on.
Read from left to right, the same product turns about the body's z axis, then about its
turned y axis, then about its twice-turned x axis: the ZYX Euler angles
{cite}`lynch2017modern` (appendix B.2).

```{code-cell} ipython3
roll, pitch, yaw = 0.3, 0.4, 0.5
matrix = RotationMatrix.from_rpy(roll, pitch, yaw)
assert np.allclose(
    matrix[:3, :3], Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
)
assert np.allclose([float(angle) for angle in matrix.rpy], [roll, pitch, yaw])
```

At a pitch of $\pm 90°$, three angles no longer describe a rotation uniquely. Turning
by $\pi/2$ about y carries the x axis onto the z axis, up to its sign: at a pitch of
$+90°$, $R_y(\pi/2)\,R_x(\gamma) = R_z(-\gamma)\,R_y(\pi/2)$, so

$$
R = R_z(\text{yaw} - \text{roll})\,R_y(\pi/2).
$$

Roll and yaw now turn about the same axis, and only their difference matters. Changing
both by the same amount does not rotate the body at all; this is called *gimbal lock*.
Reading the angles back from such a matrix can find only the difference. The code
returns a yaw of 0 and puts the difference, roll − yaw, into the roll, as Lynch and
Park's algorithm does {cite}`lynch2017modern` (appendix B.1.1).

The figure shows how this happens. Read as the ZYX Euler angles, each angle turns the
body about its own axis: yaw about the z axis of the reference frame, pitch about the y
axis once the yaw has turned it, and roll about the body's own x axis. The figure draws
these three axes through the body, and the three sliders set the angles. Each circle
shows where the marked corner of the body goes when one angle changes and the others
stay: changing an angle turns the body about that angle's axis. The box at the top left
shows the angle between the roll and yaw axes, and the angles the code reads back from
the matrix.

Move the pitch towards 90°. The pitch turns the roll axis onto the line of the yaw axis,
and the roll and yaw circles merge into one. At 90°, move the roll and the yaw: both
move the corner along the same circle, and raising both by the same amount leaves the
body where it is. Only two ways to turn the body are left, and the read-back angles
show only roll − yaw. Drag the figure to look at it from another side.

```{code-cell} ipython3
:tags: [hide-input]

start_angles = dict(roll=20, pitch=45, yaw=30)
circle_changes = np.arange(-180, 181, 5)
corner = [0.45, 0.6, 0.05]
box_corners = [
    [x, y, z] for x in (-0.45, 0.45) for y in (-0.6, 0.6) for z in (-0.05, 0.05)
]
box_faces = dict(
    i=[0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1],
    j=[1, 3, 5, 7, 1, 5, 3, 7, 2, 6, 3, 7],
    k=[3, 2, 7, 6, 5, 4, 7, 6, 6, 4, 7, 5],
)
angle_colors = dict(roll="#d62728", pitch="#2ca02c", yaw="#1f77b4")


def oriented(angles):
    return RotationMatrix.from_rpy(
        *(math.radians(angles[name]) for name in ("roll", "pitch", "yaw"))
    )


def columns(points):
    return [[float(point[axis]) for point in points] for axis in range(3)]


def line_3d(direction, color, name, width=6):
    x, y, z = columns([[-value for value in direction], direction])
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
    )


def circle(angles, name):
    points = [
        np.asarray(
            oriented(dict(angles, **{name: angles[name] + change}))
            @ Vector3Dimensional(*corner)
        )[:3]
        for change in circle_changes
    ]
    x, y, z = columns(points)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(
            color=angle_colors[name],
            width=9 if name == "yaw" else 5,
            dash="dash" if name == "roll" else None,
        ),
        name=f"where the corner goes when the {name} changes",
    )


def lock_title(angles):
    rotation = np.asarray(oriented(angles))[:3, :3]
    between = math.degrees(math.acos(min(1.0, abs(rotation[2, 0]))))
    read_back = [
        round(math.degrees(float(angle))) for angle in oriented(angles).rpy
    ]
    return (
        f"angle between roll and yaw axes: {round(between)}°<br>"
        f"read back: roll {read_back[0]}°, pitch {read_back[1]}°,"
        f" yaw {read_back[2]}°"
    )


def lock_traces(angles):
    matrix = oriented(angles)
    rotation = np.asarray(matrix)[:3, :3]
    pitch_axis = np.asarray(
        oriented(dict(roll=0, pitch=0, yaw=angles["yaw"])).to_coordinate_change()
        @ Vector3Dimensional(0, 1, 0)
    )[:3]
    x, y, z = columns([rotation @ point for point in box_corners])
    marked = columns([rotation @ corner])
    return [
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            **box_faces,
            color="#9e9e9e",
            opacity=0.45,
            flatshading=True,
            name="the body",
            showlegend=True,
        ),
        line_3d(
            rotation[:, 0], angle_colors["roll"], "roll axis: the body's x axis"
        ),
        line_3d(pitch_axis, angle_colors["pitch"], "pitch axis"),
        circle(angles, "yaw"),
        circle(angles, "roll"),
        circle(angles, "pitch"),
        go.Scatter3d(
            x=marked[0],
            y=marked[1],
            z=marked[2],
            mode="markers",
            marker=dict(color="#333333", size=6),
            name="the marked corner",
        ),
    ]


def angle_slider(name, values, y):
    return dict(
        name=name,
        active=int(np.argmin(np.abs(values - start_angles[name]))),
        currentvalue=dict(
            prefix=f"{name} (°): ", font=dict(size=13, color=angle_colors[name])
        ),
        font=dict(color="rgba(0, 0, 0, 0)"),
        ticklen=3,
        minorticklen=0,
        len=0.9,
        x=0.05,
        y=y,
        yanchor="top",
        pad=dict(t=0, b=0),
        steps=[
            dict(label=str(value), value=str(value), method="skip", args=[])
            for value in values
        ],
    )


lock_figure = go.Figure(
    data=[
        line_3d(
            [0, 0, 1],
            angle_colors["yaw"],
            "yaw axis: the reference z axis",
            width=14,
        ),
        *lock_traces(start_angles),
    ]
)
lock_figure.update_layout(
    template="plotly_white",
    height=820,
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(
        **legend_box,
        font=dict(size=11),
        title=dict(text=lock_title(start_angles)),
    ),
    scene=dict(
        domain=dict(y=[0.27, 1.0]),
        xaxis=dict(range=[-1.0, 1.0], title="x"),
        yaxis=dict(range=[-1.0, 1.0], title="y"),
        zaxis=dict(range=[-1.0, 1.0], title="z"),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.2, y=-1.7, z=0.7)),
    ),
    uirevision="lock",
    sliders=[
        angle_slider("roll", np.arange(-180, 181, 5), 0.24),
        angle_slider("pitch", np.arange(-90, 91, 5), 0.155),
        angle_slider("yaw", np.arange(-180, 181, 5), 0.07),
    ],
)

# The browser redraws the figure when a slider moves, with the same computation as
# lock_traces and lock_title above.
lock_script = """
const plot = document.getElementById('{plot_id}');
const settings = SETTINGS;
const angles = Object.assign({}, settings.start);
const radians = Math.PI / 180;
function about(axis, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  if (axis === 'x') return [[1, 0, 0], [0, c, -s], [0, s, c]];
  if (axis === 'y') return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
  return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
}
function times(a, b) {
  return a.map(row => [0, 1, 2].map(
    j => row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j]));
}
function apply(m, v) {
  return m.map(row => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
}
function oriented(a) {
  return times(about('z', a.yaw * radians),
    times(about('y', a.pitch * radians), about('x', a.roll * radians)));
}
function readBack(m) {
  const cy = Math.hypot(m[0][0], m[1][0]);
  if (cy - settings.eps > 0) {
    return [Math.atan2(m[2][1], m[2][2]), Math.atan2(-m[2][0], cy),
      Math.atan2(m[1][0], m[0][0])];
  }
  return [Math.atan2(-m[1][2], m[1][1]), Math.atan2(-m[2][0], cy), 0];
}
function columns(points) {
  return [0, 1, 2].map(axis => points.map(point => point[axis]));
}
function circle(name) {
  return columns(settings.circle.map(change => apply(
    oriented(Object.assign({}, angles, {[name]: angles[name] + change})),
    settings.corner)));
}
function line(direction) {
  return columns([direction.map(value => -value), direction]);
}
function degrees(angle) {
  return Math.round(angle / radians) + 0;
}
function title(m) {
  const between = Math.acos(Math.min(1, Math.abs(m[2][0])));
  const back = readBack(m).map(degrees);
  return 'angle between roll and yaw axes: ' + degrees(between) + '°<br>'
    + 'read back: roll ' + back[0] + '°, pitch ' + back[1] + '°, yaw ' + back[2] + '°';
}
function redraw() {
  const m = oriented(angles);
  const traces = [
    columns(settings.box.map(point => apply(m, point))),
    line([m[0][0], m[1][0], m[2][0]]),
    line(apply(about('z', angles.yaw * radians), [0, 1, 0])),
    circle('yaw'), circle('roll'), circle('pitch'),
    columns([apply(m, settings.corner)]),
  ];
  Plotly.update(plot,
    {x: traces.map(t => t[0]), y: traces.map(t => t[1]), z: traces.map(t => t[2])},
    {'legend.title.text': title(m)}, [1, 2, 3, 4, 5, 6, 7]);
}
plot.on('plotly_sliderchange', event => {
  angles[event.slider.name] = Number(event.step.value);
  redraw();
});
""".replace(
    "SETTINGS",
    json.dumps(
        dict(
            start=start_angles,
            eps=sm.EPS,
            circle=circle_changes.tolist(),
            corner=corner,
            box=box_corners,
        )
    ),
)
show(lock_figure, lock_script)
```

In code, at a pitch of exactly 90°:

```{code-cell} ipython3
locked = RotationMatrix.from_rpy(0.3, math.pi / 2, 0.5)
print("read back:", [round(float(angle), 3) for angle in locked.rpy])
assert np.allclose(
    locked, RotationMatrix.from_rpy(0.3 + 1, math.pi / 2, 0.5 + 1)
)
```

Quaternions have no such angle. That is one reason to store rotations as quaternions,
and to use roll, pitch and yaw only for entering them.

### Two quaternions for every rotation

The rotation by the angle $\theta$ about the unit axis $u$ is the quaternion

$$
q = \left(u\,\sin\tfrac{\theta}{2},\; \cos\tfrac{\theta}{2}\right),
$$

with the axis part first, as $(x, y, z)$, and the angle part $w$ last, the order the
package's constructors take them in. Turning by $\theta + 2\pi$ is the same rotation,
but it gives $-q$, so every rotation has two quaternions, $q$ and $-q$
{cite}`lynch2017modern` (appendix B.3). Comparing quaternions number by number can
therefore call two equal rotations different. `rotational_distance` compares the
rotations instead:

```{code-cell} ipython3
q = Quaternion.from_rpy(0.3, 0.4, 0.5)
print("q: ", [round(value, 3) for value in q.to_list()])
print("-q:", [round(value, 3) for value in (-q).to_list()])
assert np.allclose(q.to_coordinate_change(), (-q).to_coordinate_change())
assert abs(float(q.rotational_distance(-q))) < 1e-6
```

## From a displacement to a transformation

A `Displacement`, a rigid-body displacement as described in
[Rates of motion](rates_of_motion.md), holds a rotation and a translation: how far a body
turned, as the rotation vector $r$, and where its origin ended up, as the translation
$t$. A body that started at its reference frame's origin, with the axes of
that frame, therefore ends at the pose

$$
\begin{bmatrix} R(r) & t \\ 0\;0\;0 & 1 \end{bmatrix},
$$

where $R(r)$ is the rotation by the angle $|r|$ about the axis $r / |r|$. This is the
matrix exponential of the motion's rate {cite}`lynch2017modern` (section 3.3.3), which
the cell compares against for the screw axis of the first figure on that page.

```{code-cell} ipython3
screw = ScrewAxis(
    rotation_per_unit_position=Vector3Dimensional(0, 0, 1),
    translation_per_unit_position=Vector3Dimensional(0, -0.3, 0.05),
)
position = 2.0
displacement = screw.to_displacement(position=position)
rotation_vector = displacement.rotation_vector
end_pose = HomogeneousTransformationMatrix3Dimensional.from_xyz_axis_angle(
    *displacement.translation[:3],
    axis=rotation_vector / rotation_vector.norm(),
    angle=rotation_vector.norm(),
)

rate = np.zeros((4, 4))
rate[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
rate[:3, 3] = [0, -0.3, 0.05]
assert np.allclose(end_pose, expm(rate * position))
print("origin after θ = 2:", end_pose[:3, 3].to_list())
```
