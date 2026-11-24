---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Screw axis connections

A [revolute connection](https://en.wikipedia.org/wiki/Revolute_joint) rotates its child about an axis,
like a door on its hinges. A [prismatic connection](https://en.wikipedia.org/wiki/Prismatic_joint) moves
its child along an axis, like a drawer. In the semantic digital twin, both are a `ScrewAxisConnection`:
the connection describes how its child moves with one screw axis, how the child's pose changes per unit
of the connection's position. What a screw axis holds, and how it relates to a twist, is explained in
[Displacements, twists and screw axes](https://cram2.github.io/cognitive_robot_abstract_machine/krrood_spatial_math/rates_of_motion.html#displacements-twists-and-screw-axes)
in the krrood_spatial_math book. The idea of describing every such connection with one screw axis was
inspired by MoMa-SG {cite:p}`buechner2026articulated`.

The screw axis is normalized as {cite:t}`lynch2017modern` define it in
[Definition 3.24 of Modern Robotics](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=122).
If the child rotates, the rotation per unit of position of the screw axis is a unit vector, so the
position of a revolute connection is an angle in radians. If the child only translates, the translation
per unit of position is a unit vector, so the position of a prismatic connection is a distance in meters.

The screw axis is given in the *connection frame*. `parent_T_connection_expression` places the
connection frame relative to the parent, and `connection_T_child_expression` places the child relative
to the connection frame, so a child set off from the axis swings around it. Joints read from robot and
scene descriptions put the axis through the connection frame's origin, as the examples below do; the
axis may also lie elsewhere, as {ref}`Where the axis lies <where-the-axis-lies>` explains.

## Revolute and prismatic connections

Two factories build the screw axes of revolute and prismatic connections:

| Connection | Factory | Position |
|---|---|---|
| Revolute | `ScrewAxisConnection.create_revolute_screw_axis(axis=axis)` | radians about the axis |
| Prismatic | `ScrewAxisConnection.create_prismatic_screw_axis(axis=axis)` | meters along the axis |

In the figure below, choose a connection with the buttons and move it with the slider. The dashed black
line is the screw axis, and the grey line is the path the origin of the child follows over the range of
the slider. The thin axes show the root frame, the dashed axes the connection frame, and the thick axes
the child frame. The forward kinematics of the semantic digital twin computes every pose in the figure.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML

from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.world_entity import Body

pitch = 0.1

screw_axes = {
    "Revolute": ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
    "Prismatic": ScrewAxisConnection.create_prismatic_screw_axis(axis=Vector3.Z()),
    "Right-handed helical": ScrewAxisConnection.create_helical_screw_axis(
        axis=Vector3.Z(), pitch=pitch
    ),
    "Left-handed helical": ScrewAxisConnection.create_helical_screw_axis(
        axis=Vector3.Z(), pitch=-pitch
    ),
}
travel = {
    "Revolute": (1.5 * np.pi, "radians"),
    "Prismatic": (0.3, "meters"),
    "Right-handed helical": (4 * np.pi, "radians"),
    "Left-handed helical": (4 * np.pi, "radians"),
}

world = World()
root = Body(name=PrefixedName("root"))
connections = {}
with world.modify_world():
    for kind, screw_axis in screw_axes.items():
        connection = ScrewAxisConnection.create_with_dofs(
            world=world,
            parent=root,
            child=Body(name=PrefixedName(kind)),
            screw_axis=screw_axis,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.2, y=0.1, z=0.15, roll=0.5, pitch=0.3
            ),
            connection_T_child_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.15
            ),
        )
        world.add_connection(connection)
        connections[kind] = connection

root_T_connection = connections["Revolute"].parent_T_connection_expression.to_np()
axis_origin, axis_direction = root_T_connection[:3, 3], root_T_connection[:3, 2]
frame_colors = ["#d62728", "#2ca02c", "#1f77b4"]
box_corners = np.array(
    [[x, y, z] for z in (-1, 1) for x, y in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
) * [0.05, 0.035, 0.02]
box_triangles = dict(
    i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
    j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
    k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
)


def root_T_child(kind: str, position: float) -> np.ndarray:
    connection = connections[kind]
    connection.position = position
    return world.compute_forward_kinematics_np(root=root, tip=connection.child)


def frame_points(transform: np.ndarray, length: float) -> dict:
    """
    :return: The three axes of a frame as one line broken between the axes.
    """
    origin = transform[:3, 3]
    points = []
    for axis in range(3):
        points += [origin, origin + length * transform[:3, axis], [None] * 3]
    x, y, z = zip(*points)
    return dict(x=x, y=y, z=z)


def frame_trace(transform: np.ndarray, length: float, width: int, dash: str) -> go.Scatter3d:
    return go.Scatter3d(
        **frame_points(transform, length), mode="lines", hoverinfo="skip",
        showlegend=False,
        line=dict(
            color=[0, 0, 0, 1, 1, 1, 2, 2, 2], cmin=0, cmax=2, width=width, dash=dash,
            colorscale=[[0, frame_colors[0]], [0.5, frame_colors[1]], [1, frame_colors[2]]],
        ),
    )


def child_pose(kind: str, position: float) -> tuple[dict, dict]:
    """
    :return: The corners of the child's box and its frame's axes at that position.
    """
    transform = root_T_child(kind, position)
    corners = box_corners @ transform[:3, :3].T + transform[:3, 3]
    box = dict(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2])
    return box, frame_points(transform, 0.08)


def connection_figure(kinds: list[str]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(frame_trace(np.eye(4), 0.1, 3, "solid"))
    figure.add_trace(frame_trace(root_T_connection, 0.1, 4, "dash"))
    groups, sliders = {}, {}
    for kind in kinds:
        visible = kind == kinds[0]
        group_start = len(figure.data)
        axis_line = np.array(
            [axis_origin - 0.3 * axis_direction, axis_origin + 0.3 * axis_direction]
        )
        figure.add_trace(go.Scatter3d(
            x=axis_line[:, 0], y=axis_line[:, 1], z=axis_line[:, 2], mode="lines",
            line=dict(color="black", width=3, dash="dash"), name="screw axis",
            hoverinfo="skip", visible=visible,
        ))
        path = np.array([
            root_T_child(kind, fraction * travel[kind][0])[:3, 3]
            for fraction in np.linspace(0, 1, 200)
        ])
        figure.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines",
            line=dict(color="grey", width=3), name="path of the child",
            hoverinfo="skip", visible=visible,
        ))
        box, axes = child_pose(kind, 0)
        figure.add_trace(go.Mesh3d(
            **box, **box_triangles, color="#ff7f0e", opacity=0.6, flatshading=True,
            hoverinfo="skip", name="child", visible=visible,
        ))
        figure.add_trace(frame_trace(np.eye(4), 0.08, 8, "solid").update(**axes))
        figure.data[-1].visible = visible
        groups[kind] = range(group_start, len(figure.data))

        full_travel, unit = travel[kind]
        steps = []
        for position in np.linspace(0, full_travel, 41):
            box, axes = child_pose(kind, position)
            steps.append(dict(
                method="restyle",
                label=f"{position:.2f}",
                args=[
                    {key: [box[key], axes[key]] for key in ("x", "y", "z")},
                    [groups[kind][-2], groups[kind][-1]],
                ],
            ))
        sliders[kind] = dict(
            steps=steps, active=0, pad=dict(t=10),
            currentvalue=dict(prefix=f"Position in {unit}: "),
        )

    trace_count = len(figure.data)
    initial_data = {
        key: [figure.data[index][key] for index in range(trace_count)]
        for key in ("x", "y", "z")
    }
    buttons = [
        dict(
            label=kind,
            method="update",
            args=[
                initial_data
                | {"visible": [index < 2 or index in group for index in range(trace_count)]},
                {"sliders": [sliders[kind]]},
            ],
        )
        for kind, group in groups.items()
    ]

    every_point = np.array([
        point
        for trace in figure.data
        for point in zip(trace.x, trace.y, trace.z)
        if None not in point
    ], dtype=float)
    low, high = every_point.min(axis=0) - 0.05, every_point.max(axis=0) + 0.05
    half_size = (high - low).max() / 2
    centre = (low + high) / 2
    axis_titles = {"x": "x in meters", "y": "y in meters", "z": "z in meters"}
    figure.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        uirevision="keep",
        legend=dict(x=0.01, y=0.99),
        scene=dict(
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=-1.4, z=0.9)),
            **{
                f"{name}axis": dict(
                    range=[centre[index] - half_size, centre[index] + half_size],
                    title=title,
                )
                for index, (name, title) in enumerate(axis_titles.items())
            },
        ),
        updatemenus=[dict(type="buttons", direction="right", x=0.5, xanchor="center",
                          y=1.06, buttons=buttons, showactive=True)],
        sliders=[sliders[kinds[0]]],
    )
    return figure


def embedded(figure: go.Figure) -> HTML:
    return HTML(
        figure.to_html(full_html=False, include_plotlyjs="cdn", include_mathjax=False)
    )


embedded(connection_figure(["Revolute", "Prismatic"]))
```

A revolute connection keeps its child at a constant distance from the axis, and a prismatic connection
moves it without rotating it.

## Building a connection

A connection is built from its screw axis. This one rotates a door about the vertical axis of its
cabinet:

```{code-cell} ipython3
cabinet_world = World()
cabinet = Body(name=PrefixedName("cabinet"))
door = Body(name=PrefixedName("door"))
with cabinet_world.modify_world():
    door_connection = ScrewAxisConnection.create_with_dofs(
        world=cabinet_world,
        parent=cabinet,
        child=door,
        screw_axis=ScrewAxisConnection.create_revolute_screw_axis(axis=Vector3.Z()),
    )
    cabinet_world.add_connection(door_connection)

door_connection.position = np.pi / 2
print(door_connection.behaves_like_a_revolute_connection)
```

Code that has to tell revolute and prismatic connections apart asks
`behaves_like_a_revolute_connection` and `behaves_like_a_prismatic_connection`, and
`behaves_like_a_helical_connection` for the helical connections of the next section. They read the kind
off the screw axis with a tolerance in physical units: a child that advances along the axis by less than
`ScrewAxisConnection.smallest_advance_per_radian`, a micrometer per radian, does not advance, so that the
rounding errors of a screw axis computed from measurements do not make a door helical.

## Beyond revolute and prismatic connections

Because the screw axis holds both a rotation and a translation per unit of position, the child can also
rotate and translate at once: a bottle cap, a jar
lid or the spindle of a vice turns and advances along its axis in proportion. Such a *helical* motion is
described by its pitch $p$, the distance the child advances per revolution.

| Connection | Factory | Position |
|---|---|---|
| Helical | `ScrewAxisConnection.create_helical_screw_axis(axis=axis, pitch=pitch)` | radians about the axis |

A positive pitch gives a right-handed thread, and a negative pitch a left-handed thread:

```{code-cell} ipython3
:tags: [hide-input]

embedded(connection_figure(["Right-handed helical", "Left-handed helical"]))
```

```{note}
The `pitch` of a helical connection is the distance it advances per *revolution*, as the
[pitch of a single-start thread](https://en.wikipedia.org/wiki/Screw_thread#Lead,_pitch,_and_starts).
Lynch and Park define the pitch $h$ of a screw axis per *radian* instead, so $h = p / 2\pi$. The
semantic digital twin keeps the definition its earlier `ScrewConnection` used, so that the pitch of a
thread can be taken over unchanged from its specification, where it is given per revolution.
```

The position of a helical connection stays an angle. Where a distance along the axis is the natural
goal, such as how far to unscrew a cap, `position_for_translation` gives the position that advances the
child that far:

```{code-cell} ipython3
bottle_world = World()
bottle = Body(name=PrefixedName("bottle"))
cap = Body(name=PrefixedName("cap"))
with bottle_world.modify_world():
    cap_connection = ScrewAxisConnection.create_with_dofs(
        world=bottle_world,
        parent=bottle,
        child=cap,
        screw_axis=ScrewAxisConnection.create_helical_screw_axis(
            axis=Vector3.Z(), pitch=0.004
        ),
    )
    bottle_world.add_connection(cap_connection)

cap_connection.position = cap_connection.position_for_translation(distance=0.002)
print(cap_connection.position)
```

Half a pitch is half a revolution, so the cap turned by $\pi$ radians. The same method gives a prismatic
connection's position unchanged, and a revolute connection, which never translates its child, refuses it
with a `ScrewAxisWithoutTranslationError`.

## Why a rotation and a translation per unit of position

A screw axis is often described by three quantities, as in
[section 3.3.2.2 of Modern Robotics](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=121):

- a point $q$ on the axis, which places the axis in space; any point on the axis serves,
- the unit vector $\hat{s}$ along the axis, whose sense gives the direction of a positive rotation by the
  right-hand rule,
- the pitch $h$, how far the child advances along $\hat{s}$ per radian it rotates about it.

In the figure below, $q$ is the origin of the connection frame, and the grey line is the path the origin
of the child follows over one revolution. Move the slider to change $h$. At $h = 0$ the path is a
circle, as for a revolute connection. The larger $h$, the farther one revolution advances the child,
by $2\pi h$, shown as the orange line along the axis.

```{code-cell} ipython3
:tags: [hide-input]

pitches_per_radian = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
with world.modify_world():
    pitch_connections = []
    for index, pitch_per_radian in enumerate(pitches_per_radian):
        pitch_connection = ScrewAxisConnection.create_with_dofs(
            world=world,
            parent=root,
            child=Body(name=PrefixedName(f"pitch {index}")),
            screw_axis=ScrewAxisConnection.create_helical_screw_axis(
                axis=Vector3.Z(), pitch=2 * np.pi * pitch_per_radian
            ),
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.2, y=0.1, z=0.15, roll=0.5, pitch=0.3
            ),
            connection_T_child_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.15
            ),
        )
        world.add_connection(pitch_connection)
        pitch_connections.append(pitch_connection)


def one_revolution(connection: ScrewAxisConnection) -> tuple[dict, dict]:
    """
    :return: The path of the child's origin over one revolution, and the line along the
        axis that spans how far that revolution advances it.
    """
    points = []
    for angle in np.linspace(0, 2 * np.pi, 120):
        connection.position = angle
        points.append(
            world.compute_forward_kinematics_np(root=root, tip=connection.child)[:3, 3]
        )
    points = np.array(points)
    advance = (points[-1] - points[0]).dot(axis_direction)
    advance_line = np.array([axis_origin, axis_origin + advance * axis_direction])
    return (
        dict(x=points[:, 0], y=points[:, 1], z=points[:, 2]),
        dict(x=advance_line[:, 0], y=advance_line[:, 1], z=advance_line[:, 2]),
    )


pitch_figure = go.Figure()
pitch_figure.add_trace(frame_trace(np.eye(4), 0.1, 3, "solid"))
axis_line = np.array([axis_origin - 0.25 * axis_direction, axis_origin + 0.45 * axis_direction])
pitch_figure.add_trace(go.Scatter3d(
    x=axis_line[:, 0], y=axis_line[:, 1], z=axis_line[:, 2], mode="lines",
    line=dict(color="black", width=3, dash="dash"), name="screw axis", hoverinfo="skip",
))
pitch_figure.add_trace(go.Scatter3d(
    x=[0, axis_origin[0]], y=[0, axis_origin[1]], z=[0, axis_origin[2]],
    mode="lines+text", text=["", "q"], textposition="middle left", textfont=dict(size=16),
    line=dict(color="#9467bd", width=4), name="q, a point on the axis", hoverinfo="skip",
))
direction_tip = axis_origin + 0.12 * axis_direction
pitch_figure.add_trace(go.Cone(
    x=[direction_tip[0]], y=[direction_tip[1]], z=[direction_tip[2]],
    u=[axis_direction[0]], v=[axis_direction[1]], w=[axis_direction[2]],
    anchor="tip", sizemode="absolute", sizeref=0.04, showscale=False,
    colorscale=[[0, "black"], [1, "black"]], hoverinfo="skip", name="ŝ",
))
pitch_figure.add_trace(go.Scatter3d(
    x=[direction_tip[0]], y=[direction_tip[1]], z=[direction_tip[2]], mode="text",
    text=["ŝ"], textposition="middle right", textfont=dict(size=16), showlegend=False,
    hoverinfo="skip",
))
path, advance = one_revolution(pitch_connections[0])
pitch_figure.add_trace(go.Scatter3d(
    **path, mode="lines", line=dict(color="grey", width=4),
    name="path over one revolution", hoverinfo="skip",
))
pitch_figure.add_trace(go.Scatter3d(
    **advance, mode="lines", line=dict(color="#ff7f0e", width=10),
    name="advance per revolution, 2πh", hoverinfo="skip",
))
moving = [len(pitch_figure.data) - 2, len(pitch_figure.data) - 1]
steps = []
for pitch_per_radian, connection in zip(pitches_per_radian, pitch_connections):
    path, advance = one_revolution(connection)
    steps.append(dict(
        method="restyle",
        label=f"{pitch_per_radian:.3f}",
        args=[{key: [path[key], advance[key]] for key in ("x", "y", "z")}, moving],
    ))
pitch_figure.update_layout(
    height=560,
    margin=dict(l=0, r=0, t=10, b=0),
    uirevision="keep",
    legend=dict(x=0.01, y=0.99),
    scene=dict(
        aspectmode="cube",
        camera=dict(eye=dict(x=1.4, y=-1.4, z=0.7)),
        xaxis=dict(range=[-0.15, 0.55], title="x in meters"),
        yaxis=dict(range=[-0.35, 0.35], title="y in meters"),
        zaxis=dict(range=[-0.1, 0.6], title="z in meters"),
    ),
    sliders=[dict(
        steps=steps, active=0, pad=dict(t=10),
        currentvalue=dict(prefix="Pitch h in meters per radian: "),
    )],
)
embedded(pitch_figure)
```

That description works for revolute connections, whose pitch is zero, and for helical ones, but not for
prismatic connections: a pure translation rotates by no angle at all, so its pitch would be infinite, and
as nothing turns about its axis, every point in space could serve as $q$. A connection built on it would
have to treat translation as a special case.

The screw axis $\mathcal{S} = (\omega, v)$ of
[Definition 3.24](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=122), a rotation $\omega$
and a translation $v$ per unit of position, has no such gap. It
gives every motion of the family with finite numbers:

| Connection | Rotation per unit of position $\omega$ | Translation per unit of position $v$ |
|---|---|---|
| Revolute | unit vector along the axis | zero |
| Prismatic | zero | unit vector along the axis |
| Helical | unit vector along the axis | $\omega \cdot p / 2\pi$ |

### Why the screw axis is normalized

Following $k\,\mathcal{S}$ for $\theta / k$ moves the child as following $\mathcal{S}$ for the position
$\theta$ does, for any number $k \neq 0$. The normalization fixes $k$:

- The position is in radians when the child rotates and in meters when it only translates. Limits of
  degrees of freedom, the joints of URDF, MJCF and SDF files, and MuJoCo's hinge and slide joints all
  assume these units.
- Connections that move their child alike have equal screw axes.
- Observed poses determine only $\theta\,\mathcal{S}$, so an estimator does not have to find the scale.

The sign stays open: $-\mathcal{S}$ at $-\theta$ moves the child like $\mathcal{S}$ at $\theta$. To move
the child faster or slower per unit of position, keep the screw axis normalized and set the connection's
`multiplier`. `ScrewAxis.is_normalized` checks the normalization, and a `ScrewAxisConnection` raises an
`UnnormalizedScrewAxisError` for a screw axis that fails it.

```{code-cell} ipython3
:tags: [remove-cell]

from krrood_spatial_math.spatial_types import ScrewAxis

normalized = ScrewAxisConnection.create_helical_screw_axis(axis=Vector3.Z(), pitch=0.004)
scaled = ScrewAxis(
    rotation_per_unit_position=normalized.rotation_per_unit_position * 2.5,
    translation_per_unit_position=normalized.translation_per_unit_position * 2.5,
)
flipped = ScrewAxis(
    rotation_per_unit_position=-normalized.rotation_per_unit_position,
    translation_per_unit_position=-normalized.translation_per_unit_position,
)
assert normalized.is_normalized and not scaled.is_normalized and flipped.is_normalized
for position in [-1.2, 0.0, 0.7]:
    expected = normalized.to_displacement(position=position).to_np()
    assert np.allclose(scaled.to_displacement(position=position / 2.5).to_np(), expected)
    assert np.allclose(flipped.to_displacement(position=-position).to_np(), expected)
```

Following the screw axis for a position $\theta$ moves the child by the displacement that the screw axis
produces over that change of position.
[From a rate to a displacement](https://cram2.github.io/cognitive_robot_abstract_machine/krrood_spatial_math/rates_of_motion.html#from-a-rate-to-a-displacement)
in the krrood_spatial_math book derives it: the child turns by the rotation vector $r = \theta\,\omega$,
and its origin moves by $t = \theta\,v$ plus two correction terms built from $r \times t$, which carry it
along the arc it follows while it turns.

## The transformation matrix

The world model of the semantic digital twin is the direct successor of the world model of Giskard,
described in chapter 4 of {cite:t}`stelter25giskard`, which builds on
{cite:t}`roefer2022kineverse`. In it, each connection gives the
[homogeneous transformation matrix](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=107) of its child relative to its parent as a symbolic
expression of the degrees of freedom. Stelter shows a revolute joint as a rotation about the $z$-axis and
a prismatic joint as a translation along a fixed axis, in
[figures 4.2 and 4.3 of his thesis](https://doi.org/10.26092/elib/3743). These are the standard
matrices of a [rotation about a coordinate axis](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=92) and of a pure translation. A screw axis
connection describes all of its motions with one such matrix. For a position $\theta$, the connection frame moves
to the exponential of the screw axis {cite:p}`lynch2017modern`
([Proposition 3.25](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=123)):

$$
T(\theta) =
\begin{bmatrix}
R(\omega, \theta) & \theta v + (1 - \cos\theta)\,\omega \times v + (\theta - \sin\theta)\,\omega \times (\omega \times v) \\
0 & 1
\end{bmatrix},
\qquad
R(\omega, \theta) = I + \sin\theta\,[\omega]_\times + (1 - \cos\theta)[\omega]_\times^2,
$$

where $R(\omega, \theta)$ rotates by the angle $\theta$ about the unit vector $\omega$, with $I$ the
identity and $[\omega]_\times$ the matrix that takes the cross product with $\omega$, by
[Rodrigues' rotation formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula). A screw axis
that does not turn the child has $\omega = 0$: there is no rotation and the translation is $\theta v$.
Both cases are one expression without any division, so the connection builds the same symbolic
expression for each of them. The pose of the child relative to the parent is
`parent_T_connection_expression` $\cdot\, T(\theta) \,\cdot$ `connection_T_child_expression`.

For an axis through the origin of the connection frame, $v = h\omega$ is parallel to $\omega$, with
$h$ the advance per radian. Then $\omega \times v = 0$, both correction terms vanish, and the
translation is $h\theta\,\omega$, along the axis. For an axis along $z$, with $\hat{z}$ the unit vector
along it, the three kinds of connection give:

$$
\underbrace{\begin{bmatrix}
\cos\theta & -\sin\theta & 0 & 0 \\
\sin\theta & \cos\theta & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}}_{\text{revolute: } \omega = \hat{z},\ v = 0}
\qquad
\underbrace{\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & \theta \\
0 & 0 & 0 & 1
\end{bmatrix}}_{\text{prismatic: } \omega = 0,\ v = \hat{z}}
\qquad
\underbrace{\begin{bmatrix}
\cos\theta & -\sin\theta & 0 & 0 \\
\sin\theta & \cos\theta & 0 & 0 \\
0 & 0 & 1 & h\theta \\
0 & 0 & 0 & 1
\end{bmatrix}}_{\text{helical: } \omega = \hat{z},\ v = h\hat{z}}
$$

The revolute and prismatic matrices are those of Stelter's figures, the prismatic one with its axis along
$z$. The helical matrix combines both: its rotation is that of the revolute connection, and its
translation grows with the angle by the pitch $h = p / 2\pi$ per radian. For an axis in any other
direction $\hat{s}$, a helical connection rotates about $\hat{s}$ and translates by $h\theta$ along
$\hat{s}$, not along $z$.

(where-the-axis-lies)=
## Where the axis lies

A screw axis may lie anywhere in the frame it is expressed in. Its translation $v$ is that of the point
of the child at the frame's origin, which circles an axis that misses the origin, as
[Displacements, twists and screw axes](https://cram2.github.io/cognitive_robot_abstract_machine/krrood_spatial_math/rates_of_motion.html#displacements-twists-and-screw-axes)
explains. The translation therefore carries the position of the axis,
$v = -\omega \times q + h\omega$ for a point $q$ on the axis and the advance $h$ per radian.
`ScrewAxis.axis_point_closest_to_origin` gives $q$ and `ScrewAxis.translation_per_radian` gives $h$.
The [product of exponentials formula](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=159)
uses this: it expresses the screw axes of all joints of a robot in one base frame, each at its own place.

A connection's motion can therefore be split between its connection frame and its screw axis in many
ways. A [URDF joint](https://wiki.ros.org/urdf/XML/joint), and the joints of MJCF, SDF and USD, consist
of an origin and an axis direction through it, and the parsers turn them into a connection frame at that
origin and a screw axis through it, with $q = 0$. A connection whose axis misses its frame's origin
moves its child just as well. `move_connection_frame_onto_axis` brings it into the parsed form: it
replaces the connection by one whose connection frame is moved by `connection_T_frame_on_axis`, onto
the axis's point closest to the old origin, whose child is placed relative to the moved frame, and whose
screw axis keeps its direction and pitch through the new origin. The child moves exactly as before, and
the name and the degree of freedom stay. A connection already in that form, as
`ScrewAxis.passes_through_origin` tells, is left as it is, without a model change.

Below, a door hangs from a connection whose frame lies at the door's handle side, 0.8 meters from the
hinge:

```{code-cell} ipython3
from krrood_spatial_math.spatial_types import ScrewAxis

hinge_world = World()
wall = Body(name=PrefixedName("wall"))
hinged_door = Body(name=PrefixedName("hinged_door"))
hinge_point = np.array([-0.8, 0.0, 0.0])
with hinge_world.modify_world():
    hinge_world.add_body(wall)
    hinge_world.add_body(hinged_door)
    hinge = ScrewAxisConnection.create_with_dofs(
        world=hinge_world,
        parent=wall,
        child=hinged_door,
        screw_axis=ScrewAxis(
            rotation_per_unit_position=Vector3.Z(),
            translation_per_unit_position=Vector3.from_iterable(
                -np.cross([0.0, 0.0, 1.0], hinge_point)
            ),
        ),
    )
    hinge_world.add_connection(hinge)

hinge.position = np.pi / 2
before = hinge_world.compute_forward_kinematics_np(root=wall, tip=hinged_door)
print("axis point in the connection frame:", hinge.screw_axis.axis_point_closest_to_origin[:3].to_list())

hinge = hinge.move_connection_frame_onto_axis()
hinge.position = np.pi / 2
after = hinge_world.compute_forward_kinematics_np(root=wall, tip=hinged_door)
print("connection frame relative to the wall:", hinge.parent_T_connection_expression.to_np()[:3, 3].tolist())
print("the door at a quarter turn, before and after, agrees:", np.allclose(before, after))
```

Exporters to formats whose joints sit on their axis read the axis's place from
`connection_T_frame_on_axis` without changing the world: the MuJoCo export puts a hinge's position on
the axis, in the child's frame.

```{code-cell} ipython3
:tags: [remove-cell]

assert np.allclose(before, after)
assert np.allclose(hinge.parent_T_connection_expression.to_np()[:3, 3], hinge_point)
assert np.allclose(hinge.connection_T_frame_on_axis.to_np(), np.eye(4))
assert hinge.behaves_like_a_revolute_connection


def distance_from_axis(point: np.ndarray) -> float:
    offset = point - axis_origin
    return np.linalg.norm(offset - offset.dot(axis_direction) * axis_direction)


for angle in np.linspace(0, 2 * np.pi, 7):
    assert np.isclose(distance_from_axis(root_T_child("Revolute", angle)[:3, 3]), 0.15)

assert np.allclose(
    root_T_child("Prismatic", 0.3)[:3, :3], root_T_child("Prismatic", 0)[:3, :3]
)

for kind, handed_pitch in [("Right-handed helical", pitch), ("Left-handed helical", -pitch)]:
    start, after_one_revolution = root_T_child(kind, 0), root_T_child(kind, 2 * np.pi)
    assert np.allclose(after_one_revolution[:3, :3], start[:3, :3])
    assert np.allclose(
        after_one_revolution[:3, 3] - start[:3, 3], handed_pitch * axis_direction
    )

assert door_connection.behaves_like_a_revolute_connection
assert np.isclose(cap_connection.position, np.pi)

helical_pitch = pitch / (2 * np.pi)
for kind, expected_rotation, expected_translation in [
    ("Revolute", True, 0.0),
    ("Prismatic", False, 1.0),
    ("Right-handed helical", True, helical_pitch),
]:
    angle = 0.7
    connection = connections[kind]
    connection.position = angle
    connection_T_moved = (
        np.linalg.inv(connection.parent_T_connection_expression.to_np())
        @ world.compute_forward_kinematics_np(root=root, tip=connection.child)
        @ np.linalg.inv(connection.connection_T_child_expression.to_np())
    )
    expected = np.eye(4)
    if expected_rotation:
        expected[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    expected[2, 3] = expected_translation * angle
    assert np.allclose(connection_T_moved, expected)

for pitch_per_radian, connection in zip(pitches_per_radian, pitch_connections):
    _, advance = one_revolution(connection)
    advance_start, advance_end = (np.array([advance[key][index] for key in "xyz"]) for index in (0, 1))
    assert np.isclose(np.linalg.norm(advance_end - advance_start), 2 * np.pi * pitch_per_radian)
```

## References

```{bibliography}
:filter: docname in docnames
```
