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

# Rates of motion and the displacements they produce

This page explains how a rate of motion, a twist or a screw axis, becomes the
displacement a body makes when it moves at that rate. It also covers how the code keeps
that displacement accurate when the body barely turns.

```{code-cell} ipython3
:tags: [hide-input]

import math
import sys

import krrood.symbolic_math.symbolic_math as sm
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

from krrood_spatial_math.spatial_types import ScrewAxis, Twist, Vector3


def show(figure):
    return HTML(
        figure.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            include_mathjax=False,
            config=dict(displayModeBar=False),
        )
    )


def numbers(*values, digits=2):
    return (
        "("
        + ", ".join(f"{round(v, digits) + 0.0:.{digits}f}" for v in values)
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
```

## Displacements, twists and screw axes

Three types describe how a rigid body moves. Each holds six numbers: a rotational part
and a translational part, given along the axes of the same reference frame.

| Type | Rotational part | Translational part | Per unit of |
|---|---|---|---|
| `Displacement` | the rotation vector: the axis the body turned around, scaled by the angle it turned by | where the body's origin ended up | nothing: it is a finished motion |
| `Twist` | the angular velocity $\omega$ | the velocity $v$ of the point of the body at the frame's origin | time, usually |
| `ScrewAxis` | the angle $\omega$ the body turns by about each axis per unit of position | how far $v$ the point of the body at the frame's origin moves per unit of position | a position, such as a joint's |

For both kinds of rate, $\omega$ names the rotational part and $v$ the translational
part. They differ in what the change of the pose is measured against:

- A **screw axis** is how the pose changes per unit of the joint's position $\theta$:
  an angle in radians for a joint that turns, a distance in meters for one that only
  slides.
- A **twist** is how the pose changes per second. It is the screw axis times the joint's
  speed, $\dot\theta$.

A twist is therefore not the derivative of a screw axis. A joint's screw axis stays the
same however fast the joint moves; the twist scales it by that speed.

The translational part $v$ describes the point of the body that is at the reference
frame's origin, which need not be the body's own origin. If the axis the body turns about
does not run through the frame's origin, that point circles the axis, so $v$ is not zero
even when the body slides nowhere along the axis. A door turning at 1 rad/s about a
hinge 0.5 m beside the frame's origin has $\omega = (0, 0, 1)$ rad/s and $v$ of
0.5 m/s, sideways.

A screw axis describes how a joint moves the body, independent of how fast it does so.
The body turns about a line and slides along it in proportion, so every point of the body
follows a helix around that line {cite}`lynch2017modern`.

The figure moves a body along a screw axis that runs along z through the point
$(0.3, 0, 0)$ and rises 5 cm per radian: $\omega = (0, 0, 1)$ and
$v = (0, -0.3, 0.05)$. The point of the body at the frame's origin circles the axis at a
distance of 0.3 m, so it starts out moving sideways by 0.3 m per radian, and it rises by
0.05 m per radian. At the position $\theta$, the body has turned by the rotation vector
$r = \theta\,\omega$. The box at the top left shows $r$, the translational part times
the position, $t = \theta\,v$, and where the body's origin really is, according to
`ScrewAxis.to_displacement`. The origin is not at $t$; the next section explains why.
Drag the slider to move the body, and drag the figure to look at it from another side.

```{code-cell} ipython3
:tags: [hide-input]

rotation_per_position = np.array([0.0, 0.0, 1.0])
translation_per_position = np.array([0.0, -0.3, 0.05])
screw = ScrewAxis(
    rotation_per_unit_position=Vector3(*rotation_per_position),
    translation_per_unit_position=Vector3(*translation_per_position),
)
positions = np.linspace(0, 4 * math.pi, 73)
displacements = [screw.to_displacement(position=float(p)) for p in positions]
origins = np.array([np.asarray(d.translation[:3]) for d in displacements])
orientations = [
    Rotation.from_rotvec(np.asarray(d.rotation_vector[:3])).as_matrix()
    for d in displacements
]


def screw_state(index):
    traces = [
        go.Scatter3d(
            x=origins[: index + 1, 0],
            y=origins[: index + 1, 1],
            z=origins[: index + 1, 2],
            mode="lines",
            line=dict(color="#333333", width=5),
            name="path of the body's origin",
        )
    ]
    for axis, (color, name) in enumerate(
        (("#d62728", "x"), ("#2ca02c", "y"), ("#1f77b4", "z"))
    ):
        tip = origins[index] + 0.1 * orientations[index][:, axis]
        traces.append(
            go.Scatter3d(
                x=[origins[index][0], tip[0]],
                y=[origins[index][1], tip[1]],
                z=[origins[index][2], tip[2]],
                mode="lines",
                line=dict(color=color, width=8),
                name=f"body's {name} axis",
            )
        )
    position = positions[index]
    values = (
        f"θ = {position:.2f} rad<br>"
        f"r = θω = {numbers(*(position * rotation_per_position))}<br>"
        f"t = θv = {numbers(*(position * translation_per_position))}<br>"
        f"origin = {numbers(*origins[index])}"
    )
    return traces, {"legend.title.text": values}


screw_figure = go.Figure(
    data=[
        go.Scatter3d(
            x=[0.3, 0.3],
            y=[0, 0],
            z=[-0.1, 0.75],
            mode="lines",
            line=dict(color="#e0a800", width=6),
            name="screw axis",
        ),
        go.Scatter3d(
            x=origins[:, 0],
            y=origins[:, 1],
            z=origins[:, 2],
            mode="lines",
            line=dict(color="#bbbbbb", width=2),
            name="the whole helix",
        ),
        *screw_state(0)[0],
    ],
)
screw_figure.update_layout(
    template="plotly_white",
    height=600,
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(
        **legend_box, title=dict(text=screw_state(0)[1]["legend.title.text"])
    ),
    scene=dict(
        xaxis=dict(range=[-0.15, 0.75], title="x (m)"),
        yaxis=dict(range=[-0.45, 0.45], title="y (m)"),
        zaxis=dict(range=[-0.15, 0.75], title="z (m)"),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
    ),
    uirevision="screw",
    sliders=[
        slider(
            [f"{p:.1f}" for p in positions],
            "position θ (rad): ",
            active=0,
            states=[screw_state(index) for index in range(len(positions))],
            traces=[2, 3, 4, 5],
        )
    ],
)
show(screw_figure)
```

### The three types in code

A screw-on lid shows the three types in code. Its screw axis runs along the z axis of
the lid's frame, through the frame's origin. The lid's position is the angle it has
turned by, in radians, and the lid rises 2 mm for every radian:

```{code-cell} ipython3
lid = ScrewAxis(
    rotation_per_unit_position=Vector3(0, 0, 1),
    translation_per_unit_position=Vector3(0, 0, 0.002),
)
```

A quarter turn is the position $\theta = \pi/2$. The lid has then turned by $\pi/2$
about z, and, since its axis runs through its origin, the origin has only risen, by
$\pi/2 \cdot 2\,\text{mm} \approx 3.1\,\text{mm}$:

```{code-cell} ipython3
quarter_turn = lid.to_displacement(position=math.pi / 2)
print("rotation vector:", quarter_turn.rotation_vector[:3].to_list())
print("translation:    ", quarter_turn.translation[:3].to_list())
```

The screw axis says how the lid's pose changes per radian it is turned. The twist says
how it changes per second: it is the screw axis times the lid's speed $\dot\theta$, and
that is what `ScrewAxis.to_twist` computes. Turned at $\dot\theta = 0.5$ rad/s, the lid
turns at $\omega = (0, 0, 0.5)$ rad/s and rises at $v = (0, 0, 0.001)$ m/s:

```{code-cell} ipython3
twist = lid.to_twist(position_velocity=0.5)
print("angular velocity:", twist.angular_velocity[:3].to_list())
print("linear velocity: ", twist.linear_velocity[:3].to_list())
```

At 0.5 rad/s, the quarter turn of $\pi/2$ rad takes $(\pi/2) / 0.5 = \pi$ seconds. So
moving at the twist for $\pi$ seconds gives the same displacement as the quarter turn:

```{code-cell} ipython3
same_turn = twist.to_displacement(duration=math.pi)
assert np.allclose(same_turn, quarter_turn)
```

## From a rate to a displacement

Moving for a while at a constant rate, the body turns by the rotation vector
$r = \omega\,\Delta$, where $\Delta$ is how much the parameter of the rate changes: the
duration for a twist, the change of position for a screw axis. Its origin, however, does
not end up at $t = v\,\Delta$. That is where it would end if it moved without turning.
Because the body turns while it moves, its direction of motion turns with it, and the
origin follows an arc instead of a straight line. It ends up at

$$
t + A(\varphi)\,(r \times t) + B(\varphi)\,(r \times (r \times t)),
\qquad
A(\varphi) = \frac{1 - \cos\varphi}{\varphi^2},
\quad
B(\varphi) = \frac{\varphi - \sin\varphi}{\varphi^3}.
$$

Here $\times$ is the cross product, and $\varphi$ is the angle the body turns by. Since
the rotation vector $r$ is the axis scaled by that angle, $\varphi$ is the length of $r$:
$\varphi = |r|$. This is the exponential of the rate {cite}`sola2018micro`
(equation (172)).

Every term of the sum is a fixed matrix times $t$, so the sum is one matrix times $t$:

$$
J(r)\,t,
\qquad
J(r) = I + A(\varphi)\,[r]_\times + B(\varphi)\,[r]_\times^2,
$$

where $I$ is the identity and $[r]_\times$ is the matrix that takes the cross product
with $r$, $[r]_\times\,t = r \times t$. This matrix $J(r)$ is the *left Jacobian* of $r$
{cite}`eade2017lie` (section 3.2, equation (83)).

Take a body that drives forward at $v = (1, 0, 0)$ m/s while it turns at
$\omega = (0, 0, 1)$ rad/s, for $\Delta$ seconds. Then $r = (0, 0, \Delta)$,
$\varphi = \Delta$, $t = (\varphi, 0, 0)$, $r \times t = (0, \varphi^2, 0)$ and
$r \times (r \times t) = (-\varphi^3, 0, 0)$, so the formula gives
$(\sin\varphi, 1 - \cos\varphi, 0)$, a point on the unit circle's arc:

```{code-cell} ipython3
driving = Twist(
    angular_velocity=Vector3(0, 0, 1), linear_velocity=Vector3(1, 0, 0)
)
angle = 1.3
end = driving.to_displacement(duration=angle).translation
print("end:", end[:3].to_list())
assert np.allclose(end[:3], [math.sin(angle), 1 - math.cos(angle), 0])
```

The figure draws the three parts of the formula, and the box at the top left shows each
of them with its current values. Drag the slider to let the body turn further: the two
correction terms always carry the end of $t$ onto the arc the body really follows, and
they grow with the angle.

```{code-cell} ipython3
:tags: [hide-input]

turned_angles = np.round(np.linspace(0.05, 2.0, 40), 2)


def arrow(start, end, color, name):
    return go.Scatter(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        mode="lines+markers",
        line=dict(color=color, width=4),
        marker=dict(
            symbol="arrow", angleref="previous", size=[0, 16], color=color
        ),
        name=name,
    )


def turning_state(index):
    angle = float(turned_angles[index])
    end = np.asarray(driving.to_displacement(duration=angle).translation[:2])
    first_order_coefficient = (1 - math.cos(angle)) / angle**2
    second_order_coefficient = (angle - math.sin(angle)) / angle**3
    straight = np.array([angle, 0.0])
    first_end = straight + first_order_coefficient * np.array([0.0, angle**2])
    second_end = first_end + second_order_coefficient * np.array(
        [-(angle**3), 0.0]
    )
    assert np.allclose(second_end, end)
    arc = np.linspace(0, angle, 100)
    traces = [
        go.Scatter(
            x=np.sin(arc),
            y=1 - np.cos(arc),
            mode="lines",
            line=dict(color="#2ca02c", width=5),
            name=f"the body's path, ending at {numbers(*end)}",
        ),
        arrow([0, 0], straight, "#7f7f7f", f"t = vΔ = {numbers(angle, 0)}"),
        arrow(
            straight,
            first_end,
            "#e0a800",
            f"A(φ) (r × t) = {first_order_coefficient:.3f} · {numbers(0, angle**2)}",
        ),
        arrow(
            first_end,
            second_end,
            "#9467bd",
            f"B(φ) (r × (r × t)) = {second_order_coefficient:.3f}"
            f" · {numbers(-(angle**3), 0)}",
        ),
    ]
    values = (
        f"Δ = {angle:.2f} s, φ = |r| = {angle:.2f} rad<br>"
        f"r = ωΔ = (0, 0, {angle:.2f})"
    )
    return traces, {"legend.title.text": values}


initial = int(np.argmin(np.abs(turned_angles - 1.3)))
turning_figure = go.Figure(data=turning_state(initial)[0])
turning_figure.update_layout(
    template="plotly_white",
    height=560,
    margin=dict(l=45, r=5, t=5, b=0),
    legend=dict(
        **legend_box,
        title=dict(text=turning_state(initial)[1]["legend.title.text"]),
    ),
    xaxis=dict(range=[-0.6, 2.1], title="x (m)"),
    yaxis=dict(range=[-0.1, 1.55], title="y (m)", scaleanchor="x"),
    sliders=[
        slider(
            [f"{angle:.2f}" for angle in turned_angles],
            "Δ (s): ",
            active=initial,
            states=[
                turning_state(index) for index in range(len(turned_angles))
            ],
            traces=[0, 1, 2, 3],
        )
    ],
)
show(turning_figure)
```

### Why not scipy's matrix exponential

`scipy.linalg.expm` gives the same displacement, and the package's tests check the
formula against it. But `expm` works only on numbers, while the package's positions are
usually symbolic variables, whose expressions the controller differentiates. The formula
above uses only operations the symbolic math has, so it works on those variables too.

## Closed form and series

The formulas for $A$ and $B$ divide by a power of $\varphi$, and $\varphi$ is often
zero: a drawer slides without turning, and a hinge at its zero position has not turned
yet. At $\varphi = 0$ both formulas are $0/0$, although $A$ and $B$ themselves are well
defined there: $A$ tends to $1/2$ and $B$ to $1/6$. Close to zero, the formulas still
give a number, but an inaccurate one. The code therefore computes $A$ and $B$ in two
ways: with the formulas far from zero, and with a series that divides by nothing near
it. This section introduces the series, and the next one shows where the code switches.

- **Closed form:** the formulas above, a fixed number of steps with standard
  functions such as $\cos$ and $\sin$.
- **Series:** their Taylor series around $\varphi = 0$ {cite}`eade2017lie` (section 3.2,
  equations (74) and (75)). Subtracting the Taylor series of the cosine,
  $\cos\varphi = 1 - \varphi^2/2! + \varphi^4/4! - \dots$, from 1 and dividing every term
  by $\varphi^2$ gives the series of $A$; the sine gives $B$ the same way. Here
  $k! = 1 \cdot 2 \cdots k$.

  $$
  A(\varphi) = \frac{1}{2!} - \frac{\varphi^2}{4!} + \frac{\varphi^4}{6!}
  - \frac{\varphi^6}{8!} + \dots,
  \qquad
  B(\varphi) = \frac{1}{3!} - \frac{\varphi^2}{5!} + \frac{\varphi^4}{7!} - \dots
  $$

  A computer can only add up finitely many terms, so the sum is cut off after a few.
  The terms alternate in sign, and for every angle below $\sqrt{12} \approx 3.5$ rad they
  shrink from the first one on. The cut-off sum then misses less than the first term it
  leaves out.

The figure adds up the series of $A$ term by term. Each partial sum matches $A$ near
zero and peels away further out, and every further term widens the range where the two
agree to three digits, shaded green. Hover over the figure to read both values at an
angle.

```{code-cell} ipython3
:tags: [hide-input]

series_angles = np.linspace(0.005, 6.5, 1300)
lowest_value, highest_value = -0.1, 0.85
first_order_values = (1 - np.cos(series_angles)) / series_angles**2
term_texts = (
    "1/2!",
    "φ<sup>2</sup>/4!",
    "φ<sup>4</sup>/6!",
    "φ<sup>6</sup>/8!",
    "φ<sup>8</sup>/10!",
    "φ<sup>10</sup>/12!",
)


def series_state(terms):
    values = sum(
        (-(series_angles**2)) ** term / math.factorial(2 * term + 2)
        for term in range(terms)
    )
    outside = np.flatnonzero((values < lowest_value) | (values > highest_value))
    shown = outside[0] if len(outside) else len(values)
    disagreeing = np.flatnonzero(
        np.abs(values - first_order_values) > 1e-3 * first_order_values
    )
    agreeing = (
        series_angles[disagreeing[0]] if len(disagreeing) else series_angles[-1]
    )
    formula = term_texts[0] + "".join(
        (" − " if term % 2 else " + ") + term_texts[term]
        for term in range(1, terms)
    )
    traces = [
        go.Scatter(
            x=[0, agreeing, agreeing, 0, 0],
            y=[
                lowest_value,
                lowest_value,
                highest_value,
                highest_value,
                lowest_value,
            ],
            mode="lines",
            fill="toself",
            fillcolor="rgba(44, 160, 44, 0.15)",
            line=dict(width=0),
            hoverinfo="skip",
            name=f"agrees to 3 digits up to φ ≈ {agreeing:.2f}",
        ),
        go.Scatter(
            x=series_angles[:shown],
            y=values[:shown],
            mode="lines",
            line=dict(color="#1f77b4", width=3),
            name=f"{terms} term{'s' if terms > 1 else ''}: {formula}",
            hovertemplate="%{y:.4f}",
        ),
    ]
    return traces, {}


series_figure = go.Figure(
    data=[
        go.Scatter(
            x=series_angles,
            y=first_order_values,
            mode="lines",
            line=dict(color="#222222", width=3),
            name="A(φ) = (1 − cos φ) / φ<sup>2</sup>",
            hovertemplate="%{y:.4f}",
        ),
        *series_state(1)[0],
    ]
)
series_figure.update_layout(
    template="plotly_white",
    height=500,
    margin=dict(l=45, r=5, t=5, b=0),
    hovermode="x unified",
    legend=legend_box,
    xaxis=dict(range=[0, 6.5], title="rotation angle φ (rad)"),
    yaxis=dict(range=[lowest_value, highest_value]),
    sliders=[
        slider(
            [str(terms) for terms in range(1, len(term_texts) + 1)],
            "terms of the series: ",
            active=0,
            states=[
                series_state(terms) for terms in range(1, len(term_texts) + 1)
            ],
            traces=[1, 2],
        )
    ],
)
show(series_figure)
```

Both are exact in exact arithmetic, but not with floats. At small angles the closed form
divides $0$ by $0$ at $\varphi = 0$. Shortly before that, $1 - \cos\varphi$ subtracts two
floats that agree in most of their digits, so most of the digits of the difference are
lost. The series has no such subtraction, and its terms shrink fast when $\varphi$ is
small. At $\varphi = 10^{-4}$, compared with the first 15 terms of the series, which are
exact to the last digit of a float below 1 radian:

```{code-cell} ipython3
def exact_first_order_coefficient(angle):
    return sum(
        (-(angle**2)) ** term / math.factorial(2 * term + 2)
        for term in range(15)
    )


angle = 1e-4
print(f"1 - cos φ:   {1 - math.cos(angle):.16e}")
print(f"exact:       {exact_first_order_coefficient(angle) * angle**2:.16e}")
closed_form = (1 - math.cos(angle)) / angle**2
series = (
    1 / math.factorial(2)
    - angle**2 / math.factorial(4)
    + angle**4 / math.factorial(6)
)
exact = exact_first_order_coefficient(angle)


def relative_error_of(value, exact):
    return abs(value - exact) / abs(exact)


for name, value in (("closed form", closed_form), ("series", series)):
    error = relative_error_of(value, exact)
    print(f"{name + ':':12} {value:.16f}  relative error {error:.1e}")
```

## Switching between them

The code uses the closed form at large angles and the first three terms of the series at
small ones {cite}`todesco2023highly` (section 8). The closed form loses about
$2\varepsilon / \varphi^2$ of its value to round-off, where
$\varepsilon \approx 2.2 \cdot 10^{-16}$ is the machine epsilon, the gap between 1 and
the next larger float. Three terms of the series leave out about $\varphi^6 / 8!$,
which is $2\varphi^6 / 8!$ relative to $A(0) = 1/2$. The switch sits where the two are
equal:

$$
\frac{2\varphi^6}{8!} = \frac{2\varepsilon}{\varphi^2}
\quad\Rightarrow\quad
\varphi = (8!\,\varepsilon)^{1/8} \approx 0.042.
$$

Below that angle the series' error is smaller, above it the closed form's. At the switch
both are about $2.6 \cdot 10^{-13}$, which is the worst the switch allows.

The figure measures the relative errors of $A$ (top) and $B$ (bottom) against the first
15 terms of their series. Each line shows the worst error among a few neighbouring
angles, since round-off jumps from one angle to the next. Drag the slider to change the
number of terms $n$: the switch moves to $\varphi = ((2n + 2)!\,\varepsilon)^{1/(2n + 2)}$,
and the box at the top left shows where it is and the worst error it leaves. $B$'s
errors follow $A$'s closely, so one switch serves both.

The black dots measure the code itself, through `ScrewAxis.to_displacement` with three
terms. For a turn by $\varphi$ about the z axis while moving by $\varphi$ along the x
axis, $r \times t = (0, \varphi^2, 0)$, so the y entry of the translation is
$A(\varphi)\,\varphi^2$, and dividing it by $\varphi^2$ gives back the code's $A$. $B$
has no entry of its own: it only shows up in the x entry, $\varphi - B(\varphi)\,\varphi^3$,
and taking it back out of that subtracts two nearly equal numbers, which is the very
loss of digits the figure measures. The bottom row therefore evaluates the code's
formulas for $B$ with numpy.

```{code-cell} ipython3
:tags: [hide-input]

angles = np.geomspace(1e-4, 1, 400)


def exact_second_order_coefficient(angle):
    return sum(
        (-(angle**2)) ** term / math.factorial(2 * term + 3)
        for term in range(15)
    )


exact_values = {
    "A": np.array([exact_first_order_coefficient(angle) for angle in angles]),
    "B": np.array([exact_second_order_coefficient(angle) for angle in angles]),
}
closed_forms = {
    "A": (1 - np.cos(angles)) / angles**2,
    "B": (angles - np.sin(angles)) / angles**3,
}
first_factorials = {"A": 2, "B": 3}
turn_and_move = ScrewAxis(
    rotation_per_unit_position=Vector3(0, 0, 1),
    translation_per_unit_position=Vector3(1, 0, 0),
)
code = np.array(
    [
        float(
            turn_and_move.to_displacement(position=float(angle)).translation.y
        )
        / angle**2
        for angle in angles
    ]
)


def relative_error(values, coefficient):
    exact = exact_values[coefficient]
    return np.maximum(np.abs(values - exact) / exact, 1e-17)


def worst_nearby(errors, neighbours=4):
    padded = np.pad(errors, neighbours, mode="edge")
    return np.array(
        [padded[i : i + 2 * neighbours + 1].max() for i in range(len(errors))]
    )


def switch_angle(terms):
    power = 2 * terms + 2
    return (math.factorial(power) * sys.float_info.epsilon) ** (1 / power)


def series_of(terms, coefficient):
    return sum(
        (-(angles**2)) ** term
        / math.factorial(2 * term + first_factorials[coefficient])
        for term in range(terms)
    )


def switching_error(terms, coefficient):
    switching = np.where(
        angles < switch_angle(terms),
        series_of(terms, coefficient),
        closed_forms[coefficient],
    )
    return relative_error(switching, coefficient)


def switching_state(terms):
    switch = switch_angle(terms)
    power = 2 * terms + 2
    worst = {c: switching_error(terms, c).max() for c in ("A", "B")}
    traces = []
    for row, coefficient in enumerate(("A", "B")):
        traces += [
            go.Scatter(
                x=angles,
                y=worst_nearby(
                    relative_error(series_of(terms, coefficient), coefficient)
                ),
                mode="lines",
                line=dict(color="#1f77b4"),
                legendgroup="series",
                showlegend=row == 0,
                name=f"{terms} terms of the series only",
            ),
            go.Scatter(
                x=[switch, switch],
                y=[1e-17, 100],
                mode="lines",
                line=dict(color="#555555", dash="dash"),
                legendgroup="switch",
                showlegend=row == 0,
                name=f"switch at ({power}! ε)<sup>1/{power}</sup> = {switch:.4f} rad",
            ),
            go.Scatter(
                x=angles,
                y=worst_nearby(switching_error(terms, coefficient)),
                mode="lines",
                line=dict(color="#2ca02c", width=3),
                legendgroup="switching",
                showlegend=row == 0,
                name=f"switching: worst {worst['A']:.1e} for A, {worst['B']:.1e} for B",
            ),
        ]
    return traces, {}


switch_figure = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04
)
for row, coefficient in enumerate(("A", "B")):
    switch_figure.add_trace(
        go.Scatter(
            x=angles,
            y=worst_nearby(
                relative_error(closed_forms[coefficient], coefficient)
            ),
            mode="lines",
            line=dict(color="#d62728"),
            opacity=0.6,
            legendgroup="closed form",
            showlegend=row == 0,
            name="closed form only",
        ),
        row=row + 1,
        col=1,
    )
    if coefficient == "A":
        switch_figure.add_trace(
            go.Scatter(
                x=angles,
                y=relative_error(code, "A"),
                mode="markers",
                marker=dict(color="#000000", size=3),
                name="ScrewAxis.to_displacement",
            ),
            row=1,
            col=1,
        )
initial_traces = switching_state(3)[0]
for index, trace in enumerate(initial_traces):
    switch_figure.add_trace(trace, row=index // 3 + 1, col=1)
switch_figure.update_layout(
    template="plotly_white",
    height=820,
    margin=dict(l=60, r=5, t=5, b=0),
    legend=dict(**legend_box, font=dict(size=11)),
    sliders=[
        slider(
            ["2", "3", "4", "5"],
            "terms of the series n: ",
            active=1,
            states=[switching_state(terms) for terms in (2, 3, 4, 5)],
            traces=[3, 4, 5, 6, 7, 8],
        )
    ],
)
switch_figure.update_xaxes(
    type="log", range=[-4, 0], dtick=1, exponentformat="power"
)
switch_figure.update_xaxes(title="rotation angle φ (rad)", row=2, col=1)
switch_figure.update_yaxes(
    type="log", range=[-17, 2], dtick=2, exponentformat="power"
)
switch_figure.update_yaxes(title="relative error of A(φ)", row=1, col=1)
switch_figure.update_yaxes(title="relative error of B(φ)", row=2, col=1)
assert relative_error(code, "A").max() < 1e-12
show(switch_figure)
```

### How many terms

Any number of terms can be switched the same way. Each further term moves the switch to
a larger angle and makes $A$ and its slope $dA/d\varphi$ more accurate there. Measured
against 50-digit arithmetic:

| Terms | Switch | Worst error of $A$ | Worst error of $dA/d\varphi$ |
|---|---|---|---|
| 2 | 0.0074 rad | $8.2 \cdot 10^{-12}$ | $3.6 \cdot 10^{-6}$ |
| 3 | 0.042 rad | $2.6 \cdot 10^{-13}$ | $5.3 \cdot 10^{-9}$ |
| 4 | 0.12 rad | $2.9 \cdot 10^{-14}$ | $9.3 \cdot 10^{-11}$ |
| 5 | 0.26 rad | $6.5 \cdot 10^{-15}$ | $5.6 \cdot 10^{-12}$ |

A further term costs one multiplication and one addition per coefficient. The code uses
three. With them, $A$ is accurate to $2.6 \cdot 10^{-13}$, less than a picometre on a
displacement of a metre. Four or five terms would gain one more digit in $A$, and two to
three more in its slope.

(why-not-the-value-at-zero)=
### Why not just the value at zero

Todesco and Brüls also describe a simpler way around $0/0$ {cite}`todesco2023highly`
(section 5.1, equation (91)): below some small angle, replace $A$ and $B$ by their
values at zero, $A(0) = 1/2$ and $B(0) = 1/6$. That keeps only the first term of each
series and leaves out about $\varphi^2/12$ of $A$, relative to its value. Balanced
against the closed form's $2\varepsilon/\varphi^2$, the best switch is at
$\varphi = (24\,\varepsilon)^{1/4} \approx 2.7 \cdot 10^{-4}$, and both errors there are
about $6 \cdot 10^{-9}$. That is half of a float's 16 digits, where three terms keep 12.

Derivatives widen the gap. Code built on this package differentiates its expressions,
for example to get the Jacobian a controller needs, which says how a body moves as a
joint moves. A constant has no slope, so below the switch $dA/d\varphi$ would be 0 where
it should be about $-\varphi/12$. Above the switch, the closed form's slope,
$dA/d\varphi = (\varphi \sin\varphi - 2(1 - \cos\varphi))/\varphi^3$, subtracts nearly
equal numbers as well, and loses digits much faster than the value does:

```{code-cell} ipython3
def exact_first_order_slope(angle):
    return sum(
        (-1) ** term
        * 2
        * term
        * angle ** (2 * term - 1)
        / math.factorial(2 * term + 2)
        for term in range(1, 15)
    )


for angle in (1e-2, 1e-3, 2.7e-4):
    closed_form_slope = (
        angle * math.sin(angle) - 2 * (1 - math.cos(angle))
    ) / angle**3
    error = relative_error_of(closed_form_slope, exact_first_order_slope(angle))
    print(f"φ = {angle:.1e}: relative error {error:.1e}")
```

Right above the switch for the value at zero, the slope is off by several percent. The
series has neither problem: it is a polynomial, so its slope is a polynomial too, and
accurate wherever the series is.

The slope of $A$ reaches the derivative of the displacement with respect to the
position. For the turn and move from the figure above, the y entry of the translation is
$A(\varphi)\,\varphi^2 = 1 - \cos\varphi$, and its derivative is $\sin\varphi$. With $A$
held at $1/2$, it would be the derivative of $\varphi^2/2$, which is $\varphi$:

```{code-cell} ipython3
position = sm.FloatVariable(name="position")
turn_and_move = ScrewAxis(
    rotation_per_unit_position=Vector3(0, 0, 1),
    translation_per_unit_position=Vector3(1, 0, 0),
)
y = turn_and_move.to_displacement(position=position).translation.y
derivative = y.jacobian(variables=[position]).compile()

angle = 2.5e-4
exact = math.sin(angle)
from_code = float(derivative(np.array([angle]))[0])
from_value_at_zero = angle
for name, value in (("code", from_code), ("value at zero", from_value_at_zero)):
    error = relative_error_of(value, exact)
    print(f"{name + ':':14} relative error {error:.1e}")
```

Held at its value at zero, the derivative keeps 8 digits. The code's worst, at its own
switch, is $9.8 \cdot 10^{-13}$.
