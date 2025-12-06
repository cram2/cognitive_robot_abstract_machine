# Style Guide

## Notation for Homogenous transformations
### TL;DR: Transformation Matrix Style Guide

#### Core Notation
- **Transforms**: `root_T_tip` (4x4 matrix from tip frame to root frame)
- **Rotations**: `root_R_tip` (4x4 matrix with zero translation)
- **Points**: `root_P_tip` (4D vector with w=1)
- **Vectors**: `root_V_tip` (4D vector with w=0)

#### Key Rules
1. **Match inner frames**: `a_T_c = a_T_b @ b_T_c`
2. **Points - Points = Vectors**: `a_V_error = a_P_p2 - a_P_p1`
3. **Points + Vectors = Points**: `a_P_new = a_P_old + a_V_offset`
4. **Rotation transpose = inverse**: `b_R_a = a_R_b.T`

#### Purpose
Prevents frame confusion errors and makes transformation chains self-documenting through consistent naming conventions.

### Detailed explanation
### Transforms
Transformations are represented using homogeneous transformation matrices.
The 4x4 matrix $_{root}T_{tip}$ refers to the transformation matrix to transform things from frame/body $tip$ to frame/body $root$, or the pose for $tip$ relative to $root$, with:
```{math}
{_{root}T_{tip}} = \begin{bmatrix}
	r_{0,0} & r_{0,1} & r_{0,2} & x \\
	r_{1,0} & r_{1,1} & r_{1,2} & y \\
	r_{2,0} & r_{2,1} & r_{2,2} & z \\
	0 & 0 & 0 & 1
	\end{bmatrix}
```
In code please use `root_T_tip` to refer to such matrices.

### Rotation matrices
Rotation matrices are also represented as 4x4 matrices, where the translation component is 0.
```{math}
{_{root}R_{tip}} =\begin{bmatrix}
	r_{0,0} & r_{0,1} & r_{0,2} & 0 \\
	r_{1,0} & r_{1,1} & r_{1,2} & 0 \\
	r_{2,0} & r_{2,1} & r_{2,2} & 0 \\
	0 & 0 & 0 & 1
	\end{bmatrix}
```
This format is used instead of a 3x3 matrix, to make them compatible with transformation matrices.
In code please use `root_R_tip`.

### Points
Points are represented as 4d vectors:
```{math}
{_{root}P_{tip}} =\begin{bmatrix}
	x \\
	y \\
	z \\
	1
	\end{bmatrix}
```
In code please use `root_P_tip`.

### Vectors
Vectors are represented as 4d vectors:
```{math}
{_{root}V_{tip}} =\begin{bmatrix}
	x \\
	y \\
	z \\
	0
	\end{bmatrix}
```
In code please use `root_V_tip`.

### Frame names
When naming, choose the name of the corresponding bodies whenever possible, e.g.:
```python
map_T_base_link
```

## Combination of different types
Following this notation prevents most trivial transformation mistakes.
When transforming things from one frame to another, make sure that the "inner" frame names match:

### Transform poses
Transforms pose1 from frame b to frame a.
```python
a_T_pose1 = a_T_b @ b_T_pose1  
```

### Combine multiple transformations
```python
a_T_c = a_T_b @ b_T_c 
```

### Invert transformations
```python
from semantic_digital_twin.spatial_types.math import inverse_frame
c_T_a = inverse_frame(a_T_c)
```
Remember that:

```python
import krrood.entity_query_language.utils

krrood.entity_query_language.utils.T != inverse_frame(a_T_c)
```

### Combine rotation matrices and transformations
```python
a_R_r = a_T_b @ b_R_r
a_R_r = a_R_b @ b_R_r  #rotation matrices don't care about translation, so these two are equivalent
```

### Invert rotation matrices

```python
import krrood.entity_query_language.utils
from semantic_digital_twin.spatial_types.math import inverse_frame

b_R_a = inverse_frame(a_R_b)
b_R_a = krrood.entity_query_language.utils.T  # for rotation matrices only, the transpose is equal to its inverse.
```

### Transform points
```python
a_P_p1 = a_T_b @ b_P_p1
```

### Rotating points
Rotating points is weird and could mean that you are doing something wrong.
When you rotate a point, it means you are trying to move the point somewhere else, thus creating a new point.
```python
a_P_p2 = a_R_b @ b_P_p1
```
This is conceptually different from transforming, because there the point stays at the same place, just expressed relative to a different frame.


### Transform Vectors
```python
a_V_v1 = a_T_b @ b_V_v1
a_V_v1 = a_R_b @ b_V_v1  # a vector only expresses a direction, it has no fixed place in space, therefore applying a full transformation or only a rotation results in the same vector.
```

### Subtract points to automatically get a vector
You must ensure that the two points are represented relative to the same reference frame, notice that both have "a" on the left.
Since the last entires for points are 1, the subtraction will automatically have a 0 as 4th entries, which represents a vector.
```python
a_V_error = a_P_p2 - a_P_p1 
```

### Adding points
Make sure that this is actually what you are trying to do, because it doesn't make geometric sense.
The result of such an addition will have "2" as the 4th entry and is not a valid point anymore.

### Adding vectors
You must ensure that both vectors are represented relative to the same reference frame.
This is possible, as the 4th entry remains 0.
```python
a_V_v3 = a_V_v1 + a_V_v2
```

### Translating points with vectors
You must ensure that both vectors are represented relative to the same reference frame.
```python
a_P_p3 = a_P_p1 + a_V_v3
```
