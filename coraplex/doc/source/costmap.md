# Costmaps

A costmap is a square grid of cells laid out on the floor around an origin pose. Every cell holds a rating: a cell
rated above zero is a position that meets the costmap's criterion, and the higher the rating, the better it meets it.
There is, for example, a costmap whose positive cells are every position from which a certain object is visible.

In CoraPlex costmaps are what the locations in {mod}`coraplex.locations.locations` sample their poses from (see
{doc}`notebooks/location_designator`). A {class}`~coraplex.locations.locations.ReachabilityLocation`, for example,
merges an occupancy costmap with a ring costmap, so its poses are positions where the robot can stand that are also at
the right distance to reach its target.

All costmaps live in {mod}`coraplex.locations.costmaps` and are dataclasses, so their parameters are best given as
keyword arguments. Every costmap needs the {class}`~semantic_digital_twin.world.World` it is built for, the
`resolution` (the edge length of a cell, in meters) and the `origin` it is centred on. Four kinds of costmaps are
implemented:

* Occupancy Costmap
* Visibility Costmap
* Gaussian Costmap
* Ring Costmap

## Occupancy Costmap

An occupancy costmap marks every position where the robot can stand without colliding with anything. For every cell, a
ray is cast from just above the height of the robot's base straight down to the ground; a cell whose ray hits something
is occupied. The robot itself, whatever it carries and the floors it drives on do not count as obstacles. Obstacles are
then inflated by `distance_to_obstacle`, so the robot keeps that distance to them.

```python
from coraplex.locations.costmaps import OccupancyCostmap
from semantic_digital_twin.spatial_types.spatial_types import Pose

occupancy = OccupancyCostmap(
    resolution=0.02,
    width=200,
    height=200,
    origin=Pose.from_xyz_rpy(1, 0.3, 0, reference_frame=world.root),
    world=world,
    robot_view=robot,
    distance_to_obstacle=0.2,
)
```

See {class}`~coraplex.locations.costmaps.OccupancyCostmap` for the full parameter reference.

For the common case there is the classmethod {meth}`~coraplex.locations.costmaps.OccupancyCostmap.default_map`. It
centres the costmap on the floor below a target pose and inflates obstacles by the radius of the robot's base. The
resolution and the number of cells along each side are optional keyword arguments.

```python
occupancy = OccupancyCostmap.default_map(context, target_pose)
```

## Visibility Costmap

A visibility costmap marks every position from which the robot can see its origin. Four depth images are rendered
from the origin, one in each direction. A position counts as visible if, in these images, nothing blocks the line of
sight from the origin to any height between `minimum_height` and `maximum_height` above that position, the heights a
camera can be at. A {class}`~coraplex.locations.locations.VisibilityLocation` takes these heights from the robot's
default camera.

```python
from coraplex.locations.costmaps import VisibilityCostmap
from semantic_digital_twin.spatial_types.spatial_types import Pose

visibility = VisibilityCostmap(
    minimum_height=1.27,
    maximum_height=1.6,
    resolution=0.02,
    width=200,
    height=200,
    origin=Pose.from_xyz_rpy(1, 0.3, 0, reference_frame=world.root),
    world=world,
)
```

See {class}`~coraplex.locations.costmaps.VisibilityCostmap` for the full parameter reference. The method is explained
in Lorenz Mösenlechner's [PhD thesis](https://mediatum.ub.tum.de/doc/1239461/1239461.pdf) (page 173).

## Gaussian Costmap

A gaussian costmap is a 2D gaussian distribution with its peak at the centre of the costmap, which favours positions
close to the origin. A small square at the very centre is cut out, so the robot does not stand on top of the origin.
Here `mean` is the number of cells along each side of the costmap and `sigma` the standard deviation, in cells.

```python
from coraplex.locations.costmaps import GaussianCostmap
from semantic_digital_twin.spatial_types.spatial_types import Pose

gauss = GaussianCostmap(
    mean=200,
    sigma=15,
    resolution=0.02,
    origin=Pose.from_xyz_rpy(1, 0.3, 0, reference_frame=world.root),
    world=world,
)
```

See {class}`~coraplex.locations.costmaps.GaussianCostmap` for the full parameter reference.

## Ring Costmap

A ring costmap is like a gaussian costmap shaped as a donut: the highest ratings form a ring at `distance` meters from
the origin, falling off with `standard_deviation` cells to either side. This is what reaching needs: the robot has to
stand close enough to the target, but not on top of it.

```python
from coraplex.locations.costmaps import RingCostmap
from semantic_digital_twin.spatial_types.spatial_types import Pose

ring = RingCostmap(
    standard_deviation=15,
    distance=0.7,
    resolution=0.02,
    width=200,
    height=200,
    origin=Pose.from_xyz_rpy(1, 0.3, 0, reference_frame=world.root),
    world=world,
)
```

The classmethod {meth}`~coraplex.locations.costmaps.RingCostmap.from_arm_reach_distance` draws the ring at a fraction
of an arm's length around a target, which is how a {class}`~coraplex.locations.locations.ReachabilityLocation` uses it.

```python
from coraplex.datastructures.enums import ReachFraction

ring = RingCostmap.from_arm_reach_distance(
    context, robot.left_arm, target_pose, ReachFraction.GRASPING
)
```

See {class}`~coraplex.locations.costmaps.RingCostmap` for the full parameter reference.

## Sampling Poses from a Costmap

Every costmap is a {class}`~coraplex.locations.base.Location` itself, so iterating it yields pose candidates, at most
`number_of_samples` of them, drawn with its `seed`. {meth}`~coraplex.locations.costmaps.Costmap.sample` draws a given
number of candidates directly:

```python
for pose in ring.sample(number_of_samples=10, seed=0):
    print(pose)
```

The candidates are drawn as follows:

* The costmap is split into segments: groups of neighbouring cells rated above zero. The segment with the best rated
  cell is sampled from first.
* The samples are shared out among the segments in proportion to their summed ratings.
* Within a segment, cells are drawn at random, weighted by their rating, and no cell is drawn twice.
* Every candidate is a pose in the world frame, at the position of its cell, facing the costmap's origin.

The same seed always draws the same candidates from the same costmap; a seed of `None` draws different ones each time.

## Visualization of Costmaps

The {func}`~coraplex.locations.costmaps.plot_grid` function plots the 2D numpy array that holds a costmap's ratings with
matplotlib:

```python
from coraplex.locations.costmaps import plot_grid

plot_grid(visibility.map)
```

## Merging Costmaps

Merging costmaps gives a costmap whose cells meet the criteria of all of them. For example, merging a visibility and an
occupancy costmap gives the positions where the robot can stand and see a specific point. Costmaps are merged with the
`&` operator (`+` does the same):

```python
visible_and_free = occupancy & visibility
```

A merged cell is rated above zero only if it is rated above zero in both costmaps. Its rating is the product of the two
ratings, scaled so the best cell is rated 1. The merged costmap keeps the `number_of_samples` and `seed` of the left
one.

Only costmaps that cover the same cells can be merged, so a `ValueError` is raised unless both costmaps have:

* the same width and height,
* the same origin, apart from its height,
* the same resolution,
* the same world.
