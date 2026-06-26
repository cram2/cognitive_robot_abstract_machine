========
Costmaps
========

Costmaps are way to describe positions in a defined area around a pose with respect to certain constrains. For example,
there is a costmap which contains every position from which a certain object is visible.

In CoraPlex these costmaps are used to dynamically generate poses for certain criteria like visibility, reachability
or occupancy. So if you, for example, want to find a position from where the robot can see a certain object you would
generate a costmap for the visibility and one for occupancy. These costmaps can then be merged with one another and
result in a costmap which contains every position from which the object is visible and where the robot can stand.

All costmaps live in :mod:`coraplex.locations.costmaps` and are dataclasses, so their parameters are best given as
keyword arguments. Every costmap needs the :class:`~semantic_digital_twin.world.World` it is built for. Currently there
are four types of costmaps implemented:

    * Occupancy Costmap
    * Visibility Costmap
    * Gaussian Costmap
    * Ring Costmap


-----------------
Occupancy Costmap
-----------------
Occupancy costmaps represent all positions that don't have any objects positioned above them. Meaning the robot can
stand there without colliding with anything. The map is generated directly from the semantic digital twin world by
casting rays around the origin. A user can additionally specify a value by which obstacles should be inflated, creating
a boundary around obstacles to avoid colliding with them.

.. code-block:: python

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

The given parameter are:

    * ``resolution``: how many metre a single cell of the costmap represents (0.02 translates to 2 centimetre).
    * ``width`` / ``height``: the dimensions of the costmap in cells. Costmaps are usually created as a square.
    * ``origin``: the pose around which the costmap is centered.
    * ``world``: the semantic digital twin world the costmap is built from.
    * ``robot_view``: the robot semantic annotation used to create the map.
    * ``distance_to_obstacle``: the inflation radius around obstacles in metre.

For the common case of a costmap centered on a target pose with the inflation radius derived from the robot base, there
is the convenience classmethod :meth:`~coraplex.locations.costmaps.OccupancyCostmap.default_map`.

.. code-block:: python

    occupancy = OccupancyCostmap.default_map(context, target_pose)

You can see an image of the final Occupancy costmap with an inflation radius of 0.2 m below.

.. image:: _static/images/occupancy_costmap.png

------------------
Visibility Costmap
------------------

Visibility costmaps show the visibility for a specific position in a restricted area. This means every position from
which the robot can see the position given as the map origin.

The visibility costmap is created by taking depth images from the origin
position of the costmap and then checking which positions are occluded by other objects. Essentially, this specifies
how far you can look from the object in all direction. Afterwards, a 2D representation is created from these depth images.

.. code-block:: python

    from coraplex.locations.costmaps import VisibilityCostmap
    from semantic_digital_twin.spatial_types.spatial_types import Pose

    visibility = VisibilityCostmap(
        min_height=1.27,
        max_height=1.6,
        resolution=0.02,
        width=200,
        height=200,
        origin=Pose.from_xyz_rpy(1, 0.3, 0, reference_frame=world.root),
        world=world,
    )

The parameter for the constructor are:

    * ``min_height`` / ``max_height``: the lowest and highest position of the camera from the ground. This accounts for
      a movable camera, for example when the torso moves. If the camera height is fixed, set both to the same value.
    * ``resolution``: how many metre a cell of the costmap represents.
    * ``width`` / ``height``: the dimensions of the resulting costmap in cells.
    * ``origin``: the position the costmap is centered around as well as the position for which the visibility is
      calculated.
    * ``world``: the semantic digital twin world.

A simple visibility costmap with two objects can be seen below.

.. image:: _static/images/visibility_costmap.png

----------------
Gaussian Costmap
----------------

A gaussian costmap is essentially a 2D gauss distribution with its peak at the centre of the the costmap. Gaussian
costmaps are, for example, used to approximate reachability of objects. The idea being that to reach an object the
robot has to be relatively close to the object to reach it.

Since all other objects use just 0 or 1 to represent if an entry in the costmap is valid according to their respective
constraint Gaussian Costmaps add some variance to highlight a certain point. This is especially useful when keeping in
mind that sampling from the costmap is based on the maximum likelihood, meaning the cells with the highest value will be
sampled first.

.. code-block:: python

    from coraplex.locations.costmaps import GaussianCostmap
    from semantic_digital_twin.spatial_types.spatial_types import Pose

    gauss = GaussianCostmap(
        mean=200,
        sigma=15,
        resolution=0.02,
        origin=Pose.from_xyz_rpy(1, 0.3, 0, reference_frame=world.root),
        world=world,
    )

The parameter given to the gaussian costmap are:

    * ``mean``: the mean of the gaussian distribution. This also specifies the side length of the resulting (square)
      costmap.
    * ``sigma``: the sigma of the gaussian distribution.
    * ``resolution`` / ``origin`` / ``world``: the same as for the other costmaps.

A plot of the gaussian costmap can be seen below. This is a matplotlib plot of the costmap to better show the
distribution.

.. image:: _static/images/gaussian_costmap.png


------------
Ring Costmap
------------

A ring costmap is similar to the gaussian costmap but looks more like a donut: the high-probability area forms a ring
at a configurable distance from the center. This is useful to create poses for reaching a point, where standing exactly
on top of the target is not desirable.

.. code-block:: python

    from coraplex.locations.costmaps import RingCostmap
    from semantic_digital_twin.spatial_types.spatial_types import Pose

    ring = RingCostmap(
        std=15,
        distance=0.7,
        resolution=0.02,
        width=200,
        height=200,
        origin=Pose.from_xyz_rpy(1, 0.3, 0, reference_frame=world.root),
        world=world,
    )

The parameter given to the ring costmap are:

    * ``std``: the standard deviation of the gaussian distribution that forms the ring.
    * ``distance``: the distance between the center of the costmap and the center of the ring. A distance of ``0``
      results in a gaussian costmap.
    * ``resolution`` / ``width`` / ``height`` / ``origin`` / ``world``: the same as for the other costmaps.

-------------------------
Visualization of Costmaps
-------------------------

For a comprehensive visualization of a costmap there is the :func:`~coraplex.locations.costmaps.plot_grid` function in
``costmaps.py``. It creates a matplotlib plot of the 2D numpy array which represents the costmap and can be called as
follows:

.. code-block:: python

    from coraplex.locations.costmaps import plot_grid

    plot_grid(visibility.map)

The image for the gaussian costmap shows such a matplotlib plot.

----------------
Merging Costmaps
----------------

It is possible to merge different costmaps to create a costmap that contains positions that adhere to more than one
constraint. For example, if you merge a visibility and occupancy costmap you get a costmap that contains positions
where the robot can stand and see a specific point. Costmaps support the ``+`` operator (additive merge) and the ``&``
operator.

.. code-block:: python

    reachability_map = occupancy + gauss
    visible_and_free = occupancy & visibility

To be able to merge different costmaps there are a few restrictions that you have to follow. The restrictions are:

    * The costmaps must have the same size
    * The costmaps must have the same origin
    * The costmaps must have the same resolution
    * The costmaps must belong to the same world

These restrictions make it much easier to merge costmaps and also reduce the probability of errors occurring in the
resulting costmap. Since for all costmaps these parameter can be set when creating them, it shouldn't pose a
problem to match these parameter for all created costmaps, making them able to be merged.
