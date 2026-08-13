"""
Convert the downloaded chemical-laboratory FBX asset into the demo's environment mesh.

Produces ``meshes/chemical_laboratory.obj``/``.mtl`` next to this script: the whole
laboratory interior as one z-up mesh, scaled to real-world size and with its floor at
``z = 0``. The ceiling and its light fixtures are removed so an orbiting viewer can
look into the room, and fixtures outside the room's walls are dropped.

The asset ships without textures and paints every surface the same grey, so the
conversion recolors it like a real laboratory: glassware keeps its transparency
with a light glass tint, walls become off-white over a light epoxy floor, cabinets
steel blue, bench-top slabs charcoal, tall units like fume hoods light grey-blue,
chairs lab-stool blue, the scientist desks wood with dark electronics, and the
equipment standing on the benches gets muted accent colors picked per object.
Every paint also carries metallic and roughness values -- painted steel for the
cabinets and fume hoods, a satin sheen on the work tops and instrument housings,
matte plaster and wood -- which the ``.mtl`` export encodes in its reflection model
(``Ka`` for metallic, ``Ns`` for glossiness, ``illum 3`` for reflective surfaces).

Run once with Blender (the generated files are checked in)::

    blender --background --python convert_environment.py -- <path-to-fbx>

The FBX is the ``chemical laboratory`` asset; by default it is read from
``~/Downloads/hemical-laboratory/source/chemical laboratory.fbx``.
"""

import hashlib
import os
import sys
from dataclasses import dataclass

import bpy
from mathutils import Vector

# %% conversion parameters

SCALE = 15.0
"""
Uniform scale from asset units to meters.

The asset's lab benches are 0.06 units tall; this scale puts their work surfaces at a
realistic bench height of 0.9 meters, giving the room a 2.4 meter ceiling.
"""

CEILING_CUTOFF = 0.145
"""
Height in asset units above which whole objects are removed.

The ceiling plane and its lamp fixtures all start above this height; removing them
leaves the room open to a bird's-eye camera.
"""

ROOM_MAXIMUM_X = 0.20
"""
The room's +x wall sits at 0.19 asset units; objects entirely beyond it are outdoor
fixtures invisible from inside and are removed.
"""

FLOOR_OBJECT_NAME = "pPlane1"
"""
The object whose top surface is the laboratory floor; the export rests it on
``z = 0``.
"""

WALL_OBJECT_NAMES = {
    "pPlane23",
    "pPlane24",
    "pPlane26",
    "polySurface653",
    "pCube42",
    "pCube43",
    "pCube44",
    "pCube45",
    "pCube91",
}
"""
The room's wall planes and corner columns.
"""

FLOOR_OBJECT_NAMES = {"pPlane1", "pCube135"}
"""
The floor plane and the slab directly beneath it.
"""

DESK_GROUP_NAMES = {"locator2", "locator3"}
"""
The scene groups holding the two scientist desks and their computer equipment.
"""


@dataclass(frozen=True)
class Paint:
    """
    The full surface finish a category of objects is painted with.
    """

    color: tuple[float, float, float, float]
    """
    The paint's base color.
    """

    metallic: float = 0.0
    """
    How metallic the surface is, 1 being bare metal.
    """

    roughness: float = 0.6
    """
    How rough the surface is, 0 being a mirror finish.
    """


FLOOR_PAINT = Paint((0.70, 0.73, 0.70, 1.0), metallic=0.0, roughness=0.35)
"""
Light grey-green epoxy floor with a slight sheen.
"""

WALL_PAINT = Paint((0.93, 0.92, 0.88, 1.0), metallic=0.0, roughness=0.85)
"""
Warm off-white matte plaster walls.
"""

CHAIR_PAINT = Paint((0.20, 0.40, 0.65, 1.0), metallic=0.45, roughness=0.45)
"""
Lab-stool blue chairs on metal frames.
"""

CABINET_PAINT = Paint((0.25, 0.42, 0.58, 1.0), metallic=0.75, roughness=0.35)
"""
Steel-blue painted-metal bench and island cabinets.
"""

BENCH_TOP_PAINT = Paint((0.16, 0.17, 0.19, 1.0), metallic=0.55, roughness=0.30)
"""
Charcoal work-top slabs with a satin brushed finish.
"""

TALL_UNIT_PAINT = Paint((0.62, 0.70, 0.76, 1.0), metallic=0.75, roughness=0.35)
"""
Light grey-blue sheet-metal fume hoods, wall cabinets and door frames.
"""

DESK_WOOD_PAINT = Paint((0.58, 0.44, 0.30, 1.0), metallic=0.0, roughness=0.55)
"""
Wooden desk bodies.
"""

ELECTRONICS_PAINT = Paint((0.16, 0.17, 0.19, 1.0), metallic=0.35, roughness=0.45)
"""
Dark monitors, keyboards and other desk electronics.
"""

GLASS_PAINT = Paint((0.70, 0.82, 0.84, 1.0), metallic=0.0, roughness=0.15)
"""
Light glass tint; each glass surface keeps the transparency it was authored with.
"""

ACCENT_PAINTS = [
    Paint((0.85, 0.55, 0.15, 1.0), metallic=0.55, roughness=0.40),
    Paint((0.70, 0.25, 0.20, 1.0), metallic=0.55, roughness=0.40),
    Paint((0.15, 0.50, 0.50, 1.0), metallic=0.55, roughness=0.40),
    Paint((0.45, 0.55, 0.25, 1.0), metallic=0.55, roughness=0.40),
    Paint((0.35, 0.50, 0.65, 1.0), metallic=0.55, roughness=0.40),
    Paint((0.50, 0.35, 0.60, 1.0), metallic=0.55, roughness=0.40),
    Paint((0.86, 0.87, 0.88, 1.0), metallic=0.70, roughness=0.35),
    Paint((0.86, 0.87, 0.88, 1.0), metallic=0.70, roughness=0.35),
    Paint((0.90, 0.91, 0.92, 1.0), metallic=0.70, roughness=0.35),
]
"""
Muted metallic accent paints for the equipment standing on the benches: amber, brick
red, teal, olive, steel blue and violet, padded with brushed steel so most devices
stay neutral and the accents read as labels and housings rather than as a rainbow.
"""

BENCH_TOP_HEIGHT = 0.055
"""
Height in asset units just below the 0.06-unit work surfaces; objects starting above it
stand on a bench.
"""

BENCH_TOP_SLAB_MAXIMUM_THICKNESS = 0.012
"""
Slabs at work-surface height at most this thick in asset units are work tops rather than
whole bench bodies.
"""

TALL_UNIT_MINIMUM_HEIGHT = 0.10
"""
Objects reaching above this height in asset units are wall-height units like fume hoods
and storage cabinets.
"""

DESK_BODY_MINIMUM_EXTENT = 0.03
"""
Desk-group objects larger than this in some direction, in asset units, are desk bodies
rather than the electronics standing on them.
"""

OPAQUE_ALPHA_THRESHOLD = 0.99
"""
Material slots with an alpha below this stay glass; the rest are recolored opaquely.
"""

OUTPUT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")
"""
Where the converted mesh lands.
"""

DEFAULT_FBX_PATH = os.path.expanduser(
    "~/Downloads/hemical-laboratory/source/chemical laboratory.fbx"
)
"""
Where the downloaded asset lives unless a path is passed after ``--``.
"""

# %% mesh import and pruning


def fbx_path_from_arguments() -> str:
    """
    :return: The FBX path passed after ``--``, or the default download location.
    """
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1]
    return DEFAULT_FBX_PATH


def world_bounding_box(mesh_object: bpy.types.Object) -> tuple[Vector, Vector]:
    """
    :param mesh_object: The object to measure.
    :return: The world-space minimum and maximum corners of the object's bounding box.
    """
    corners = [
        mesh_object.matrix_world @ Vector(corner) for corner in mesh_object.bound_box
    ]
    minimum = Vector(
        (
            min(c.x for c in corners),
            min(c.y for c in corners),
            min(c.z for c in corners),
        )
    )
    maximum = Vector(
        (
            max(c.x for c in corners),
            max(c.y for c in corners),
            max(c.z for c in corners),
        )
    )
    return minimum, maximum


def import_laboratory(fbx_path: str) -> list[bpy.types.Object]:
    """
    :param fbx_path: The FBX file holding the laboratory scene.
    :return: All imported mesh objects.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    return [obj for obj in bpy.data.objects if obj.type == "MESH"]


def prune_hidden_fixtures(meshes: list[bpy.types.Object]) -> list[bpy.types.Object]:
    """
    Remove the ceiling, its lamps, and outdoor fixtures from the scene.

    :param meshes: All mesh objects of the imported scene.
    :return: The mesh objects that remain part of the room.
    """
    kept = []
    for mesh_object in meshes:
        minimum, _ = world_bounding_box(mesh_object)
        if minimum.z > CEILING_CUTOFF or minimum.x > ROOM_MAXIMUM_X:
            bpy.data.objects.remove(mesh_object, do_unlink=True)
        else:
            kept.append(mesh_object)
    return kept


# %% recoloring


def principled_node(material: bpy.types.Material) -> bpy.types.ShaderNode:
    """
    :param material: A node-based material.
    :return: Its Principled BSDF shader node.
    """
    return next(
        node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    )


def solid_material(name: str, paint: Paint, alpha: float) -> bpy.types.Material:
    """
    :param name: The material's name, reused if it already exists.
    :param paint: The surface finish the material renders with.
    :param alpha: The material's transparency, 1 being opaque.
    :return: A Principled material with that finish.
    """
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = principled_node(material)
    shader.inputs["Base Color"].default_value = paint.color
    shader.inputs["Metallic"].default_value = paint.metallic
    shader.inputs["Roughness"].default_value = paint.roughness
    shader.inputs["Alpha"].default_value = alpha
    material.diffuse_color = (*paint.color[:3], alpha)
    material.metallic = paint.metallic
    material.roughness = paint.roughness
    material.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"
    return material


def scene_group_of(mesh_object: bpy.types.Object) -> str:
    """
    :param mesh_object: An object of the laboratory scene.
    :return: The name of the top-level scene group the object belongs to.
    """
    ancestor = mesh_object
    while ancestor.parent is not None:
        ancestor = ancestor.parent
    return ancestor.name.split(".")[0]


def accent_paint(mesh_object: bpy.types.Object) -> Paint:
    """
    :param mesh_object: A piece of equipment standing on a bench.
    :return: Its accent paint, picked stably from :data:`ACCENT_PAINTS` by name.
    """
    digest = hashlib.md5(mesh_object.name.encode()).digest()
    return ACCENT_PAINTS[digest[0] % len(ACCENT_PAINTS)]


def category_paint(mesh_object: bpy.types.Object) -> Paint:
    """
    :param mesh_object: An object of the laboratory scene.
    :return: The paint its opaque surfaces get.
    """
    if mesh_object.name.split(".")[0] in FLOOR_OBJECT_NAMES:
        return FLOOR_PAINT
    if mesh_object.name.split(".")[0] in WALL_OBJECT_NAMES:
        return WALL_PAINT
    if mesh_object.name.startswith("chair"):
        return CHAIR_PAINT
    if scene_group_of(mesh_object) in DESK_GROUP_NAMES:
        minimum, maximum = world_bounding_box(mesh_object)
        if max(maximum - minimum) > DESK_BODY_MINIMUM_EXTENT:
            return DESK_WOOD_PAINT
        return ELECTRONICS_PAINT
    minimum, maximum = world_bounding_box(mesh_object)
    is_at_work_surface_height = BENCH_TOP_HEIGHT < maximum.z < BENCH_TOP_HEIGHT + 0.013
    if (
        is_at_work_surface_height
        and maximum.z - minimum.z < BENCH_TOP_SLAB_MAXIMUM_THICKNESS
    ):
        return BENCH_TOP_PAINT
    if minimum.z > BENCH_TOP_HEIGHT:
        return accent_paint(mesh_object)
    if maximum.z > TALL_UNIT_MINIMUM_HEIGHT:
        return TALL_UNIT_PAINT
    return CABINET_PAINT


def recolor(meshes: list[bpy.types.Object]) -> None:
    """
    Paint the untextured asset: opaque slots get their object's category paint, glass
    slots keep their transparency under a light glass tint.

    :param meshes: All mesh objects of the room.
    """
    for mesh_object in meshes:
        paint = category_paint(mesh_object)
        for slot in mesh_object.material_slots:
            if slot.material is None:
                slot.material = solid_material("painted_cabinet", CABINET_PAINT, 1.0)
                continue
            alpha = principled_node(slot.material).inputs["Alpha"].default_value
            if alpha < OPAQUE_ALPHA_THRESHOLD:
                slot.material = solid_material(
                    "painted_glass_%.2f" % alpha, GLASS_PAINT, alpha
                )
            else:
                slot.material = solid_material(
                    "painted_%.2f_%.2f_%.2f_m%.2f" % (*paint.color[:3], paint.metallic),
                    paint,
                    1.0,
                )


# %% joining, scaling and export


def join_into_single_mesh(meshes: list[bpy.types.Object]) -> bpy.types.Object:
    """
    :param meshes: The mesh objects to merge.
    :return: One mesh object containing the whole room, with transforms applied.
    """
    bpy.ops.object.select_all(action="DESELECT")
    for mesh_object in meshes:
        mesh_object.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return joined


def scale_to_world_size(room: bpy.types.Object, floor_top: float) -> None:
    """
    Scale the room to meters and rest its floor surface on ``z = 0``.

    :param room: The joined room mesh.
    :param floor_top: The floor surface's height in asset units before scaling.
    """
    room.scale = (SCALE, SCALE, SCALE)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    room.location.z = -floor_top * SCALE
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def export_room(room: bpy.types.Object) -> str:
    """
    :param room: The joined room mesh.
    :return: The path of the exported OBJ file.
    """
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIRECTORY, "chemical_laboratory.obj")
    bpy.ops.object.select_all(action="DESELECT")
    room.select_set(True)
    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_selected_objects=True,
        export_materials=True,
        forward_axis="Y",
        up_axis="Z",
    )
    return output_path


meshes = prune_hidden_fixtures(import_laboratory(fbx_path_from_arguments()))
recolor(meshes)
floor_top = world_bounding_box(bpy.data.objects[FLOOR_OBJECT_NAME])[1].z
room = join_into_single_mesh(meshes)
scale_to_world_size(room, floor_top)
output_path = export_room(room)

minimum, maximum = world_bounding_box(room)
print(
    "exported %s: min=(%.3f, %.3f, %.3f) max=(%.3f, %.3f, %.3f)"
    % (output_path, minimum.x, minimum.y, minimum.z, maximum.x, maximum.y, maximum.z)
)
