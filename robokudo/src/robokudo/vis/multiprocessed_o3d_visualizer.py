"""Open3D-based visualization for RoboKudo pipelines.

This module provides 3D visualization capabilities for RoboKudo pipelines using Open3D.
It handles:

* 3D geometry visualization
* Point cloud rendering
* Camera control
* Coordinate frame display
* Window management
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from threading import Lock, Thread

import numpy as np
import open3d as o3d  # this import creates a SIGINT during unit test execution....
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Self,
    Tuple,
    Type,
    TypeVar,
    Union,
    override,
)

from robokudo.annotators.core import BaseAnnotator
from robokudo.defs import PACKAGE_NAME
from robokudo.vis.o3d_visualizer import Viewer3D
from robokudo.vis.visualizer import Visualizer

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    import numpy.typing as npt


class O3DVisualizer(Visualizer, Visualizer.Observer):
    """Open3D-based visualizer for 3D geometry data.

    This class provides visualization of 3D geometry data from pipeline annotators using
    Open3D windows. It supports:

    * 3D geometry visualization
    * Point cloud rendering
    * Camera control
    * Coordinate frame display
    * Shared visualization state

    .. note::
        This Visualizer works with a shared state and needs notifications
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Open3D visualizer."""
        super().__init__(*args, **kwargs)

        self.viewer3d: Optional[MultiprocessedViewer3D] = None
        """Open3D viewer instance"""

        self.shared_visualizer_state.register_observer(self)

    def notify(
        self,
        observable: Visualizer.Observable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle notification of state changes.

        :param observable: The object that sent the notification
        """
        self.update_output = True

    def tick(self) -> None:
        """Update the visualization display.

        This method:

        * Initializes viewer if needed
        * Gets current annotator outputs
        * Updates display if needed
        * Handles viewer lifecycle

        :returns: False if visualization should terminate, True otherwise
        """
        if self.viewer3d is None:
            self.viewer3d = MultiprocessedViewer3D(self.window_title() + "_3D")

        annotator_outputs = self.get_visualized_annotator_outputs_for_pipeline()

        active_annotator_instance: BaseAnnotator = (
            self.shared_visualizer_state.active_annotator
        )

        self.update_output_flag_for_new_data()

        if self.update_output:
            self.update_output = False

            geometries = None
            # We might not yet have visual output set up for this annotator
            # This might happen in dynamic perception pipelines, where annotators have not been set up
            # during construction of the tree AND don't generate cloud outputs.
            # => Fetch geometry if present
            if active_annotator_instance.name in annotator_outputs.outputs:
                geometries = annotator_outputs.outputs[
                    active_annotator_instance.name
                ].geometries

            self.viewer3d.update_cloud(geometries)

        tick_result = (
            self.viewer3d.tick()
        )  # right now, this is the last update call. if that's true, the GUI is happy.

        if not tick_result:
            self.indicate_termination_var = True

    def window_title(self) -> str:
        """Get the window title for this visualizer."""
        return self.identifier()


@dataclass(
    slots=True,
    frozen=True,
)
class MemoryMap(object):
    """A base memory map for shared memory."""

    byte_size: int
    """Size of the underlying data in bytes."""


T = TypeVar("T", bound=object)


@dataclass(slots=True, frozen=True)
class ObjectMemoryMap(MemoryMap, Generic[T]):
    """A memory map for an object in shared memory."""

    mapped_attributes = []  # Class attribute (no type hint)
    """A list of (attribute name, attribute type) for the attributes mapped by the memory map."""

    @classmethod
    def get_memory_map(cls, obj: Any) -> Union[ArrayMemoryMap, ObjectMemoryMap]:
        """Get the memory map for the given object.

        This works for iterables that can be converted to numpy arrays, sets and objects that have a memory map
        available in ObjectMemoryMapFactory. If the given object is not a set, numpy array or object for which a memory
        map is available, the default behaviour will try to read it to a numpy array.

        :param obj: The object to get the memory map for.
        :return: The object or array memory map.
        """
        if isinstance(obj, set):
            return ArrayMemoryMap.from_numpy_array(np.asarray(list(obj)))
        elif isinstance(obj, np.ndarray):
            return ArrayMemoryMap.from_numpy_array(obj)
        elif ObjectMemoryMapFactory.has_proxy(obj):
            return ObjectMemoryMapFactory.from_object(obj)
        else:
            return ArrayMemoryMap.from_numpy_array(np.asarray(obj))

    @classmethod
    def create_mapped_attributes_memory_maps(
        cls, obj: Any
    ) -> Tuple[Dict[str, Optional[Union[MemoryMap, List[MemoryMap]]]], int]:
        """Create memory maps for all mapped attributes using the values of an object.

        :param obj: The object to create memory maps for.
        :return: A map of attribute name to memory map and the total size of all memory maps.
        :raises AttributeError: If a mapped attribute is not present in the given object.
        """
        size = 0
        attribute_dict = {}

        for attribute, _ in cls.mapped_attributes:
            attribute_value = getattr(obj, attribute)
            if attribute_value is None:
                attribute_dict[attribute] = None
            elif isinstance(attribute_value, list):
                attribute_maps = [cls.get_memory_map(v) for v in attribute_value]
                attribute_dict[attribute] = attribute_maps
                size += sum(attr.byte_size for attr in attribute_maps)
            else:
                attribute_map = cls.get_memory_map(getattr(obj, attribute))
                attribute_dict[attribute] = attribute_map
                size += attribute_map.byte_size

        return attribute_dict, size

    @staticmethod
    def _write_attribute(
        write_buf: memoryview,
        write_idx: int,
        attribute_map: Union[ArrayMemoryMap, ObjectMemoryMap],
        geometry_attribute: Any,
    ) -> int:
        """Write the given geometry attribute to the given buffer using the attribute memory map.

        This works for object memory maps, sets and iterables that can be converted to numpy arrays.

        :param write_buf: The buffer to write to.
        :param write_idx: The byte index to start writing at.
        :param attribute_map: The memory map of the attribute.
        :param geometry_attribute: The attribute to write.
        :return: The byte index after writing.
        """
        if isinstance(attribute_map, ObjectMemoryMap):
            return attribute_map.write_object(write_buf, write_idx, geometry_attribute)
        else:
            buf = np.ndarray(
                attribute_map.shape,
                dtype=attribute_map.dtype,
                buffer=write_buf[write_idx : write_idx + attribute_map.byte_size],
            )

            if isinstance(geometry_attribute, set):
                buf[:] = np.asarray(list(geometry_attribute))[:]
            else:
                buf[:] = np.asarray(geometry_attribute)[:]

            return write_idx + attribute_map.byte_size

    def write_mapped_attributes(
        self, input_obj: object, write_buf: memoryview, write_idx: int
    ) -> int:
        """Write the mapped attributes to a buffer using the data from the given object.

        :param input_obj: The object to write the data from to a buffer.
        :param write_buf: The buffer to write to.
        :param write_idx: The byte index to start writing at.
        :return: The new byte index to start writing new data at.
        :raises AttributeError: If a mapped attribute is not present in the given object.
        """
        for attribute, _ in self.mapped_attributes:
            attribute_map = getattr(self, attribute)
            if attribute_map is None:
                continue

            if isinstance(attribute_map, list):
                if len(attribute_map) == 0:
                    continue

                geometry_attrs = getattr(input_obj, attribute)
                for i, attr in enumerate(attribute_map):
                    write_idx = self._write_attribute(
                        write_buf, write_idx, attr, geometry_attrs[i]
                    )
            else:
                if attribute_map.byte_size == 0:
                    continue

                write_idx = self._write_attribute(
                    write_buf, write_idx, attribute_map, getattr(input_obj, attribute)
                )
        return write_idx

    @staticmethod
    def _read_attribute(
        attribute_map: Union[ArrayMemoryMap, ObjectMemoryMap],
        attribute_type: Type,
        read_buf: memoryview,
        read_idx: int,
    ) -> Tuple[Any, int]:
        """Read the given attribute map from the read buffer starting from the byte index.

        :param attribute_map: The memory map of the attribute to read from the buffer.
        :param attribute_type: The type mapped out by the memory map.
        :param read_buf: The buffer to read from.
        :param read_idx: The byte index to start reading at.
        :return: The data read from the buffer as an instance of `attribute_type` and the new read byte index.
        """
        if isinstance(attribute_map, ObjectMemoryMap):
            attribute_value, read_idx = attribute_map.read_object(read_buf, read_idx)
        else:
            buf = np.ndarray(
                attribute_map.shape,
                dtype=attribute_map.dtype,
                buffer=read_buf[read_idx : read_idx + attribute_map.byte_size],
            )
            if attribute_type == np.ndarray:
                attribute_value = buf
            else:
                attribute_value = attribute_type(buf)
            read_idx += attribute_map.byte_size
        return attribute_value, read_idx

    def read_mapped_attributes_to_object(
        self, output_obj: object, read_buf: memoryview, read_idx: int
    ) -> int:
        """Read the mapped attributes from a buffer into the given object.

        :param output_obj: The object to read the data in.
        :param read_buf: The buffer to read from.
        :param read_idx: The byte index to start reading at.
        :return: The new byte index to start reading new data at.
        :raises AttributeError: If a mapped attribute is not present in the given object.
        """
        for attribute, attribute_type in self.mapped_attributes:
            attribute_map = getattr(self, attribute)
            if attribute_map is None:
                setattr(output_obj, attribute, None)
            elif isinstance(attribute_map, list):
                if len(attribute_map) == 0:
                    continue
                attrs = []
                for i, attr in enumerate(attribute_map):
                    attr_val, read_idx = self._read_attribute(
                        attr, attribute_type, read_buf, read_idx
                    )
                    attrs.append(attr_val)
                setattr(output_obj, attribute, attrs)
            else:
                if attribute_map.byte_size == 0:
                    continue
                attr_val, read_idx = self._read_attribute(
                    attribute_map, attribute_type, read_buf, read_idx
                )
                setattr(output_obj, attribute, attr_val)
        return read_idx

    @classmethod
    def from_object(cls, obj: T) -> ObjectMemoryMap[T]:
        """Create a new memory map for the given object.

        :param obj: The object to create the memory map from.
        """
        ...

    def write_object(
        self,
        write_buf: memoryview,
        write_idx: int,
        obj: T,
    ) -> int:
        """Write the given object to the shared memory using the memory map.

        :param write_buf: The memoryview to write to.
        :param write_idx: The index to start writing at.
        :param obj: The object to write.
        :return: The new write index.
        """
        ...

    def read_object(
        self,
        read_buf: memoryview,
        read_idx: int,
    ) -> Tuple[T, int]:
        """Read an object from the shared memory using the memory map.

        :param read_buf: The memoryview to read from.
        :param read_idx: The index to start reading at.
        :return: The object and the new read index.
        """
        ...


@dataclass(
    slots=True,
    frozen=True,
)
class ArrayMemoryMap(MemoryMap):
    """A memory map for a numpy array in shared memory."""

    shape: Tuple
    """Shape of the underlying array."""

    dtype: str
    """Datatype of the underlying data as a string."""

    @classmethod
    def from_numpy_array(cls, array: npt.NDArray) -> Self:
        """Create a new memory map for the given numpy array."""
        return cls(
            shape=array.shape,
            dtype=str(array.dtype),
            byte_size=array.size * array.dtype.itemsize,
        )


G = TypeVar("G", bound=o3d.geometry.Geometry3D)


@dataclass(slots=True, frozen=True, kw_only=True)
class Geometry3DMemoryMap(ObjectMemoryMap[G], Generic[G]):
    """A memory map for a geometry in shared memory."""

    name: str
    """Name of the underlying geometry."""

    type: Type
    """Type of the underlying geometry."""

    material: Optional[ObjectMemoryMap] = None
    """Material property of the o3d geometry."""

    group: Optional[str] = None
    """Group property of the o3d geometry."""

    time: Optional[float] = None
    """Time property of the o3d geometry."""

    is_visible: Optional[bool] = None
    """IsVisible property of the o3d geometry."""

    @classmethod
    def from_geometry(
        cls,
        name: str,
        geometry: o3d.geometry.Geometry3D,
        material: Optional[o3d.visualization.rendering.MaterialRecord] = None,
        group: Optional[str] = None,
        time: Optional[float] = None,
        is_visible: Optional[bool] = None,
    ) -> Self:
        """Create a new memory memory map for the given geometry.

        :param name: The o3d name of the geometry.
        :param geometry: The o3d geometry to create a memory map for.
        :param material: The o3d material property of the geometry.
        :param group: The o3d group property of the geometry.
        :param time: The o3d time property of the geometry.
        :param is_visible: The o3d is_visible property of the geometry.
        """
        attribute_dict, size = cls.create_mapped_attributes_memory_maps(geometry)

        if material is not None:
            material_map = ObjectMemoryMapFactory.from_object(material)
            size += material_map.byte_size
        else:
            material_map = None

        return cls(
            name=name,
            type=type(geometry),
            material=material_map,
            group=group,
            time=time,
            is_visible=is_visible,
            byte_size=size,
            **attribute_dict,
        )

    @classmethod
    def from_geometry_dict(cls, geometry: Dict[str, Any]) -> Geometry3DMemoryMap:
        """Create a new memory map from a geometry dictionary.

        :param geometry: The geometry dictionary to create a memory map from.
        :return: The memory map created from the geometry dictionary.
        """
        instance = cls.from_geometry(**geometry)
        return instance

    def read_geometry_dict(
        self, read_buf: memoryview, read_idx: int
    ) -> Tuple[Dict, int]:
        """Create an open3d geometry dict from the memory map.

        :param read_buf: The memory buffer to read from.
        :param read_idx: The byte index to start reading from.
        :return: The geometry dictionary and the byte index after reading.
        """
        geometry, read_idx = self.read_geometry(read_buf, read_idx)

        geometry_dict: Dict[str, Any] = {
            "name": self.name,
            "geometry": geometry,
        }

        if self.material is not None:
            geometry_dict["material"], read_idx = self.material.read_object(
                read_buf, read_idx
            )
        if self.group is not None:
            geometry_dict["group"] = self.group
        if self.time is not None:
            geometry_dict["time"] = self.time
        if self.is_visible is not None:
            geometry_dict["is_visible"] = self.is_visible

        return geometry_dict, read_idx

    def write_geometry(
        self,
        write_buf: memoryview,
        write_idx: int,
        geometry: Union[o3d.geometry.Geometry3D, Dict],
    ) -> int:
        """Write the given geometry to the shared memory using the memory map.

        :param write_buf: The shared memory to write to.
        :param write_idx: The byte index to start writing at.
        :param geometry: The geometry to write.
        :return: The byte index after writing.
        """

        if isinstance(geometry, dict):
            write_idx = self.write_mapped_attributes(
                geometry["geometry"], write_buf, write_idx
            )

            if self.material is not None:
                write_idx = self.material.write_object(
                    write_buf, write_idx, geometry["material"]
                )
        else:
            write_idx = self.write_mapped_attributes(geometry, write_buf, write_idx)

        return write_idx

    def read_geometry(
        self, read_buf: memoryview, read_idx: int
    ) -> Tuple[o3d.geometry.PointCloud, int]:
        """Read the geometry from the shared memory using the memory map.

        :param read_buf: The shared memory to read from.
        :param read_idx: The byte index to start reading from.
        :return: The geometry read from the shared memory and the byte index after reading.
        """
        geometry = self.type()
        read_idx = self.read_mapped_attributes_to_object(geometry, read_buf, read_idx)
        return geometry, read_idx


@dataclass(
    slots=True,
    frozen=True,
)
class PointCloudMemoryMap(Geometry3DMemoryMap[o3d.geometry.PointCloud]):
    points: ArrayMemoryMap
    """Memory map of the point clouds points."""

    normals: ArrayMemoryMap
    """Memory map of the point clouds point normals."""

    colors: ArrayMemoryMap
    """Memory map of the point clouds point colors."""

    covariances: ArrayMemoryMap
    """Memory map of the point clouds point covariances."""

    mapped_attributes = [
        ("points", o3d.utility.Vector3dVector),
        ("colors", o3d.utility.Vector3dVector),
        ("normals", o3d.utility.Vector3dVector),
        ("covariances", o3d.utility.Matrix3dVector),
    ]


@dataclass(slots=True, frozen=True)
class LineSetMemoryMap(Geometry3DMemoryMap[o3d.geometry.LineSet]):
    colors: ArrayMemoryMap
    """Memory map of the line set colors."""

    lines: ArrayMemoryMap
    """Memory map of the line set lines."""

    points: ArrayMemoryMap
    """Memory map of the line set points."""

    mapped_attributes = [
        ("colors", o3d.utility.Vector3dVector),
        ("lines", o3d.utility.Vector2iVector),
        ("points", o3d.utility.Vector3dVector),
    ]


@dataclass(
    slots=True,
    frozen=True,
)
class MeshBaseMemoryMap(Geometry3DMemoryMap[o3d.geometry.MeshBase]):
    vertices: ArrayMemoryMap
    """Memory map of the mesh vertices."""

    vertex_normals: ArrayMemoryMap
    """Memory map of the vertex normals."""

    vertex_colors: ArrayMemoryMap
    """Memory map of the vertex colors."""

    mapped_attributes = [
        ("vertices", o3d.utility.Vector3dVector),
        ("vertex_normals", o3d.utility.Vector3dVector),
        ("vertex_colors", o3d.utility.Vector3dVector),
    ]


@dataclass(
    slots=True,
    frozen=True,
)
class TriangleMeshMemoryMap(Geometry3DMemoryMap[o3d.geometry.TriangleMesh]):
    vertices: ArrayMemoryMap
    """Memory map of the mesh vertices."""

    vertex_normals: ArrayMemoryMap
    """Memory map of the vertex normals."""

    vertex_colors: ArrayMemoryMap
    """Memory map of the vertex colors."""

    triangles: ArrayMemoryMap
    """Memory map of the mesh triangles."""

    triangle_normals: ArrayMemoryMap
    """Memory map of the mesh triangle normals."""

    triangle_uvs: ArrayMemoryMap
    """Memory map of the mesh triangle uvs."""

    triangle_material_ids: ArrayMemoryMap
    """Memory map of the mesh triangle material ids."""

    textures: List[ArrayMemoryMap]
    """Memory map of the mesh textures."""

    adjacency_list: List[ArrayMemoryMap]
    """Memory map of the mesh adjacency list."""

    mapped_attributes = [
        ("vertices", o3d.utility.Vector3dVector),
        ("vertex_normals", o3d.utility.Vector3dVector),
        ("vertex_colors", o3d.utility.Vector3dVector),
        ("triangles", o3d.utility.Vector3iVector),
        ("triangle_normals", o3d.utility.Vector3dVector),
        ("triangle_uvs", o3d.utility.Vector2dVector),
        ("triangle_material_ids", o3d.utility.IntVector),
        ("textures", o3d.geometry.Image),
        ("adjacency_list", set),
    ]


@dataclass(
    slots=True,
    frozen=True,
)
class OrientedBoundingBoxMemoryMap(
    Geometry3DMemoryMap[o3d.geometry.OrientedBoundingBox]
):
    center: ArrayMemoryMap
    """Memory map of the oriented bounding box center."""

    color: ArrayMemoryMap
    """Memory map of the oriented bounding box color."""

    extent: ArrayMemoryMap
    """Memory map of the oriented bounding box extent."""

    R: ArrayMemoryMap
    """Memory map of the oriented bounding box extent."""

    mapped_attributes = [
        ("center", np.ndarray),
        ("color", np.ndarray),
        ("extent", np.ndarray),
        ("R", np.ndarray),
    ]


@dataclass(
    slots=True,
    frozen=True,
)
class AxisAlignedBoundingBoxMemoryMap(
    Geometry3DMemoryMap[o3d.geometry.AxisAlignedBoundingBox]
):
    color: ArrayMemoryMap
    """Memory map of the axis aligned bounding box color."""

    max_bound: ArrayMemoryMap
    """Memory map of the axis aligned bounding box maximum bound."""

    min_bound: ArrayMemoryMap
    """Memory map of the axis aligned bounding box maximum bound."""

    mapped_attributes = [
        ("color", np.ndarray),
        ("max_bound", np.ndarray),
        ("min_bound", np.ndarray),
    ]


@dataclass(slots=True, frozen=True)
class TetraMeshMemoryMap(Geometry3DMemoryMap[o3d.geometry.TetraMesh]):
    tetras: ArrayMemoryMap
    """Memory map of the tetra mesh tetras."""

    vertex_colors: ArrayMemoryMap
    """Memory map of the tetra mesh vertex colors."""

    vertex_normals: ArrayMemoryMap
    """Memory map of the tetra mesh vertex normals."""

    vertices: ArrayMemoryMap
    """Memory map of the tetra mesh vertices."""

    mapped_attributes = [
        ("tetras", o3d.utility.Vector4iVector),
        ("vertex_colors", o3d.utility.Vector3dVector),
        ("vertex_normals", o3d.utility.Vector3dVector),
        ("vertices", o3d.utility.Vector3dVector),
    ]


@dataclass(slots=True, frozen=True)
class MaterialRecordMemoryMap(
    ObjectMemoryMap[o3d.visualization.rendering.MaterialRecord]
):
    absorption_color: ArrayMemoryMap
    absorption_distance: float
    albedo_img: Optional[ArrayMemoryMap]
    anisotropy_img: Optional[ArrayMemoryMap]
    ao_img: Optional[ArrayMemoryMap]
    ao_rough_metal_img: Optional[ArrayMemoryMap]
    aspect_ratio: float
    base_anisotropy: float
    base_clearcoat: float
    base_clearcoat_roughness: float
    base_color: ArrayMemoryMap
    base_metallic: float
    base_reflectance: float
    base_roughness: float
    clearcoat_img: Optional[ArrayMemoryMap]
    clearcoat_roughness_img: Optional[ArrayMemoryMap]
    emissive_color: ArrayMemoryMap
    # TODO: Map these out
    # generic_imgs: dict[str, open3d.cuda.pybind.geometry.Image]
    # generic_params: dict[str, numpy.ndarray[numpy.float32[4, 1]]]
    # gradient: Gradient
    ground_plane_axis: float
    has_alpha: bool
    metallic_img: Optional[ArrayMemoryMap]
    normal_img: Optional[ArrayMemoryMap]
    point_size: float
    reflectance_img: Optional[ArrayMemoryMap]
    roughness_img: Optional[ArrayMemoryMap]
    sRGB_color: bool
    scalar_max: float
    scalar_min: float
    shader: str
    thickness: float
    transmission: float

    mapped_attributes = [
        ("absorption_color", np.ndarray),
        ("albedo_img", o3d.geometry.Image),
        ("anisotropy_img", o3d.geometry.Image),
        ("ao_img", o3d.geometry.Image),
        ("ao_rough_metal_img", o3d.geometry.Image),
        ("base_color", np.ndarray),
        ("clearcoat_img", o3d.geometry.Image),
        ("clearcoat_roughness_img", o3d.geometry.Image),
        ("emissive_color", np.ndarray),
        # ("generic_imgs", dict),
        # ("generic_params", dict),
        # ("gradient", Gradient)
        ("metallic_img", o3d.geometry.Image),
        ("normal_img", o3d.geometry.Image),
        ("reflectance_img", o3d.geometry.Image),
        ("roughness_img", o3d.geometry.Image),
    ]

    @override
    @classmethod
    def from_object(cls, obj: o3d.visualization.rendering.MaterialRecord) -> Self:
        attribute_dict, size = cls.create_mapped_attributes_memory_maps(obj)
        return cls(
            byte_size=size,
            absorption_distance=obj.absorption_distance,
            aspect_ratio=obj.aspect_ratio,
            base_anisotropy=obj.base_anisotropy,
            base_clearcoat=obj.base_clearcoat,
            base_clearcoat_roughness=obj.base_clearcoat_roughness,
            base_metallic=obj.base_metallic,
            base_reflectance=obj.base_reflectance,
            base_roughness=obj.base_roughness,
            ground_plane_axis=obj.ground_plane_axis,
            has_alpha=obj.has_alpha,
            point_size=obj.point_size,
            sRGB_color=obj.sRGB_color,
            scalar_max=obj.scalar_max,
            scalar_min=obj.scalar_min,
            shader=obj.shader,
            thickness=obj.thickness,
            transmission=obj.transmission,
            **attribute_dict,
        )

    @override
    def write_object(
        self,
        write_buf: memoryview,
        write_idx: int,
        obj: o3d.visualization.rendering.MaterialRecord,
    ) -> int:
        """Write the given geometry to the shared memory using the memory map.

        :param write_buf: The memory buffer to write to.
        :param write_idx: The byte index to start writing at.
        :param obj: The geometry to write.
        :return: The byte index after writing.
        """
        return self.write_mapped_attributes(obj, write_buf, write_idx)

    @override
    def read_object(
        self,
        read_buf: memoryview,
        read_idx: int,
    ) -> Tuple[o3d.visualization.rendering.MaterialRecord, int]:
        """Read the geometry from the shared memory using the memory map.

        :param read_buf: The memory buffer to read from.
        :param read_idx: The byte index to start reading from.
        :return: The geometry read from the shared memory and the byte index after reading.
        """
        geometry = o3d.visualization.rendering.MaterialRecord()
        read_idx = self.read_mapped_attributes_to_object(geometry, read_buf, read_idx)

        for attribute in [
            "absorption_distance",
            "aspect_ratio",
            "base_anisotropy",
            "base_clearcoat",
            "base_clearcoat_roughness",
            "base_metallic",
            "base_reflectance",
            "base_roughness",
            "ground_plane_axis",
            "has_alpha",
            "point_size",
            "sRGB_color",
            "scalar_max",
            "scalar_min",
            "shader",
            "thickness",
            "transmission",
        ]:
            setattr(geometry, attribute, getattr(self, attribute))
        return geometry, read_idx


@dataclass(slots=True, frozen=True)
class HalfEdgeMemoryMap(ObjectMemoryMap[o3d.geometry.HalfEdge]):
    data: ArrayMemoryMap
    """Memory map containing next, triangle_index, twin and vertex_indices."""

    @override
    @classmethod
    def from_object(cls, obj: o3d.geometry.HalfEdge) -> Self:
        """Create a HalfEdgeMemoryMap from a HalfEdge object.

        :param obj: The HalfEdge object to create a HalfEdgeMemoryMap from.
        :return: The HalfEdgeMemoryMap created from the HalfEdge object.
        """
        data_list = [obj.next, obj.triangle_index, obj.twin]
        data_list.extend(obj.vertex_indices.tolist())

        data = ArrayMemoryMap.from_numpy_array(
            np.array(
                data_list,
                dtype=np.int32,
            )
        )
        size = data.byte_size
        return cls(byte_size=size, data=data)

    @override
    def write_object(
        self,
        write_buf: memoryview,
        write_idx: int,
        obj: o3d.geometry.HalfEdge,
    ) -> int:
        """Write the given HalfEdge object to the given buffer using the memory map.

        :param write_buf: The buffer to write to.
        :param write_idx: The byte index to start writing at.
        :param obj: The HalfEdge object to write.
        :return: The byte index after writing.
        """
        buf = np.ndarray(
            self.data.shape,
            dtype=self.data.dtype,
            buffer=write_buf[write_idx : write_idx + self.data.byte_size],
        )
        data_list = [obj.next, obj.triangle_index, obj.twin]
        data_list.extend(obj.vertex_indices.tolist())
        buf[:] = np.array(
            data_list,
            dtype=np.int32,
        )
        return write_idx + self.byte_size

    @override
    def read_object(
        self,
        read_buf: memoryview,
        read_idx: int,
    ) -> Tuple[o3d.geometry.HalfEdge, int]:
        """Read the given buffer from the given byte index as a HalfEdge object using the memory map.

        :param read_buf: The buffer to read from.
        :param read_idx: The byte index to start reading from.
        :return: The HalfEdge object read from the buffer and the byte index after reading.
        """
        buf = np.ndarray(
            self.data.shape,
            dtype=self.data.dtype,
            buffer=read_buf[read_idx : read_idx + self.data.byte_size],
        )

        half_edge = o3d.geometry.HalfEdge()
        half_edge.next = buf[0]
        half_edge.triangle_index = buf[1]
        half_edge.twin = buf[2]
        half_edge.vertex_indices = buf[3:].tolist()

        return half_edge, read_idx + self.byte_size


@dataclass(slots=True, frozen=True)
class HalfEdgeTriangleMeshMemoryMap(
    Geometry3DMemoryMap[o3d.geometry.HalfEdgeTriangleMesh]
):
    half_edges: List[HalfEdgeMemoryMap]
    """Memory map of the half edge mesh half edges."""

    ordered_half_edge_from_vertex: List[ArrayMemoryMap]
    """Memory map of the half edge mesh ordered half edge from vertex."""

    triangle_normals: ArrayMemoryMap
    """Memory map of the half edge mesh triangle normals."""

    triangles: ArrayMemoryMap
    """Memory map of the half edge mesh triangles."""

    vertex_colors: ArrayMemoryMap
    """Memory map of the half edge mesh vertex colors."""

    vertex_normals: ArrayMemoryMap
    """Memory map of the half edge mesh vertex normals."""

    vertices: ArrayMemoryMap
    """Memory map of the half edge mesh vertices."""

    mapped_attributes = [
        ("half_edges", o3d.geometry.HalfEdge),
        ("ordered_half_edge_from_vertex", o3d.utility.IntVector),
        ("triangle_normals", o3d.utility.Vector3dVector),
        ("triangles", o3d.utility.Vector3iVector),
        ("vertex_colors", o3d.utility.Vector3dVector),
        ("vertex_normals", o3d.utility.Vector3dVector),
        ("vertices", o3d.utility.Vector3dVector),
    ]


# @dataclass(
#     slots=True,
#     frozen=True,
# )
# class VoxelGrid3DMemoryMap(Geometry3DMemoryMap):
#     # TODO: No actual data accessible in the VoxelGrid how to transfer?
#     origin: ArrayMemoryMap
#     """Memory map of the voxel grid origin."""
#
#     voxel_size: ArrayMemoryMap
#     """Memory map of the voxel grid voxel size."""
#
#     mapped_attributes = [
#         ("origin", o3d.utility.Vector3dVector),
#         ("voxel_size", o3d.utility.Vector3dVector),
#     ]
#
#
# @dataclass(slots=True, frozen=True)
# class Octree3DMemoryMap(Geometry3DMemoryMap):
#     # TODO: No actual data accessible in the Octree how to transfer?
#     max_depth: int
#     """Maximum depth of the octree."""
#
#     origin: ArrayMemoryMap
#     """Memory map of the octree origin."""
#
#     root_node: o3d.geometry.OctreeNode
#     """Memory map of the octree root node."""
#
#     size: float
#     """Memory map of the octree size."""


class ObjectMemoryMapFactory:
    """A factory class for creating geometry memory maps from open3d geometry objects."""

    proxies: Dict[Type, Type[ObjectMemoryMap]] = {
        o3d.geometry.HalfEdge: HalfEdgeMemoryMap,
        o3d.visualization.rendering.MaterialRecord: MaterialRecordMemoryMap,
    }
    """Map of open3d geometry types to their corresponding memory map types."""

    @classmethod
    def has_proxy(cls, obj: Any) -> bool:
        """Check whether the factory has a proxy for the given object class.

        :param obj: The object to check.
        :return: True if there is a proxy available for the given object class, False otherwise.
        """
        return type(obj) in cls.proxies

    @classmethod
    def from_object(cls, obj: Any) -> ObjectMemoryMap:
        """Create a geometry proxy from a Geometry3D object.

        :param obj: The object to create a proxy for.
        :raises KeyError: If the factory has no proxy for the geometry type.
        """
        return cls.proxies[type(obj)].from_object(obj)


class Geometry3DMemoryMapFactory:
    """A factory class for creating geometry memory maps from open3d geometry3d objects."""

    proxies: Dict[Type, Type[Geometry3DMemoryMap]] = {
        o3d.geometry.PointCloud: PointCloudMemoryMap,
        o3d.geometry.MeshBase: MeshBaseMemoryMap,
        o3d.geometry.TetraMesh: TetraMeshMemoryMap,
        o3d.geometry.TriangleMesh: TriangleMeshMemoryMap,
        o3d.geometry.HalfEdgeTriangleMesh: HalfEdgeTriangleMeshMemoryMap,
        o3d.geometry.OrientedBoundingBox: OrientedBoundingBoxMemoryMap,
        o3d.geometry.AxisAlignedBoundingBox: AxisAlignedBoundingBoxMemoryMap,
        o3d.geometry.LineSet: LineSetMemoryMap,
    }
    """Map of open3d geometry types to their corresponding memory map types."""

    @classmethod
    def has_proxy(cls, obj: Any) -> bool:
        """Check whether the factory has a proxy for the given object class.

        :param obj: The object to check.
        :return: True if there is a proxy available for the given object class, False otherwise.
        """
        return type(obj) in cls.proxies

    @classmethod
    def from_geometry(
        cls, name: str, geometry: o3d.geometry.Geometry3D
    ) -> Geometry3DMemoryMap:
        """Create a geometry proxy from a Geometry3D object.

        :param name: The name of the geometry used to add the geometry to the O3DVisualizer.
        :param geometry: The geometry object to create a proxy for.
        :raises KeyError: If the factory has no proxy for the geometry type.
        """
        return cls.proxies[type(geometry)].from_geometry(name, geometry)

    @classmethod
    def from_geometry_dict(cls, geometry: Dict) -> Geometry3DMemoryMap:
        """Create a geometry proxy from a Geometry3D object.

        :param geometry: The geometry dictionary to create a proxy for.
        :return: The geometry proxy created from the geometry dictionary.
        :raises KeyError: If the factory has no proxy for the geometry type.
        """
        return cls.proxies[type(geometry["geometry"])].from_geometry(**geometry)


@dataclass(slots=True, frozen=True)
class MemoryMapTransport(object):
    """A message containing geometry data for visualization."""

    shm_name: str
    """The shared memory to read from."""

    memory_maps: List[Geometry3DMemoryMap] = field(default_factory=list)
    """The memory mappings for the geometries."""


@dataclass(slots=True)
class SharedMemoryManager(object):
    """A manager for geometries in shared memory."""

    _shm: shared_memory.SharedMemory
    """The shared memory managed by the instance."""

    memory_maps: List[Geometry3DMemoryMap] = field(default_factory=list)
    """A list of all memory maps managed by this object."""

    memory_maps_size: int = 0
    """The total size of all memory maps managed by this object."""

    write_cursor: int = 0
    """The current end byte index of the shared memory (sum of all memory map sizes)."""

    @classmethod
    def with_shm(cls, size: int) -> Self:
        """Create a SharedMemoryManager with a newly created shared memory of the given size.

        :param size: The size of the shared memory to create.
        :return: A SharedMemoryManager instance with the newly created shared memory.
        """
        return cls(_shm=shared_memory.SharedMemory(create=True, size=size))

    @classmethod
    def for_shm(cls, name: str) -> Self:
        """Create a SharedMemoryManager for an existing shared memory with the given name.

        :param name: The name of the shared memory to use.
        :return: A SharedMemoryManager instance for the existing shared memory.
        """
        return cls(_shm=shared_memory.SharedMemory(name=name))

    @property
    def name(self) -> str:
        """The name of the shared memory managed by the instance."""
        return self._shm.name

    def append(self, memory_map: Geometry3DMemoryMap) -> None:
        """Add a memory map to the shared memory manager.

        :param memory_map: MemoryMap to add to the shared memory manager.
        :return: The byte index to start writing to for the appended memory map
        """
        self.memory_maps.append(memory_map)
        self.memory_maps_size += memory_map.byte_size
        if self.memory_maps_size > self._shm.size:
            raise RuntimeError(
                "The shared memory is too small to store all memory maps."
            )

    def extend(self, memory_maps: List[Geometry3DMemoryMap]) -> None:
        """Add a list of memory maps to the shared memory manager.

        :return: The byte indices to start writing to for each of the appended memory maps
        """
        self.memory_maps.extend(memory_maps)

        for memory_map in memory_maps:
            self.memory_maps_size += memory_map.byte_size
        if self.memory_maps_size > self._shm.size:
            raise RuntimeError(
                "The shared memory is too small to store all memory maps."
            )

    def write_geometry(
        self,
        geometry_dict: Union[Dict, o3d.geometry.Geometry3D],
        memory_map: Geometry3DMemoryMap,
    ) -> None:
        """Write a geometry to the shared memory using the given memory map.

        :param geometry_dict: The geometry dictionary to write to the shared memory.
        :param memory_map: The memory map to use for writing the geometry.
        """
        write_buf = self._shm.buf
        if write_buf is None:
            raise RuntimeError("The shared memory buffer is None")
        if memory_map not in self.memory_maps:
            self.append(memory_map)
        memory_map.write_geometry(write_buf, self.write_cursor, geometry_dict)
        self.write_cursor += memory_map.byte_size

    def read_geometries(self) -> Iterator[Dict]:
        """Read all memory maps from the shared memory manager.

        :return: An iterator over the geometries read from the shared memory.
        """
        read_buf = self._shm.buf
        if read_buf is None:
            raise RuntimeError("The shared memory buffer is None")
        read_cursor = 0
        for memory_map in self.memory_maps:
            geometry, _ = memory_map.read_geometry_dict(read_buf, read_cursor)
            yield geometry
            read_cursor += memory_map.byte_size

    def reset(self) -> None:
        """Reset the shared memory manager to its initial state."""
        self.write_cursor = 0
        self.memory_maps_size = 0
        self.memory_maps.clear()

    def close(self) -> None:
        """Close the underlying shared memory."""
        self._shm.close()

    def unlink_and_close(self) -> None:
        """Unlink and close the underlying shared memory."""
        self._shm.unlink()
        self.close()


class MultiprocessedViewer3DClient(object):
    """A client for the MultiprocessedViewer3D server."""

    def __init__(self, title: str, cmd_conn: Connection, tick_return: mp.Event) -> None:
        self.rk_logger: logging.Logger = logging.getLogger(PACKAGE_NAME)
        """Logger instance"""

        self.viewer3d = Viewer3D(title)
        """Viewer3D instance for visualization."""

        self.cmd_conn = cmd_conn
        """Communication connection for sending and receiving commands from the main process."""

        self.tick_return = tick_return
        """Event indicating that the o3d visualizer shut down."""

        self.name_to_shm_manager: Dict[str, SharedMemoryManager] = {}
        """Mapping of shared memory names to shared memory instances."""

        self.visualized_geometries: List[str] = []
        """List of the names of the currently visualized geometries"""

        self.geometries_lock: Lock = Lock()
        """Lock for synchronizing access to the geometries list."""

        self.geometries: List[Union[Dict, o3d.geometry.Geometry3D]] = []
        """List of the geometries to use for updating the viewer."""

        self.receiver_thread = Thread(target=self.listen, daemon=True)
        """A thread for listening to commands from the main process."""

    def get_shm_manager(self, shm_name: str) -> SharedMemoryManager:
        """Get the current shared memory manager.

        :param shm_name: The name of the shared memory to get the manager for.
        :return: The shared memory manager for the given shared memory name.
        """
        if shm_name not in self.name_to_shm_manager:
            self.name_to_shm_manager[shm_name] = SharedMemoryManager.for_shm(shm_name)
        return self.name_to_shm_manager[shm_name]

    def run(self) -> None:
        """Run the visualization client."""
        self.receiver_thread.start()

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        self.viewer3d.main_vis.add_geometry("dummy", coordinate_frame)

        interval = 1.0 / 60.0
        last_tick = time.monotonic()

        try:
            while not self.tick_return.is_set():
                current_time = time.monotonic()
                elapsed = current_time - last_tick

                if elapsed >= interval:
                    # Force a proper redraw
                    self.viewer3d.main_vis.remove_geometry("dummy")
                    self.viewer3d.main_vis.add_geometry("dummy", coordinate_frame)
                    tick_return = self.viewer3d.tick()
                    if not tick_return:
                        self.rk_logger.debug("Visualizer indicates shutdown.")
                        self.tick_return.set()
                        break

                    last_tick = current_time
                else:
                    time.sleep(interval - elapsed)
        except KeyboardInterrupt:
            self.rk_logger.debug("Keyboard interrupt received.")
        finally:
            self.close()

    def close(self) -> None:
        """Close the visualization client and clean up resources."""
        self.rk_logger.debug("Shutting down...")
        if self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=1.0 / 10.0)
        for manager in self.name_to_shm_manager.values():
            self.rk_logger.debug(f"Closing shared memory {manager.name}")
            manager.close()
        self.rk_logger.debug("Shutdown complete.")

    def update_geometry(self) -> None:
        """Update the geometry in the viewer."""
        start_t = time.perf_counter()
        with self.geometries_lock:
            self.viewer3d.update_cloud(self.geometries)
        self.rk_logger.debug(
            f"Updated geometry in {time.perf_counter() - start_t:.4f}s"
        )

    def listen(self) -> None:
        """Listen for commands from the main process and handle them accordingly."""
        while not self.tick_return.is_set():
            try:
                cmd = self.cmd_conn.recv()
            except EOFError as e:
                self.rk_logger.error(
                    f"Pipe closed unexpectedly, shutting down visualizer: {e}"
                )
                self.tick_return.set()
                break

            if isinstance(cmd, MemoryMapTransport):
                self.rk_logger.debug(f"Received memory map: {cmd}")

                if self.tick_return.is_set():
                    break

                start_t = time.perf_counter()

                # Load into memory manager
                shm_manager = self.get_shm_manager(cmd.shm_name)
                shm_manager.reset()
                shm_manager.extend(cmd.memory_maps)

                # Reconstruct geometries
                with self.geometries_lock:
                    self.geometries = [
                        geometry for geometry in shm_manager.read_geometries()
                    ]

                o3d.visualization.gui.Application.instance.post_to_main_thread(
                    self.viewer3d.main_vis, self.update_geometry
                )

                self.rk_logger.debug(
                    f"Processed memory map in {time.perf_counter() - start_t:.4f}s"
                )


class MultiprocessedViewer3D(object):
    """A wrapper class for the Viewer3D class to run it in a separate process."""

    def __init__(self, title: str, shm_size: int = 5_000_000_000) -> None:
        """Initialize the 3D viewer.

        :param title: Window title for the viewer
        :param shm_size: Size of the shared memory to use for communication
        """

        self.rk_logger: logging.Logger = logging.getLogger(PACKAGE_NAME)
        """Logger instance"""

        self.buffer_count = 2
        """Number of buffers to use for communication."""

        self.buffer_write_cursor = 0
        """Index of the memory manager to use for writing."""

        self.buffer_read_cursor = 1
        """Index of the memory manager to use for reading."""

        self.tick_return = mp.Event()
        """Event indicating that the o3d visualizer shut down."""

        self.memory_manager = [
            SharedMemoryManager.with_shm(shm_size) for _ in range(self.buffer_count)
        ]
        """Manager instances for shared memory handling."""

        parent_cmd_conn, child_cmd_conn = mp.Pipe()

        self.parent_cmd_conn: Connection = parent_cmd_conn
        """Pipe connection for sending and receiving commands from the main process."""

        self.child_cmd_conn: Connection = child_cmd_conn
        """Pipe connection for sending and receiving commands on the visualizer process."""

        self.visualizer_process: mp.Process = mp.Process(
            target=self.run_visualizer,
            args=(title, self.child_cmd_conn, self.tick_return),
            name="robokudo_visualizer",
            daemon=True,
        )
        """A process running a viewer3d instance."""

        self.visualizer_process.start()

        atexit.register(self.close)

        self.rk_logger.debug(
            f"Started viewer process with PID {self.visualizer_process.pid}"
        )

    @staticmethod
    def run_visualizer(title: str, cmd_conn: Connection, tick_return: mp.Event) -> None:
        """Run the viewer3d instance in a separate process.

        :param title: Window title for the viewer.
        :param cmd_conn: Connection for sending and receiving commands from the main process.
        :param tick_return: Event indicating that the o3d visualizer shut down to the main process.
        """
        client = MultiprocessedViewer3DClient(title, cmd_conn, tick_return)
        client.run()

    @property
    def _read_manager(self) -> SharedMemoryManager:
        """The manager for the shared memory that is currently readable."""
        return self.memory_manager[self.buffer_read_cursor]

    @property
    def _write_manager(self) -> SharedMemoryManager:
        """The manager for the shared memory that is currently writeable."""
        return self.memory_manager[self.buffer_write_cursor]

    def _swap_buffers(self) -> Tuple[int, int]:
        """Rotate the read and write buffers.

        :return: The indices of the new buffers in format (read_buffer, write_buffer)
        """
        self.buffer_read_cursor = (self.buffer_read_cursor + 1) % self.buffer_count
        self.buffer_write_cursor = (self.buffer_write_cursor + 1) % self.buffer_count
        return self.buffer_read_cursor, self.buffer_write_cursor

    def tick(self) -> bool:
        """Dummy tick method indicating whether the visualizer has shut down.

        :returns: True if the visualizer has shut down, False otherwise.
        """
        return not self.tick_return.is_set()

    def update_cloud(
        self, geometries: Optional[Union[o3d.geometry.Geometry, Dict, List]]
    ) -> None:
        """Update the displayed geometries.

        This method updates the Open3D visualizer based on the outputs of the annotators.
        For the first update, it also sets up the camera and coordinate frame.

        :param geometries: Geometries to display. Can be:

        .. note::
            The dict format follows Open3D's draw() convention. See:
            https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/draw.py
        """
        if not self.visualizer_process.is_alive() or self.tick_return.is_set():
            return
        if geometries is None:
            return

        write_manager = self._write_manager

        # local method to add a single geometry. either based on the geometry being fully
        # defined with a dict or being a plain geometry object
        def add(
            g: Union[o3d.geometry.PointCloud, Dict, o3d.geometry.Geometry], n: int
        ) -> None:
            # Skip empty point clouds as they generate errors during the update
            if isinstance(g, o3d.geometry.PointCloud) and len(g.points) == 0:
                return

            try:
                if isinstance(g, dict):
                    geometry_dict = g
                else:
                    geometry_dict = {"name": "Object " + str(n), "geometry": g}
                memory_map = Geometry3DMemoryMapFactory.from_geometry_dict(
                    geometry_dict
                )
            except KeyError as e:
                self.rk_logger.warning(f"Could not create a memory map for {g}: {e}")
                return

            write_manager.write_geometry(geometry_dict, memory_map)

        write_manager.reset()

        n = 1
        if isinstance(geometries, list):
            for g in geometries:
                add(g, n)
                n += 1
        elif geometries is not None:
            add(geometries, n)

        transport = MemoryMapTransport(
            shm_name=write_manager.name,
            memory_maps=write_manager.memory_maps,
        )

        if self.visualizer_process.is_alive() and not self.tick_return.is_set():
            self.parent_cmd_conn.send(transport)

        self._swap_buffers()

    def close(self) -> None:
        """Clean up shared memory and process resources."""
        if self.visualizer_process.is_alive():
            self.visualizer_process.join(timeout=1)
            if self.visualizer_process.is_alive():
                self.visualizer_process.terminate()

        for manager in self.memory_manager:
            manager.unlink_and_close()

        self.parent_cmd_conn.close()
        self.child_cmd_conn.close()
