from ormatic.custom_types import TypeType
from sqlalchemy import Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, JSON, MetaData, String, Table
from sqlalchemy.orm import RelationshipProperty, registry, relationship
import pycram.datastructures.dataclasses
import pycram.datastructures.pose
import segmind.datastructures.events

metadata = MetaData()


t_BoundingBox = Table(
    'BoundingBox', metadata,
    Column('id', Integer, primary_key=True),
    Column('min_x', Float, nullable=False),
    Column('min_y', Float, nullable=False),
    Column('min_z', Float, nullable=False),
    Column('max_x', Float, nullable=False),
    Column('max_y', Float, nullable=False),
    Column('max_z', Float, nullable=False),
    Column('polymorphic_type', String(255))
)

t_Color = Table(
    'Color', metadata,
    Column('id', Integer, primary_key=True),
    Column('R', Float, nullable=False),
    Column('G', Float, nullable=False),
    Column('B', Float, nullable=False),
    Column('A', Float, nullable=False)
)

t_ContactPointsList = Table(
    'ContactPointsList', metadata,
    Column('id', Integer, primary_key=True),
    Column('polymorphic_type', String(255))
)

t_Event = Table(
    'Event', metadata,
    Column('id', Integer, primary_key=True),
    Column('timestamp', Float, nullable=False),
    Column('detector_thread_id', String(255)),
    Column('polymorphic_type', String(255))
)

t_FrozenWorldState = Table(
    'FrozenWorldState', metadata,
    Column('id', Integer, primary_key=True)
)

t_Header = Table(
    'Header', metadata,
    Column('id', Integer, primary_key=True),
    Column('frame_id', String(255), nullable=False),
    Column('stamp', DateTime, nullable=False),
    Column('sequence', Integer, nullable=False)
)

t_Quaternion = Table(
    'Quaternion', metadata,
    Column('id', Integer, primary_key=True),
    Column('x', Float, nullable=False),
    Column('y', Float, nullable=False),
    Column('z', Float, nullable=False),
    Column('w', Float, nullable=False)
)

t_Vector3 = Table(
    'Vector3', metadata,
    Column('id', Integer, primary_key=True),
    Column('x', Float, nullable=False),
    Column('y', Float, nullable=False),
    Column('z', Float, nullable=False),
    Column('polymorphic_type', String(255))
)

t_AxisAlignedBoundingBox = Table(
    'AxisAlignedBoundingBox', metadata,
    Column('id', ForeignKey('BoundingBox.id'), primary_key=True)
)

t_ClosestPointsList = Table(
    'ClosestPointsList', metadata,
    Column('id', ForeignKey('ContactPointsList.id'), primary_key=True)
)

t_EventWithTrackedObjects = Table(
    'EventWithTrackedObjects', metadata,
    Column('id', ForeignKey('Event.id'), primary_key=True)
)

t_LateralFriction = Table(
    'LateralFriction', metadata,
    Column('id', Integer, primary_key=True),
    Column('lateral_friction', Float, nullable=False),
    Column('lateral_friction_direction_id', ForeignKey('Vector3.id'), nullable=False)
)

t_Pose = Table(
    'Pose', metadata,
    Column('id', Integer, primary_key=True),
    Column('position_id', ForeignKey('Vector3.id'), nullable=False),
    Column('orientation_id', ForeignKey('Quaternion.id'), nullable=False),
    Column('polymorphic_type', String(255))
)

t_RotatedBoundingBox = Table(
    'RotatedBoundingBox', metadata,
    Column('id', ForeignKey('BoundingBox.id'), primary_key=True)
)

t_Vector3Stamped = Table(
    'Vector3Stamped', metadata,
    Column('id', ForeignKey('Vector3.id'), primary_key=True),
    Column('header_id', ForeignKey('Header.id'), nullable=False)
)

t_PoseStamped = Table(
    'PoseStamped', metadata,
    Column('id', Integer, primary_key=True),
    Column('pose_id', ForeignKey('Pose.id'), nullable=False),
    Column('header_id', ForeignKey('Header.id'), nullable=False),
    Column('polymorphic_type', String(255))
)

t_Transform = Table(
    'Transform', metadata,
    Column('id', ForeignKey('Pose.id'), primary_key=True)
)

t_FrozenBody = Table(
    'FrozenBody', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String(255), nullable=False),
    Column('concept', TypeType),
    Column('pose_id', ForeignKey('PoseStamped.id'), nullable=False),
    Column('is_moving', Boolean),
    Column('is_translating', Boolean),
    Column('is_rotating', Boolean),
    Column('velocity_id', ForeignKey('Vector3.id')),
    Column('bounding_box_id', ForeignKey('AxisAlignedBoundingBox.id')),
    Column('polymorphic_type', String(255))
)

t_GraspPose = Table(
    'GraspPose', metadata,
    Column('id', ForeignKey('PoseStamped.id'), primary_key=True),
    Column('arm', Enum(pycram.datastructures.enums.Arms), nullable=False)
)

t_TransformStamped = Table(
    'TransformStamped', metadata,
    Column('id', ForeignKey('PoseStamped.id'), primary_key=True),
    Column('pose_id', ForeignKey('Transform.id'), nullable=False),
    Column('child_frame_id', String(255), nullable=False)
)

t_ContactPoint = Table(
    'ContactPoint', metadata,
    Column('id', Integer, primary_key=True),
    Column('position_on_body_a_id', ForeignKey('Vector3.id')),
    Column('position_on_body_b_id', ForeignKey('Vector3.id')),
    Column('normal_on_body_b_id', ForeignKey('Vector3.id')),
    Column('distance', Float),
    Column('normal_force', Float),
    Column('lateral_friction_1_id', ForeignKey('LateralFriction.id')),
    Column('lateral_friction_2_id', ForeignKey('LateralFriction.id')),
    Column('body_a_frozen_cp_id', ForeignKey('FrozenBody.id'), nullable=False),
    Column('body_b_frozen_cp_id', ForeignKey('FrozenBody.id'), nullable=False),
    Column('contactpointslist_points_id', ForeignKey('ContactPointsList.id')),
    Column('polymorphic_type', String(255))
)

t_FrozenObject = Table(
    'FrozenObject', metadata,
    Column('id', ForeignKey('FrozenBody.id'), primary_key=True),
    Column('frozenworldstate_objects_id', ForeignKey('FrozenWorldState.id')),
    Column('path', String(255))
)

t_ClosestPoint = Table(
    'ClosestPoint', metadata,
    Column('id', ForeignKey('ContactPoint.id'), primary_key=True)
)

t_EventWithOneTrackedObject = Table(
    'EventWithOneTrackedObject', metadata,
    Column('id', ForeignKey('EventWithTrackedObjects.id'), primary_key=True),
    Column('tracked_object_frozen_cp_id', ForeignKey('FrozenObject.id')),
    Column('world_frozen_cp_id', ForeignKey('FrozenWorldState.id'))
)

t_EventWithTwoTrackedObjects = Table(
    'EventWithTwoTrackedObjects', metadata,
    Column('id', ForeignKey('EventWithTrackedObjects.id'), primary_key=True),
    Column('with_object_frozen_cp_id', ForeignKey('FrozenObject.id')),
    Column('tracked_object_frozen_cp_id', ForeignKey('FrozenObject.id')),
    Column('world_frozen_cp_id', ForeignKey('FrozenWorldState.id'))
)

t_FrozenJoint = Table(
    'FrozenJoint', metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String(255), nullable=False),
    Column('type', Enum(pycram.datastructures.enums.JointType), nullable=False),
    Column('children', JSON),
    Column('parent', String(255)),
    Column('state', Float, nullable=False),
    Column('frozenobject_joints_id', ForeignKey('FrozenObject.id'))
)

t_FrozenLink = Table(
    'FrozenLink', metadata,
    Column('id', ForeignKey('FrozenBody.id'), primary_key=True),
    Column('frozenobject_links_id', ForeignKey('FrozenObject.id'))
)

t_AbstractAgentObjectInteractionEvent = Table(
    'AbstractAgentObjectInteractionEvent', metadata,
    Column('id', ForeignKey('EventWithTwoTrackedObjects.id'), primary_key=True),
    Column('timestamp', Float),
    Column('end_timestamp', Float),
    Column('agent_frozen_cp_id', ForeignKey('FrozenObject.id'))
)

t_AbstractContactEvent = Table(
    'AbstractContactEvent', metadata,
    Column('id', ForeignKey('EventWithTwoTrackedObjects.id'), primary_key=True),
    Column('contact_points_id', ForeignKey('ContactPointsList.id'), nullable=False),
    Column('latest_contact_points_id', ForeignKey('ContactPointsList.id'), nullable=False),
    Column('bounding_box_id', ForeignKey('AxisAlignedBoundingBox.id'), nullable=False),
    Column('pose_id', ForeignKey('PoseStamped.id'), nullable=False),
    Column('with_object_bounding_box_id', ForeignKey('AxisAlignedBoundingBox.id')),
    Column('with_object_pose_id', ForeignKey('PoseStamped.id'))
)

t_DefaultEventWithTwoTrackedObjects = Table(
    'DefaultEventWithTwoTrackedObjects', metadata,
    Column('id', ForeignKey('EventWithTwoTrackedObjects.id'), primary_key=True)
)

t_MotionEvent = Table(
    'MotionEvent', metadata,
    Column('id', ForeignKey('EventWithOneTrackedObject.id'), primary_key=True),
    Column('start_pose_id', ForeignKey('PoseStamped.id'), nullable=False),
    Column('current_pose_id', ForeignKey('PoseStamped.id'), nullable=False)
)

t_NewObjectEvent = Table(
    'NewObjectEvent', metadata,
    Column('id', ForeignKey('EventWithOneTrackedObject.id'), primary_key=True)
)

t_VisualShape = Table(
    'VisualShape', metadata,
    Column('id', Integer, primary_key=True),
    Column('rgba_color_id', ForeignKey('Color.id'), nullable=False),
    Column('visual_frame_position_id', ForeignKey('Vector3.id'), nullable=False),
    Column('frozenlink_geometry_id', ForeignKey('FrozenLink.id')),
    Column('polymorphic_type', String(255))
)

t_AbstractAgentContact = Table(
    'AbstractAgentContact', metadata,
    Column('id', ForeignKey('AbstractContactEvent.id'), primary_key=True)
)

t_BoxVisualShape = Table(
    'BoxVisualShape', metadata,
    Column('id', ForeignKey('VisualShape.id'), primary_key=True),
    Column('half_extents_id', ForeignKey('Vector3.id'), nullable=False)
)

t_CapsuleVisualShape = Table(
    'CapsuleVisualShape', metadata,
    Column('id', ForeignKey('VisualShape.id'), primary_key=True),
    Column('radius', Float, nullable=False),
    Column('length', Float, nullable=False)
)

t_ContactEvent = Table(
    'ContactEvent', metadata,
    Column('id', ForeignKey('AbstractContactEvent.id'), primary_key=True)
)

t_ContainmentEvent = Table(
    'ContainmentEvent', metadata,
    Column('id', ForeignKey('DefaultEventWithTwoTrackedObjects.id'), primary_key=True)
)

t_LossOfContactEvent = Table(
    'LossOfContactEvent', metadata,
    Column('id', ForeignKey('AbstractContactEvent.id'), primary_key=True)
)

t_LossOfSupportEvent = Table(
    'LossOfSupportEvent', metadata,
    Column('id', ForeignKey('DefaultEventWithTwoTrackedObjects.id'), primary_key=True)
)

t_MeshVisualShape = Table(
    'MeshVisualShape', metadata,
    Column('id', ForeignKey('VisualShape.id'), primary_key=True),
    Column('scale_id', ForeignKey('Vector3.id'), nullable=False),
    Column('file_name', String(255), nullable=False)
)

t_PickUpEvent = Table(
    'PickUpEvent', metadata,
    Column('id', ForeignKey('AbstractAgentObjectInteractionEvent.id'), primary_key=True)
)

t_PlacingEvent = Table(
    'PlacingEvent', metadata,
    Column('id', ForeignKey('AbstractAgentObjectInteractionEvent.id'), primary_key=True),
    Column('placement_pose_id', ForeignKey('PoseStamped.id'))
)

t_PlaneVisualShape = Table(
    'PlaneVisualShape', metadata,
    Column('id', ForeignKey('VisualShape.id'), primary_key=True),
    Column('normal_id', ForeignKey('Vector3.id'), nullable=False)
)

t_RotationEvent = Table(
    'RotationEvent', metadata,
    Column('id', ForeignKey('MotionEvent.id'), primary_key=True)
)

t_SphereVisualShape = Table(
    'SphereVisualShape', metadata,
    Column('id', ForeignKey('VisualShape.id'), primary_key=True),
    Column('radius', Float, nullable=False)
)

t_StopMotionEvent = Table(
    'StopMotionEvent', metadata,
    Column('id', ForeignKey('MotionEvent.id'), primary_key=True)
)

t_SupportEvent = Table(
    'SupportEvent', metadata,
    Column('id', ForeignKey('DefaultEventWithTwoTrackedObjects.id'), primary_key=True)
)

t_TranslationEvent = Table(
    'TranslationEvent', metadata,
    Column('id', ForeignKey('MotionEvent.id'), primary_key=True)
)

t_AgentContactEvent = Table(
    'AgentContactEvent', metadata,
    Column('id', ForeignKey('ContactEvent.id'), primary_key=True)
)

t_AgentLossOfContactEvent = Table(
    'AgentLossOfContactEvent', metadata,
    Column('id', ForeignKey('LossOfContactEvent.id'), primary_key=True)
)

t_CylinderVisualShape = Table(
    'CylinderVisualShape', metadata,
    Column('id', ForeignKey('CapsuleVisualShape.id'), primary_key=True)
)

t_InterferenceEvent = Table(
    'InterferenceEvent', metadata,
    Column('id', ForeignKey('ContactEvent.id'), primary_key=True)
)

t_LossOfInterferenceEvent = Table(
    'LossOfInterferenceEvent', metadata,
    Column('id', ForeignKey('LossOfContactEvent.id'), primary_key=True)
)

t_StopRotationEvent = Table(
    'StopRotationEvent', metadata,
    Column('id', ForeignKey('StopMotionEvent.id'), primary_key=True)
)

t_StopTranslationEvent = Table(
    'StopTranslationEvent', metadata,
    Column('id', ForeignKey('StopMotionEvent.id'), primary_key=True)
)

t_AgentInterferenceEvent = Table(
    'AgentInterferenceEvent', metadata,
    Column('id', ForeignKey('InterferenceEvent.id'), primary_key=True)
)

t_AgentLossOfInterferenceEvent = Table(
    'AgentLossOfInterferenceEvent', metadata,
    Column('id', ForeignKey('LossOfInterferenceEvent.id'), primary_key=True)
)

mapper_registry = registry(metadata=metadata)

m_ContactPoint = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.ContactPoint, t_ContactPoint, properties = dict(position_on_body_a=relationship('Vector3',foreign_keys=[t_ContactPoint.c.position_on_body_a_id]), 
position_on_body_b=relationship('Vector3',foreign_keys=[t_ContactPoint.c.position_on_body_b_id]), 
normal_on_body_b=relationship('Vector3',foreign_keys=[t_ContactPoint.c.normal_on_body_b_id]), 
lateral_friction_1=relationship('LateralFriction',foreign_keys=[t_ContactPoint.c.lateral_friction_1_id]), 
lateral_friction_2=relationship('LateralFriction',foreign_keys=[t_ContactPoint.c.lateral_friction_2_id]), 
body_a_frozen_cp=relationship('FrozenBody',foreign_keys=[t_ContactPoint.c.body_a_frozen_cp_id]), 
body_b_frozen_cp=relationship('FrozenBody',foreign_keys=[t_ContactPoint.c.body_b_frozen_cp_id])), polymorphic_on = "polymorphic_type", polymorphic_identity = "ContactPoint")

m_ContactPointsList = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.ContactPointsList, t_ContactPointsList, properties = dict(points=relationship('ContactPoint',foreign_keys=[t_ContactPoint.c.contactpointslist_points_id])), polymorphic_on = "polymorphic_type", polymorphic_identity = "ContactPointsList")

m_VisualShape = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.VisualShape, t_VisualShape, properties = dict(rgba_color=relationship('Color',foreign_keys=[t_VisualShape.c.rgba_color_id]), 
visual_frame_position=relationship('Vector3',foreign_keys=[t_VisualShape.c.visual_frame_position_id])), polymorphic_on = "polymorphic_type", polymorphic_identity = "VisualShape")

m_Event = mapper_registry.map_imperatively(segmind.datastructures.events.Event, t_Event, polymorphic_on = "polymorphic_type", polymorphic_identity = "Event")

m_FrozenBody = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.FrozenBody, t_FrozenBody, properties = dict(pose=relationship('PoseStamped',foreign_keys=[t_FrozenBody.c.pose_id]), 
velocity=relationship('Vector3',foreign_keys=[t_FrozenBody.c.velocity_id]), 
bounding_box=relationship('AxisAlignedBoundingBox',foreign_keys=[t_FrozenBody.c.bounding_box_id]), 
concept=t_FrozenBody.c.concept), polymorphic_on = "polymorphic_type", polymorphic_identity = "FrozenBody")

m_BoundingBox = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.BoundingBox, t_BoundingBox, polymorphic_on = "polymorphic_type", polymorphic_identity = "BoundingBox")

m_Vector3 = mapper_registry.map_imperatively(pycram.datastructures.pose.Vector3, t_Vector3, polymorphic_on = "polymorphic_type", polymorphic_identity = "Vector3")

m_Pose = mapper_registry.map_imperatively(pycram.datastructures.pose.Pose, t_Pose, properties = dict(position=relationship('Vector3',foreign_keys=[t_Pose.c.position_id]), 
orientation=relationship('Quaternion',foreign_keys=[t_Pose.c.orientation_id])), polymorphic_on = "polymorphic_type", polymorphic_identity = "Pose")

m_LateralFriction = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.LateralFriction, t_LateralFriction, properties = dict(lateral_friction_direction=relationship('Vector3',foreign_keys=[t_LateralFriction.c.lateral_friction_direction_id])))

m_Color = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.Color, t_Color, )

m_PoseStamped = mapper_registry.map_imperatively(pycram.datastructures.pose.PoseStamped, t_PoseStamped, properties = dict(pose=relationship('Pose',foreign_keys=[t_PoseStamped.c.pose_id]), 
header=relationship('Header',foreign_keys=[t_PoseStamped.c.header_id])), polymorphic_on = "polymorphic_type", polymorphic_identity = "PoseStamped")

m_FrozenJoint = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.FrozenJoint, t_FrozenJoint, )

m_FrozenWorldState = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.FrozenWorldState, t_FrozenWorldState, properties = dict(objects=relationship('FrozenObject',foreign_keys=[t_FrozenObject.c.frozenworldstate_objects_id])))

m_Quaternion = mapper_registry.map_imperatively(pycram.datastructures.pose.Quaternion, t_Quaternion, )

m_Header = mapper_registry.map_imperatively(pycram.datastructures.pose.Header, t_Header, )

m_ClosestPoint = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.ClosestPoint, t_ClosestPoint, polymorphic_identity = "ClosestPoint", inherits = m_ContactPoint)

m_ClosestPointsList = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.ClosestPointsList, t_ClosestPointsList, polymorphic_identity = "ClosestPointsList", inherits = m_ContactPointsList)

m_BoxVisualShape = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.BoxVisualShape, t_BoxVisualShape, properties = dict(half_extents=relationship('Vector3',foreign_keys=[t_BoxVisualShape.c.half_extents_id])), polymorphic_identity = "BoxVisualShape", inherits = m_VisualShape)

m_PlaneVisualShape = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.PlaneVisualShape, t_PlaneVisualShape, properties = dict(normal=relationship('Vector3',foreign_keys=[t_PlaneVisualShape.c.normal_id])), polymorphic_identity = "PlaneVisualShape", inherits = m_VisualShape)

m_SphereVisualShape = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.SphereVisualShape, t_SphereVisualShape, polymorphic_identity = "SphereVisualShape", inherits = m_VisualShape)

m_MeshVisualShape = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.MeshVisualShape, t_MeshVisualShape, properties = dict(scale=relationship('Vector3',foreign_keys=[t_MeshVisualShape.c.scale_id])), polymorphic_identity = "MeshVisualShape", inherits = m_VisualShape)

m_CapsuleVisualShape = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.CapsuleVisualShape, t_CapsuleVisualShape, polymorphic_identity = "CapsuleVisualShape", inherits = m_VisualShape)

m_EventWithTrackedObjects = mapper_registry.map_imperatively(segmind.datastructures.events.EventWithTrackedObjects, t_EventWithTrackedObjects, polymorphic_identity = "EventWithTrackedObjects", inherits = m_Event)

m_FrozenLink = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.FrozenLink, t_FrozenLink, properties = dict(geometry=relationship('VisualShape',foreign_keys=[t_VisualShape.c.frozenlink_geometry_id])), polymorphic_identity = "FrozenLink", inherits = m_FrozenBody)

m_FrozenObject = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.FrozenObject, t_FrozenObject, properties = dict(links=relationship('FrozenLink',foreign_keys=[t_FrozenLink.c.frozenobject_links_id]), 
joints=relationship('FrozenJoint',foreign_keys=[t_FrozenJoint.c.frozenobject_joints_id])), polymorphic_identity = "FrozenObject", inherits = m_FrozenBody)

m_AxisAlignedBoundingBox = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.AxisAlignedBoundingBox, t_AxisAlignedBoundingBox, polymorphic_identity = "AxisAlignedBoundingBox", inherits = m_BoundingBox)

m_RotatedBoundingBox = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.RotatedBoundingBox, t_RotatedBoundingBox, polymorphic_identity = "RotatedBoundingBox", inherits = m_BoundingBox)

m_Vector3Stamped = mapper_registry.map_imperatively(pycram.datastructures.pose.Vector3Stamped, t_Vector3Stamped, properties = dict(header=relationship('Header',foreign_keys=[t_Vector3Stamped.c.header_id])), polymorphic_identity = "Vector3Stamped", inherits = m_Vector3)

m_Transform = mapper_registry.map_imperatively(pycram.datastructures.pose.Transform, t_Transform, polymorphic_identity = "Transform", inherits = m_Pose)

m_TransformStamped = mapper_registry.map_imperatively(pycram.datastructures.pose.TransformStamped, t_TransformStamped, properties = dict(pose=relationship('Transform',foreign_keys=[t_TransformStamped.c.pose_id])), polymorphic_identity = "TransformStamped", inherits = m_PoseStamped)

m_GraspPose = mapper_registry.map_imperatively(pycram.datastructures.pose.GraspPose, t_GraspPose, polymorphic_identity = "GraspPose", inherits = m_PoseStamped)

m_CylinderVisualShape = mapper_registry.map_imperatively(pycram.datastructures.dataclasses.CylinderVisualShape, t_CylinderVisualShape, polymorphic_identity = "CylinderVisualShape", inherits = m_CapsuleVisualShape)

m_EventWithTwoTrackedObjects = mapper_registry.map_imperatively(segmind.datastructures.events.EventWithTwoTrackedObjects, t_EventWithTwoTrackedObjects, properties = dict(with_object_frozen_cp=relationship('FrozenObject',foreign_keys=[t_EventWithTwoTrackedObjects.c.with_object_frozen_cp_id]), 
tracked_object_frozen_cp=relationship('FrozenObject',foreign_keys=[t_EventWithTwoTrackedObjects.c.tracked_object_frozen_cp_id]), 
world_frozen_cp=relationship('FrozenWorldState',foreign_keys=[t_EventWithTwoTrackedObjects.c.world_frozen_cp_id])), polymorphic_identity = "EventWithTwoTrackedObjects", inherits = m_EventWithTrackedObjects)

m_EventWithOneTrackedObject = mapper_registry.map_imperatively(segmind.datastructures.events.EventWithOneTrackedObject, t_EventWithOneTrackedObject, properties = dict(tracked_object_frozen_cp=relationship('FrozenObject',foreign_keys=[t_EventWithOneTrackedObject.c.tracked_object_frozen_cp_id]), 
world_frozen_cp=relationship('FrozenWorldState',foreign_keys=[t_EventWithOneTrackedObject.c.world_frozen_cp_id])), polymorphic_identity = "EventWithOneTrackedObject", inherits = m_EventWithTrackedObjects)

m_AbstractContactEvent = mapper_registry.map_imperatively(segmind.datastructures.events.AbstractContactEvent, t_AbstractContactEvent, properties = dict(contact_points=relationship('ContactPointsList',foreign_keys=[t_AbstractContactEvent.c.contact_points_id]), 
latest_contact_points=relationship('ContactPointsList',foreign_keys=[t_AbstractContactEvent.c.latest_contact_points_id]), 
bounding_box=relationship('AxisAlignedBoundingBox',foreign_keys=[t_AbstractContactEvent.c.bounding_box_id]), 
pose=relationship('PoseStamped',foreign_keys=[t_AbstractContactEvent.c.pose_id]), 
with_object_bounding_box=relationship('AxisAlignedBoundingBox',foreign_keys=[t_AbstractContactEvent.c.with_object_bounding_box_id]), 
with_object_pose=relationship('PoseStamped',foreign_keys=[t_AbstractContactEvent.c.with_object_pose_id])), polymorphic_identity = "AbstractContactEvent", inherits = m_EventWithTwoTrackedObjects)

m_DefaultEventWithTwoTrackedObjects = mapper_registry.map_imperatively(segmind.datastructures.events.DefaultEventWithTwoTrackedObjects, t_DefaultEventWithTwoTrackedObjects, polymorphic_identity = "DefaultEventWithTwoTrackedObjects", inherits = m_EventWithTwoTrackedObjects)

m_AbstractAgentObjectInteractionEvent = mapper_registry.map_imperatively(segmind.datastructures.events.AbstractAgentObjectInteractionEvent, t_AbstractAgentObjectInteractionEvent, properties = dict(agent_frozen_cp=relationship('FrozenObject',foreign_keys=[t_AbstractAgentObjectInteractionEvent.c.agent_frozen_cp_id])), polymorphic_identity = "AbstractAgentObjectInteractionEvent", inherits = m_EventWithTwoTrackedObjects)

m_NewObjectEvent = mapper_registry.map_imperatively(segmind.datastructures.events.NewObjectEvent, t_NewObjectEvent, polymorphic_identity = "NewObjectEvent", inherits = m_EventWithOneTrackedObject)

m_MotionEvent = mapper_registry.map_imperatively(segmind.datastructures.events.MotionEvent, t_MotionEvent, properties = dict(start_pose=relationship('PoseStamped',foreign_keys=[t_MotionEvent.c.start_pose_id]), 
current_pose=relationship('PoseStamped',foreign_keys=[t_MotionEvent.c.current_pose_id])), polymorphic_identity = "MotionEvent", inherits = m_EventWithOneTrackedObject)

m_ContactEvent = mapper_registry.map_imperatively(segmind.datastructures.events.ContactEvent, t_ContactEvent, polymorphic_identity = "ContactEvent", inherits = m_AbstractContactEvent)

m_LossOfContactEvent = mapper_registry.map_imperatively(segmind.datastructures.events.LossOfContactEvent, t_LossOfContactEvent, polymorphic_identity = "LossOfContactEvent", inherits = m_AbstractContactEvent)

m_AbstractAgentContact = mapper_registry.map_imperatively(segmind.datastructures.events.AbstractAgentContact, t_AbstractAgentContact, polymorphic_identity = "AbstractAgentContact", inherits = m_AbstractContactEvent)

m_SupportEvent = mapper_registry.map_imperatively(segmind.datastructures.events.SupportEvent, t_SupportEvent, polymorphic_identity = "SupportEvent", inherits = m_DefaultEventWithTwoTrackedObjects)

m_ContainmentEvent = mapper_registry.map_imperatively(segmind.datastructures.events.ContainmentEvent, t_ContainmentEvent, polymorphic_identity = "ContainmentEvent", inherits = m_DefaultEventWithTwoTrackedObjects)

m_LossOfSupportEvent = mapper_registry.map_imperatively(segmind.datastructures.events.LossOfSupportEvent, t_LossOfSupportEvent, polymorphic_identity = "LossOfSupportEvent", inherits = m_DefaultEventWithTwoTrackedObjects)

m_PickUpEvent = mapper_registry.map_imperatively(segmind.datastructures.events.PickUpEvent, t_PickUpEvent, polymorphic_identity = "PickUpEvent", inherits = m_AbstractAgentObjectInteractionEvent)

m_PlacingEvent = mapper_registry.map_imperatively(segmind.datastructures.events.PlacingEvent, t_PlacingEvent, properties = dict(placement_pose=relationship('PoseStamped',foreign_keys=[t_PlacingEvent.c.placement_pose_id])), polymorphic_identity = "PlacingEvent", inherits = m_AbstractAgentObjectInteractionEvent)

m_TranslationEvent = mapper_registry.map_imperatively(segmind.datastructures.events.TranslationEvent, t_TranslationEvent, polymorphic_identity = "TranslationEvent", inherits = m_MotionEvent)

m_StopMotionEvent = mapper_registry.map_imperatively(segmind.datastructures.events.StopMotionEvent, t_StopMotionEvent, polymorphic_identity = "StopMotionEvent", inherits = m_MotionEvent)

m_RotationEvent = mapper_registry.map_imperatively(segmind.datastructures.events.RotationEvent, t_RotationEvent, polymorphic_identity = "RotationEvent", inherits = m_MotionEvent)

m_InterferenceEvent = mapper_registry.map_imperatively(segmind.datastructures.events.InterferenceEvent, t_InterferenceEvent, polymorphic_identity = "InterferenceEvent", inherits = m_ContactEvent)

m_LossOfInterferenceEvent = mapper_registry.map_imperatively(segmind.datastructures.events.LossOfInterferenceEvent, t_LossOfInterferenceEvent, polymorphic_identity = "LossOfInterferenceEvent", inherits = m_LossOfContactEvent)

m_AgentContactEvent = mapper_registry.map_imperatively(segmind.datastructures.events.AgentContactEvent, t_AgentContactEvent, polymorphic_identity = "AgentContactEvent", inherits = m_ContactEvent)

m_AgentLossOfContactEvent = mapper_registry.map_imperatively(segmind.datastructures.events.AgentLossOfContactEvent, t_AgentLossOfContactEvent, polymorphic_identity = "AgentLossOfContactEvent", inherits = m_LossOfContactEvent)

m_StopTranslationEvent = mapper_registry.map_imperatively(segmind.datastructures.events.StopTranslationEvent, t_StopTranslationEvent, polymorphic_identity = "StopTranslationEvent", inherits = m_StopMotionEvent)

m_StopRotationEvent = mapper_registry.map_imperatively(segmind.datastructures.events.StopRotationEvent, t_StopRotationEvent, polymorphic_identity = "StopRotationEvent", inherits = m_StopMotionEvent)

m_AgentInterferenceEvent = mapper_registry.map_imperatively(segmind.datastructures.events.AgentInterferenceEvent, t_AgentInterferenceEvent, polymorphic_identity = "AgentInterferenceEvent", inherits = m_InterferenceEvent)

m_AgentLossOfInterferenceEvent = mapper_registry.map_imperatively(segmind.datastructures.events.AgentLossOfInterferenceEvent, t_AgentLossOfInterferenceEvent, polymorphic_identity = "AgentLossOfInterferenceEvent", inherits = m_LossOfInterferenceEvent)
