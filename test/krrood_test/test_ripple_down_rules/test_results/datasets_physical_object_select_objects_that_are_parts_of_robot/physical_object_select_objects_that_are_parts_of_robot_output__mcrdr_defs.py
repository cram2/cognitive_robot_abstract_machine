from ...datasets import PhysicalObject, Robot
from typing_extensions import Dict, List, Set, Union


def conditions_164855806603893754507167918997373216146(case) -> bool:
    def conditions_for_physical_object_select_objects_that_are_parts_of_robot(self_: PhysicalObject,
                                                                              objects: List[PhysicalObject],
                                                                              robot: Robot,
                                                                              output_: PhysicalObject) -> bool:
        """Get conditions on whether it's possible to conclude a value for PhysicalObject_select_objects_that_are_parts_of_robot.output_  of type PhysicalObject."""
        return robot is not None

    return conditions_for_physical_object_select_objects_that_are_parts_of_robot(**case)


def conclusion_164855806603893754507167918997373216146(case) -> List[PhysicalObject]:
    def physical_object_select_objects_that_are_parts_of_robot(self_: PhysicalObject, objects: List[PhysicalObject],
                                                               robot: Robot, output_: PhysicalObject) -> List[
        PhysicalObject]:
        """Get possible value(s) for PhysicalObject_select_objects_that_are_parts_of_robot.output_  of type PhysicalObject."""
        robot_parts = [obj for obj in objects if obj in robot.parts]
        return robot_parts

    return physical_object_select_objects_that_are_parts_of_robot(**case)
