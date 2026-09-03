from datetime import timedelta


class ActionConfig:
    approach_clearance = 0.1
    """
    The gap in meters between an object and the gripper waiting to close on it.
    """

    retreat_distance = 0.1
    """
    The height in meters the gripper rises by once it holds an object.
    """

    navigate_keep_joint_states = True

    face_at_keep_joint_states = True

    execution_delay: timedelta = timedelta(seconds=0.0)
    """
    The delay between the execution of actions/motions to imitate real world execution
    time.
    """
