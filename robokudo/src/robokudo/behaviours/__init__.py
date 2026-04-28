"""
Core behaviors for RoboKudo behavior trees.

This package provides a collection of reusable behaviors that form the core
functionality of RoboKudo behavior trees. It includes behaviors for:

* Action server monitoring and control
* Error handling and goal cancellation
* Timing and synchronization
* Pipeline initialization and cleanup

These behaviors are designed to be composed into larger behavior trees to
create complex robotic control systems.

Available Modules:

* action_server_base - Base class for ROS action servers
* action_server_checks - Action server state monitoring
* clear_errors - Error state management
* goal_canceled - Goal cancellation detection
* run_until - Timed execution control
"""
