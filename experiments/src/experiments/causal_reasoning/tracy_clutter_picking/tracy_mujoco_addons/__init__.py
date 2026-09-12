"""
The MuJoCo stack the clutter-picking demo drives Tracy with: parsing and mounting the
robot, position servos, contact tuning, a real-time simulation stepped from the calling
thread, trajectory planning against a scratch copy of the world, and the pick and place
actions that play those trajectories back on the actuators.
"""
