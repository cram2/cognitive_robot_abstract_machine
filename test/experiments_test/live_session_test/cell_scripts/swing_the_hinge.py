# %% cell swing the hinge to half a radian
world.state[hinge.raw_dof.id].position = 0.5
world.notify_state_change()
simulation.step_simulation(timedelta(seconds=0.2))
swung_to = simulation.simulator.get_joint_value("hinge").result
session.report("swung to half a radian")

# %% cell swing the hinge back
world.state[hinge.raw_dof.id].position = -0.3
world.notify_state_change()
simulation.step_simulation(timedelta(seconds=0.2))
session.report("swung back")
