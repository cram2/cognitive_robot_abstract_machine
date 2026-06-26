Action Conditions
=================

Actions in CoraPlex have pre -and postconditions which describe certain conditions of the world or the robot that have to
be satisfied for the action to be successfully executed. While the exact conditions for an action depend heavily on the
type of action as well as the context in which it is executed.
Usually it is easier to write conditions for more complex actions since they are more narrow in scope and are already more
restricted by their semantic.
However, the conditions of the actions currently defined are the minimal set of conditions that could reasonably be assumed.

EQL and Conditions
------------------

Pre -and Postconditions in CoraPlex are written in the Entity Query Language `EQL <https://cram2.github.io/cognitive_robot_abstract_machine/krrood/eql/intro.html>`__. which allow to query the current state
of the world with SQL like queries.
A Precondition consists of variables which are used in predicates that form the condition itself. An EQL variable represents
a value of a certain type in a domain, the role of EQL is to find a set of values for the variables such that the condition is
satisfied.
The variables in conditions can be created in two ways:
    * Bound: The domain of the variable is only the value of the action
    * Unbound: The domain of the variable is a set of all values in the world that could be used for the variable.

This allows for two use-cases: conditions with bound variables represent exactly the situation of the action and can be
evaluated to determine if the action is feasible or not.
On the other hand a condition with unbound variables can  be used to query the world state to find values for the variables
that satisfy the condition. This allows to find parameter for actions which make them executable. This can also be used in
Partial Designator to find parameter for under-specified actions.

Example
-------

As an example we will look at the pre condition for the :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction`.
In general the pre -and postcondition can be anything that is an EQL predicate, a symbolic function or something that
evaluates to bool. Conditions are defined as static methods that receive the EQL ``variables``, the execution
``context`` and the action ``kwargs``.

.. code-block:: python

    @staticmethod
    def pre_condition(variables, context, kwargs):
        end_effector = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        return and_(
            GripperIsFree(end_effector),
            IsObjectReachableBy(
                robot=context.robot,
                world=context.world,
                arm=variables["arm"],
                object_designator=kwargs["object_designator"],
                grasp_description=kwargs["grasp_description"],
            ),
        )

This condition is comprised of two conditions, the first is that the gripper that should pick up the object is free and
not holding anything (:class:`~coraplex.querying.predicates.GripperIsFree`) and the second is that the object is
reachable (:class:`~coraplex.locations.pose_validator.IsObjectReachableBy`). The arm is the queried variable here, since
querying over other parameter (like the object to be picked up) would result in very unexpected behaviour of the plan.

Now imagine the following scenario, the robot is standing near the object it should pick up but the object cannot be
picked up with the specified arm, however using the other arm would enable the robot to execute the PickUp Action.
