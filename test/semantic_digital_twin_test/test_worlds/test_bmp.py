from copy import deepcopy

from krrood.entity_query_language.conclusion import Add, Set
from krrood.entity_query_language.entity import let, entity, set_of, inference
from krrood.entity_query_language.match import match, entity_matching
from krrood.entity_query_language.quantify_entity import an
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.reasoning.predicates_base import (
    SatisfiesRequest,
    Causes,
)
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Container,
)
from semantic_digital_twin.semantic_annotations.task_effect_motion import (
    OpenedEffect,
    ClosedEffect,
    TaskRequest,
    Effect,
    Motion,
    RunMSCModel,
)
from semantic_digital_twin.testing import apartment_world
from semantic_digital_twin.world import World


class TestBodyMotionProblem:
    @staticmethod
    def _get_effect_execution_model_for_open_goal(
        handle_body, actuator, goal_value
    ) -> RunMSCModel:
        msc = MotionStatechart()
        goal = Open(
            tip_link=handle_body,
            environment_link=handle_body,
            goal_joint_state=goal_value,
        )
        msc.add_node(goal)
        msc.add_node(EndMotion.when_true(goal))

        return RunMSCModel(msc=msc, actuator=actuator, timeout=500)

    def _extend_world(self, world: World):
        with world.modify_world():
            world_reasoner = WorldReasoner(world)
            world_reasoner.reason()

        drawers = world.get_semantic_annotations_by_type(Drawer)

        # Define effects for the drawers
        effects = []
        motions = []
        property_getter = lambda obj: obj.container.body.parent_connection.position
        for drawer in drawers:
            effect_open = OpenedEffect(
                target_object=drawer, goal_value=0.3, property_getter=property_getter
            )
            close_effect = ClosedEffect(
                target_object=drawer,
                goal_value=0.0,
                property_getter=property_getter,
            )
            effects.append(effect_open)
            effects.append(close_effect)
            close_motion = Motion(
                trajectory=[],
                actuator=drawer.container.body.parent_connection,
                motion_model=self._get_effect_execution_model_for_open_goal(
                    drawer.handle.body, drawer.container.body.parent_connection, 0.0
                ),
            )
            open_motion = Motion(
                trajectory=[],
                actuator=drawer.container.body.parent_connection,
                motion_model=self._get_effect_execution_model_for_open_goal(
                    drawer.handle.body, drawer.container.body.parent_connection, 0.3
                ),
            )
            motions.append(open_motion)
            motions.append(close_motion)

        # Define simple TaskRequests
        open_task = TaskRequest(task_type="open")
        close_task = TaskRequest(task_type="close")
        return effects, motions, open_task, close_task, drawers

    def test_query_motion_satisfying_task_request1(self, apartment_world: World):
        effects, motions, open_task, close_task, _ = self._extend_world(apartment_world)

        # Define Krrood symbols
        task_sym = let(TaskRequest, domain=[open_task, close_task])
        effect_sym = let(Effect, domain=effects)
        motion_sym = let(Motion, domain=motions)

        # Define Predicates for the query
        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(
            effect=effect_sym, motion=motion_sym, environment=apartment_world
        )

        query = an(
            set_of(
                [causes_opening.motion, effect_sym, task_sym],
                satisfies_request,
                causes_opening,
            )
        )
        results = list(query.evaluate())

        motion_key, effect_key, request_key = list(results[0].data.keys())
        print(
            f"A solution to fulfill the request {results[0].data[request_key].task_type} \n"
            f"is to achieve effect {results[0].data[effect_key].name} by executing the motion \n"
            f"trajectory {results[0].data[motion_key].trajectory} \n"
            f"on the DoF {results[0].data[motion_key].actuator.name}"
        )

    def test_query_task_and_effect_satisfying_motion(self, apartment_world: World):
        effects, _, open_task, close_task, drawers = self._extend_world(apartment_world)

        # Define a motion
        motion = Motion(
            trajectory=[0.0, 0.1, 0.2, 0.3, 0.4],
            actuator=drawers[0].container.body.parent_connection,
        )

        # Define Krrood symbols
        task_sym = let(TaskRequest, domain=[open_task, close_task])
        effect_sym = let(Effect, domain=effects)
        motion_sym = let(Motion, domain=[motion])

        # Define Predicates for the query
        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(
            effect=effect_sym, motion=motion_sym, environment=apartment_world
        )

        query = an(
            set_of(
                [motion_sym, effect_sym, task_sym], satisfies_request, causes_opening
            )
        )
        results = list(query.evaluate())

        motion_key, effect_key, request_key = list(results[0].data.keys())
        print(
            f"The trajectory {results[0].data[motion_key].trajectory} \n"
            f"on the DoF {results[0].data[motion_key].actuator.name} \n"
            f"can be caused by effect {results[0].data[effect_key].name} \n"
            f"Which satisfies the request {results[0].data[request_key].task_type} \n"
        )

    def test_query_motion_if_drawers_open(self, apartment_world: World):
        effects, motions, open_task, close_task, drawers = self._extend_world(
            apartment_world
        )

        # Define Krrood symbols
        task_sym = let(TaskRequest, domain=[open_task, close_task])
        effect_sym = let(Effect, domain=effects)
        motion_sym = let(Motion, domain=motions)

        # Define Predicates for the query
        satisfies_request = SatisfiesRequest(task=task_sym, effect=effect_sym)
        causes_opening = Causes(
            effect=effect_sym, motion=motion_sym, environment=apartment_world
        )

        # Query for motions that can be causes in the current world based on defined task requests, effects
        # and the world state. Only OpenedEffects should be available, as all drawers are closed
        query = an(
            set_of(
                [causes_opening.motion, effect_sym, task_sym],
                satisfies_request,
                causes_opening,
            )
        )
        results = list(query.evaluate())

        # assert that all entries are "open"
        assert all([res.data[task_sym].task_type == "open" for res in results])
        print("first query done with task type ", results[0].data[task_sym].task_type)

        # change the world state
        for drawer in drawers:
            drawer.container.body.parent_connection.position = 0.3
        apartment_world.notify_state_change()

        # query again
        results = list(query.evaluate())

        assert all([res.data[task_sym].task_type == "close" for res in results])
        print("second query done with task type ", results[0].data[task_sym].task_type)

    def test_query_motion_satisfying_task_request2(self, apartment_world: World):
        effects, motions, open_task, close_task, drawers = self._extend_world(
            apartment_world
        )

        effect_sym = let(Effect, domain=effects)
        task_sym = let(TaskRequest, domain=[open_task, close_task])
        motion_sym = let(Motion, domain=motions)

        query = an(
            set_of(
                [task_sym.task_type, effect_sym.name, motion_sym.trajectory],
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(
                    effect=effect_sym, motion=motion_sym, environment=apartment_world
                ),
            )
        )

        results = list(query.evaluate())
        motion: Motion = results[0]
        print(len(results))
        print(motion)
