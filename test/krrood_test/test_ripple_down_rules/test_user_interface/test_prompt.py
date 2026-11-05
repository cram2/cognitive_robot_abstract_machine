from unittest import TestCase, skip

from krrood.ripple_down_rules.datastructures.dataclasses import CaseQuery
from krrood.ripple_down_rules.datastructures.enums import PromptFor
from krrood.ripple_down_rules.user_interface.prompt import UserPrompt
from ..datasets import World, Handle, Container, FixedConnection, PrismaticConnection


class TestPrompt(TestCase):

    @skip
    def test_prompt(self):
        user_prompt = UserPrompt()
        world = World()

        handle = Handle(name="h1", world=world)
        handle_2 = Handle(name="h2", world=world)
        container_1 = Container(name="c1", world=world)
        container_2 = Container(name="c2", world=world)
        connection_1 = FixedConnection(parent=container_1, child=handle, world=world)
        connection_2 = PrismaticConnection(parent=container_2, child=container_1, world=world)

        world.bodies = [handle, container_1, container_2, handle_2]
        world.connections = [connection_1, connection_2]
        case_query = CaseQuery(original_case=world, attribute_name="views", _attribute_types=(Handle,), mutually_exclusive=False)
        user_prompt.prompt_user_about_case(case_query=case_query, prompt_for=PromptFor.Conclusion)
