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

        handle = Handle("h1", world=world)
        handle_2 = Handle("h2", world=world)
        container_1 = Container("c1", world=world)
        container_2 = Container("c2", world=world)
        connection_1 = FixedConnection(container_1, handle, world=world)
        connection_2 = PrismaticConnection(container_2, container_1, world=world)

        world.bodies = [handle, container_1, container_2, handle_2]
        world.connections = [connection_1, connection_2]
        case_query = CaseQuery(world, "views", (Handle,), False)
        user_prompt.prompt_user_about_case(case_query, prompt_for=PromptFor.Conclusion)
