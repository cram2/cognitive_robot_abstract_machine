import asyncio
from typing_extensions import Callable, Any


async def wait_until_not_none(
    variable_getter: Callable[[], Any | None], check_interval: float = 0.1
) -> Any:
    while variable_getter() is None:
        await asyncio.sleep(check_interval)
    return variable_getter()


async def wait_until_none(
    variable_getter: Callable[[], Any | None], check_interval: float = 0.1
) -> Any:
    while variable_getter() is not None:
        await asyncio.sleep(check_interval)
    return variable_getter()
