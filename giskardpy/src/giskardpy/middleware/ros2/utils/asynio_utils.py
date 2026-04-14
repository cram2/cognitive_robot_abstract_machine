import asyncio
from typing import Callable, Optional, Any


async def wait_until_not_none(variable_getter: Callable[[], Optional[Any]], check_interval: float = 0.1) -> Any:
    while variable_getter() is None:
        await asyncio.sleep(check_interval)
    return variable_getter()


async def wait_until_none(variable_getter: Callable[[], Optional[Any]], check_interval: float = 0.1) -> Any:
    while variable_getter() is not None:
        await asyncio.sleep(check_interval)
    return variable_getter()
