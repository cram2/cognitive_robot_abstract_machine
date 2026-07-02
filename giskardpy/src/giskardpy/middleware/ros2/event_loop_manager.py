import asyncio

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def get_event_loop():
    global _loop
    if _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop
