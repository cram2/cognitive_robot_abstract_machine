import asyncio

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

def get_event_loop():
    return _loop
