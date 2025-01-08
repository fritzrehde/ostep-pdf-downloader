import asyncio
import functools


class AsyncCache:
    def __init__(self, func):
        self.cache = None
        self.func = func
        self.lock = asyncio.Lock()

    async def get(self):
        # Many async calls might be accessing this cache at the same time, so
        # only give exclusive access to the first so it can compute value, and
        # have others wait. Cache stampede-/dog-pilling-proof.
        async with self.lock:
            if self.cache is None:
                self.cache = await self.func()
            return self.cache


def acache(func):
    """
    Cache the result of calling an asynchronous function that takes no
    arguments. Essentially, this can be used for global "variables" (you access
    the "variable" by calling the func) that is lazily initialised on first call
    and can safely be accessed from multiple simultaneous async calls.
    """
    cache = AsyncCache(func)

    @functools.wraps(func)
    async def wrapper():
        return await cache.get()

    return wrapper
