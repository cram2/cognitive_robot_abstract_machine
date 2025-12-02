from functools import lru_cache


def test_automatic_lru_clearing():

    class LRUCacheTestClass:
        def __init__(self):
            self.current_contract = 201706
            self.futures = {201706: {"multiplier": 1000}, 201712: {"multiplier": 25}}

        def __hash__(self):
            return hash(self.current_contract)

        @property
        @lru_cache()
        def multiplier(self):
            return self.futures[self.current_contract]["multiplier"]


    CF = LRUCacheTestClass()
    assert CF.multiplier == 1000

    CF.current_contract = 201712
    assert CF.multiplier == 25, CF.multiplier
