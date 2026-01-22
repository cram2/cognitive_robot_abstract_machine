import cProfile
import unittest
import time
import random
from unittest import skip
import random_events_lib as rl


@skip
class BindingsTestCase(unittest.TestCase):

    def test_intervals_with_bindings_but_in_old_class(self):
        """
        Using bindings, but in the old structure
        """
        from random_events.interval import SimpleInterval, Bound, Interval

        t1 = time.time()
        for i in range(100):
            b = []
            for a in range(50):
                b.append(
                    SimpleInterval(
                        random.randint(0, 100),
                        random.randint(0, 100),
                        Bound.OPEN,
                        Bound.CLOSED,
                    )
                )
            z = Interval(*b)
            z.make_disjoint()
        print(f"\nwhen using the bindings within the old classes: {time.time() - t1}\n")

    def test_with_bindings_but_call_bindings_directly(self):
        """
        Only using bindings
        """
        from random_events.interval import Bound

        t1 = time.time()
        for i in range(100):
            b = set()
            for a in range(50):
                b.add(
                    rl.SimpleInterval(
                        random.randint(0, 100),
                        random.randint(0, 100),
                        Bound.OPEN.value,
                        Bound.CLOSED.value,
                    )
                )
            z = rl.Interval(b)
            z.make_disjoint()
        print(f"\nwhen calling the bindings directly: {time.time() - t1}\n")

    def test_intervals_old(self):
        """
        No bindings, only python
        """
        from random_events.before_bindings.interval_old import (
            SimpleInterval,
            Bound,
            Interval,
        )

        t1 = time.time()
        for i in range(100):
            b = []
            for a in range(50):
                b.append(
                    SimpleInterval(
                        random.randint(0, 100),
                        random.randint(0, 100),
                        Bound.OPEN,
                        Bound.CLOSED,
                    )
                )
            z = Interval(*b)
            z.make_disjoint()
        print(f"\nwithout bindings: {time.time() - t1}\n")


if __name__ == "__main__":
    cProfile.run("unittest.main()", sort="cumtime")
