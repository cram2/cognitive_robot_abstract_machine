from py_trees.decorators import Condition


class BlockingCondition(Condition):
    """
    Alias class which only provides some meaning to the standard py_trees Condition class.
    As "Condition" in the BT literature is usually defined as a leaf node and not as a decorator
    which can implement loop-like Behaviour, the name can cause confusion.
    """

    pass


class LoopingDecorator(Condition):
    """
    Alias class which only provides some meaning to the standard py_trees Condition class.
    As "Condition" in the BT literature is usually defined as a leaf node and not as a decorator
    which can implement loop-like Behaviour, the name can cause confusion.
    """

    pass
