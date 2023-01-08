from typing import Callable
import functools


def compose(functions: list[Callable]) -> Callable:
    return functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(functions), lambda x: x)

