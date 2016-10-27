import functools

from functional import compose, partial
import tensorflow as tf

def compose_all(*args):
    """Util for multiple function composition
    i.e. composed = composeAll([f, g, h])
         composed(x) # == f(g(h(x)))
    """
    return partial(functools.reduce, compose)(*args)

