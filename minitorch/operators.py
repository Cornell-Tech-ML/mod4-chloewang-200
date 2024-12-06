"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiply two numbers and return the result."""
    return x * y


def id(x: float) -> float:
    """Return the input."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers and return the result."""
    if x is None:
        return y
    elif y is None:
        return x
    return x + y


def neg(x: float) -> float:
    """Negate the input."""
    return -x * 1.0


def lt(x: float, y: float) -> bool:
    """Return True if x is less than y, False otherwise."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Return True if x is equal to y, False otherwise."""
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Return True if x is within 1e-2 of y, False otherwise."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Return the sigmoid of x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return the ReLU of x."""
    return 0.0 if x <= 0 else x


def log(x: float) -> float:
    """Return the natural logarithm of x."""
    return math.log(x)


def exp(x: float) -> float:
    """Return e raised to the power of x."""
    return math.exp(x)


def inv(x: float) -> float:
    """Return the reciprocal of x."""
    return 1.0 / x


def log_back(x: float, z: float) -> float:
    """Return the derivative of log(x) times z."""
    return z / x


def inv_back(x: float, z: float) -> float:
    """Return the derivative of 1/x times z."""
    return -z / (x * x)


def relu_back(x: float, z: float) -> float:
    """Return the derivative of ReLU(x) times z."""
    return z if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float], xs: Iterable[float]) -> Iterable[float]:
    """Applies the function fn over each element of the iterable xs."""
    return [fn(x) for x in xs]


def zipWith(
    fn: Callable[[float, float], float], xs: Iterable[float], ys: Iterable[float]
) -> Iterable[float]:
    """Applies the function fn over each pair of elements from the iterables xs and ys."""
    return [fn(x, y) for x, y in zip(xs, ys)]


def reduce(fn: Callable[[float, float], float], xs: Iterable[float]) -> float:
    """Applies the function fn over the iterable xs starting with the first element of the iterable."""
    # Convert the iterable to an iterator
    it = iter(xs)

    try:
        # Use the first element of the iterator as the initial value
        accum_value = next(it)
    except StopIteration:
        # Return 0 if the iterable is empty
        return 0.0

    # Apply the function cumulatively to the items of the iterator
    for item in it:
        accum_value = fn(accum_value, item)

    return accum_value


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate each element of the iterable xs."""
    return map(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add each element of the iterables xs and ys together."""
    return zipWith(add, xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Return the sum of the elements of the iterable xs."""
    return reduce(add, xs)


def prod(xs: Iterable[float]) -> float:
    """Return the product of the elements of the iterable xs."""
    return reduce(mul, xs)
