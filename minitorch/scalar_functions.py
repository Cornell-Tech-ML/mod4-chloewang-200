from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to a set of scalar values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the sum of two values."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of the addition function."""
        return (d_output, d_output)


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the log of the input value."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the log function."""
        a = ctx.saved_values[0]
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
# mul
class Mul(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the multiplication of two values."""
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of the multiplication function."""
        # Compute the gradient of f(x, y) = x * y
        # Gradient w.r.t. a is b, and w.r.t. b is a
        a, b = ctx.saved_values

        return b * d_output, a * d_output


# inv
class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the inverse of the input value."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the inverse function."""
        a = ctx.saved_values[0]
        return operators.inv_back(a, d_output)


# neg


class Neg(ScalarFunction):
    """Negation function f(x) = -x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the negation of the input value."""
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the negation function."""
        return operators.neg(d_output)


# sigmoid


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1/(1 + e^(-x))"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the sigmoid of the input value."""
        res = operators.sigmoid(a)
        ctx.save_for_backward(res)
        return res

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the sigmoid function."""
        sigmoid_value = ctx.saved_values[0]
        return operators.mul(
            operators.mul(d_output, sigmoid_value),
            operators.add(1.0, operators.neg(sigmoid_value)),
        )


# relu
class ReLU(ScalarFunction):
    """ReLU function f(x) = max(0, x)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the ReLU of the input value."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of the ReLU function."""
        a = ctx.saved_values[0]
        return (operators.relu_back(a, d_output),)


# exp
class Exp(ScalarFunction):
    """Exponential function f(x) = e^x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the exponential of the input value."""
        exp_value = operators.exp(a)
        ctx.save_for_backward(exp_value)
        return exp_value

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the exponential function."""
        exp_value = ctx.saved_values[0]
        return operators.mul(d_output, exp_value)


# lt
class LT(ScalarFunction):
    """Less than function f(x, y) = 1 if x < y else 0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compares two values and returns 1 if the first is less than the second."""
        return 1.0 if operators.lt(a, b) else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of the less than function."""
        return 0.0, 0.0


# eq
class EQ(ScalarFunction):
    """Equality function $f(x, y) = 1 if x = y else 0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compares two values and returns 1 if the first is equal to the second."""
        return 1.0 if operators.eq(a, b) else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of the equality function."""
        # gradients are undefined
        return 0.0, 0.0
