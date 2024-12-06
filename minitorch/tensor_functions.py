"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Return the gradient of the function."""
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Run the forward pass of the function."""
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Apply the function to the values"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    """A class to perform the negation function on a tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Return the negation of a tensor."""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Return the gradient of the negation function."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    """A class to perform the inverse function on a tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Return the inverse of a tensor."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Return the gradient of the inverse function."""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    """A class to perform element-wise addition between two tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Return the sum of two tensors."""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the gradient of the addition function."""
        return grad_output, grad_output


class Mul(Function):
    """A class to perform element-wise multiplication between two tensors."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Return the product of two tensors."""
        ctx.save_for_backward(a, b)

        # Tensor.backend.mul_zip(a, b)
        # Tensor wraps tensor backend (tensor.py)
        # TensorBackend is implemented in tensor_ops.py
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the gradient of the multiplication function."""
        a, b = ctx.saved_values
        return (
            grad_output.f.mul_zip(grad_output, b),
            grad_output.f.mul_zip(a, grad_output),
        )


class Sigmoid(Function):
    """A class to perform the sigmoid function on a tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Return the sigmoid of a tensor."""
        sig = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Derivative of the sigmoid function is sigmoid(x) * (1 - sigmoid(x)) * grad_output."""
        (sig,) = ctx.saved_values
        neg_sig = sig.f.neg_map(sig)
        return grad_output.f.mul_zip(
            sig.f.mul_zip(sig, 1 + neg_sig),
            grad_output,
        )


class ReLU(Function):
    """A class to perform the ReLU function on a tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Return the ReLU of a tensor."""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Return the gradient of the ReLU function."""
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    """A class to perform the natural logarithm function on a tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Return the log of a tensor."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Return the gradient of the log function."""
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    """A class to perform the exponential function on a tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Return the exponential of a tensor."""
        exp = t1.f.exp_map(t1)
        ctx.save_for_backward(exp)
        return exp

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Return the gradient of the exponential function."""
        (exp,) = ctx.saved_values
        return grad_output.f.mul_zip(exp, grad_output)


class Sum(Function):
    """A class to perform a reduction operation that sums a tensor along a dimension."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Return the sum of a tensor along a dimension."""
        ctx.save_for_backward(t1.shape, dim)
        return t1.f.add_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Return the gradient of the sum function."""
        # original_shape, dim = ctx.saved_values
        # expanded_grad = grad_output.expand(original_shape)
        return grad_output, 0.0


class All(Function):
    """A function to perform a reduction operation that checks whether all elements
    in a tensor are non-zero along a specific dimension or across the entire tensor.
    """

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all elements in the tensor are non-zero along a dimension."""
        return a.f.mul_reduce(a, int(dim.item()))


class LT(Function):
    """A function to perform a comparison operation that checks whether the first tensor
    is less than the second tensor.
    """

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Return 1 if the first tensor is less than the second."""
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Derivative of the less than function is 0."""
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(Function):
    """A class to perform a comparison operation that checks whether two tensors are equal."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Return 1 if the tensors are equal."""
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Derivative of the equal function is 0."""
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(Function):
    """A class to perform a comparison operation that checks whether two tensors are close."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Return 1 if the tensors are close."""
        return a.f.is_close_zip(a, b)


class Permute(Function):
    """A class to permute the dimensions of a tensor."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permute the dimensions of a tensor, given a new order."""
        # permute(self, *order:int) is a method defined for TensorData in tensor_data.py
        # Tensor._tensor is an instance of TensorData
        # TensorData._storage is of type Storage (tensor_data.py), a np array of float64
        # So we must convert to a list of integers to pass as `order`
        int_order = [int(x) for x in order._tensor._storage]
        ctx.save_for_backward(int_order)

        # a._new creates a new tensor with the same backend as `a`
        return a._new(a._tensor.permute(*int_order))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Permute the gradients back to the original order."""
        (new_order,) = ctx.saved_values

        # Create a mapping from value to index for efficient lookup
        # Use the mapping to construct the original order
        original_order_map = {v: i for i, v in enumerate(new_order)}
        original_order = [original_order_map[i] for i in range(len(new_order))]

        return grad_output._new(grad_output._tensor.permute(*original_order)), 0.0


class View(Function):
    """A class to view a tensor with a new shape."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Return a view of a tensor with a new shape."""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Return the gradient of the view function."""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    """A class to copy a tensor."""

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Return a copy of a tensor."""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Return the gradient of the copy function."""
        return grad_output


class MatMul(Function):
    """A class to perform matrix multiplication between two tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Return the matrix multiplication of two tensors."""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the gradient of the matrix multiplication function."""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [float(0)] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the numerical gradient of a scalar function `f` using the central difference method.

    This method estimates the partial derivative of `f` with respect to the argument `arg` at the
    specified index `ind`, based on small perturbations (`epsilon`) in both the positive and
    negative directions.

    Args:
    ----
        f (Any): The function for which the gradient is to be computed.
                 It should take multiple tensors as arguments and return a tensor output.
        *vals (Tensor): The input tensors to the function `f`. The gradient will be computed with
                        respect to the tensor at the position specified by `arg`.
        arg (int, optional): The index of the tensor in `vals` with respect to which the gradient
                             will be computed. Default is 0.
        epsilon (float, optional): The small perturbation added and subtracted to compute the
                                   central difference. Default is 1e-6.
        ind (UserIndex): The index within the tensor `vals[arg]` at which to compute the partial derivative.

    Returns:
    -------
        float: The estimated partial derivative of `f` with respect to the tensor at `vals[arg]`
               at index `ind`, computed using the central difference method.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check if autodiff result is close to central difference result"""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )


# --------
