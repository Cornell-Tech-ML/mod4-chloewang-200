from typing import Tuple

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # Calculate the new dimensions after pooling
    new_height = height // kh
    new_width = width // kw

    # Reshape the tensor to include pooling blocks
    # Step 1: Split height and width into blocks of size kh and kw
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Step 2: Move kernel dimensions to the last axis for easier pooling
    tiled = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Step 3: Flatten the last two dimensions (kh and kw)
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling over a 2D input tensor.

    Args:
    ----
        input: Tensor of size batch x channel x height x width.
        kernel: Tuple (k_height, k_width) specifying the pooling window size.

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width after average pooling.

    """
    # Reshape the tensor for pooling using the `tile` function
    tiled, new_height, new_width = tile(input, kernel)

    # Compute the average over the last dimension (kernel window)
    pooled = tiled.mean(dim=-1).contiguous()

    pooled = pooled.view(input.shape[0], input.shape[1], new_height, new_width)
    return pooled


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Computes the forward pass of the max operation."""
        ctx.save_for_backward(t1, int(dim.item()))
        return t1.f.max_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor]:
        """Computes the backward pass of the max operation."""
        t1, dim = ctx.saved_values
        return (d_output, 0.0)  # type: ignore


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max of the tensor along a specified dimension."""
    # max_vals = input.f.max_reduce(input, dim)
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax of the tensor along a specified dimension.

    Args:
    ----
        input: Input tensor.
        dim: The dimension along which to compute softmax.

    Returns:
    -------
        Tensor of the same shape with softmax applied along the specified dimension.

    """
    exp_input = input.exp()
    return exp_input / exp_input.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax of the tensor along a specified dimension.

    Args:
    ----
        input: Input tensor.
        dim: The dimension along which to compute log softmax.

    Returns:
    -------
        Tensor of the same shape with log softmax applied along the specified dimension.

    """
    max_val = max(input, dim=dim)
    shifted_input = input - max_val
    log_sum_exp = (shifted_input.exp()).sum(dim=dim).log()
    return shifted_input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling over a 2D input tensor.

    Args:
    ----
        input: Tensor of size batch x channel x height x width.
        kernel: Tuple (k_height, k_width) specifying the pooling window size.

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width after max pooling.

    """
    # Reshape the tensor for pooling using the `tile` function
    tiled, new_height, new_width = tile(input, kernel)

    # Compute the max over the last dimension (kernel window)
    pooled = max(tiled, dim=-1).contiguous()

    pooled = pooled.view(input.shape[0], input.shape[1], new_height, new_width)

    return pooled


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the tensor during training.

    Args:
    ----
        input: Input tensor.
        p: Dropout probability.
        ignore: Whether in training mode (apply dropout) or evaluation mode (no dropout).

    Returns:
    -------
        Tensor with elements randomly zeroed out with probability `p` during training.

    """
    if ignore:
        return input

    mask = rand(input.shape) > p
    return input * mask
