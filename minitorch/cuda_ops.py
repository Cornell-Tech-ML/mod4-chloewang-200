# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Return a jitted function that runs on the GPU."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Return a jitted function that runs on the GPU."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Return a function that zips two tensors together."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            """Return a new tensor with the zipped function."""
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Return a function that reduces a tensor along a dimension."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            """Return a new tensor with the reduced function."""
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Return a new tensor with the matrix product of two tensors."""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        """Run a map operation on a tensor."""
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # Calculate the global thread index

        # Ensure the thread index is within the bounds of the output tensor
        if i >= out_size:
            return

        # Convert the flat index to multi-dimensional index
        to_index(i, out_shape, out_index)

        # Adjust for broadcasting
        broadcast_index(out_index, out_shape, in_shape, in_index)

        # Calculate positions in input and output storage
        in_pos = index_to_position(in_index, in_strides)
        out_pos = index_to_position(out_index, out_strides)

        # Apply the function and store the result
        out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        """Run a zip operation on two tensors."""
        # TODO: Implement for Task 3.3.
        # Calculate global thread ID
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Ensure thread ID is within bounds
        if i >= out_size:
            return

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Convert flat index to multi-dimensional index
        to_index(i, out_shape, out_index)

        # Adjust for broadcasting
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        # Calculate positions in storage
        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)
        out_pos = index_to_position(out_index, out_strides)

        # Apply the function and store the result
        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # Load values into shared memory
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0  # Handle out-of-bounds threads

    # Synchronize threads within the block
    cuda.syncthreads()

    # Perform parallel reduction within the block
    step = BLOCK_DIM // 2
    while step > 0:
        if pos < step:
            cache[pos] += cache[pos + step]
        step //= 2
        cuda.syncthreads()

    # Write the block result to the output
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Return the sum of a tensor."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        """Run a reduction operation on a tensor."""
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)

        # Define local and global thread indices
        global_pos = cuda.blockIdx.x
        thread_pos = cuda.threadIdx.x

        # Ensure we're operating within bounds
        if global_pos < out_size:
            # Calculate the output index and map it to the input space
            output_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(global_pos, out_shape, output_index)
            output_index[reduce_dim] = thread_pos

            # Load data into shared memory
            if thread_pos < a_shape[reduce_dim]:
                input_position = index_to_position(output_index, a_strides)
                cache[thread_pos] = a_storage[input_position]
            else:
                cache[thread_pos] = reduce_value
            cuda.syncthreads()

            # Perform reduction in shared memory
            step = 1
            while step < BLOCK_DIM:
                if thread_pos % (2 * step) == 0 and thread_pos + step < BLOCK_DIM:
                    cache[thread_pos] = fn(cache[thread_pos], cache[thread_pos + step])
                cuda.syncthreads()
                step *= 2

            # Store the result back into the global memory
            if thread_pos == 0:
                out[global_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32

    # Allocate shared memory for input tiles
    a_tile = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_tile = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread indices within the block
    local_row = cuda.threadIdx.y
    local_col = cuda.threadIdx.x

    # Global indices for this thread
    global_row = cuda.blockIdx.y * BLOCK_DIM + local_row
    global_col = cuda.blockIdx.x * BLOCK_DIM + local_col

    # Initialize accumulator for the dot product
    partial_sum = 0.0

    # Loop over tiles of `a` and `b` matrices to compute the dot product.
    for tile_start in range(0, size, BLOCK_DIM):
        # Load data from global memory into shared memory for the current tile of `a`.
        if global_row < size and tile_start + local_col < size:  # Check bounds for `a`.
            a_tile[local_row, local_col] = a[global_row * size + tile_start + local_col]
        else:
            a_tile[local_row, local_col] = 0.0  # Fill with zeros if out of bounds.

        # Load data from global memory into shared memory for the current tile of `b`.
        if tile_start + local_row < size and global_col < size:  # Check bounds for `b`.
            b_tile[local_row, local_col] = b[
                (tile_start + local_row) * size + global_col
            ]
        else:
            b_tile[local_row, local_col] = 0.0  # Fill with zeros if out of bounds.

        # Synchronize threads to ensure all threads in the block have completed loading data.
        cuda.syncthreads()

        # Compute the partial dot product for the elements in this tile.
        for k in range(BLOCK_DIM):  # Iterate over the shared memory tile.
            partial_sum += a_tile[local_row, k] * b_tile[k, local_col]

        # Synchronize threads before loading the next tile.
        cuda.syncthreads()

    # Store the computed value in the output matrix if the global indices are within bounds.
    if global_row < size and global_col < size:  # Check bounds for the output matrix.
        out[global_row * size + global_col] = partial_sum  # Write the computed value.


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Return the matrix multiply of two tensors."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Compute batch strides to handle batched matrix multiplication.
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0  # Stride for batch in `a`.
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0  # Stride for batch in `b`.

    # The current batch index.
    batch = cuda.blockIdx.z  # Each block in the z-dimension corresponds to a batch.

    # Define block size and shared memory tiles for `a` and `b`.
    BLOCK_DIM = 32  # Fixed block dimension (32x32 threads per block).
    a_shared = cuda.shared.array(
        (BLOCK_DIM, BLOCK_DIM), numba.float64
    )  # Shared memory for `a`.
    b_shared = cuda.shared.array(
        (BLOCK_DIM, BLOCK_DIM), numba.float64
    )  # Shared memory for `b`.

    # Compute global indices for the element `out[i, j]` being computed by this thread.
    i = (
        cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    )  # Global row index in output.
    j = (
        cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    )  # Global column index in output.

    # Compute local indices within the block for shared memory.
    pi = cuda.threadIdx.x  # Local row index within the block.
    pj = cuda.threadIdx.y  # Local column index within the block.

    # Initialize an accumulator to store the result of the dot product.
    acc = 0.0

    # Calculate the number of tiles required to cover the shared dimension.
    num_tiles = (a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM

    # Loop over tiles to compute the result.
    for tile in range(num_tiles):
        # Calculate the starting index of the current tile in the shared dimension.
        tile_k = tile * BLOCK_DIM + pj

        # Load a tile of `a` into shared memory, checking for bounds.
        if (
            i < a_shape[1] and tile_k < a_shape[-1]
        ):  # Ensure indices are within matrix dimensions.
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride + i * a_strides[1] + tile_k * a_strides[2]
            ]
        else:
            a_shared[pi, pj] = 0.0  # Set out-of-bounds elements to zero.

        # Load a tile of `b` into shared memory, checking for bounds.
        tile_k = tile * BLOCK_DIM + pi
        if (
            tile_k < b_shape[1] and j < b_shape[2]
        ):  # Ensure indices are within matrix dimensions.
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride + tile_k * b_strides[1] + j * b_strides[2]
            ]
        else:
            b_shared[pi, pj] = 0.0  # Set out-of-bounds elements to zero.

        # Synchronize threads to ensure all elements of the tile are loaded into shared memory.
        cuda.syncthreads()

        # Perform the dot product for the current tile.
        for k in range(BLOCK_DIM):
            acc += a_shared[pi, k] * b_shared[k, pj]  # Multiply and accumulate.

        # Synchronize again before loading the next tile to avoid race conditions.
        cuda.syncthreads()

    # Store the computed result in the output tensor if indices are within bounds.
    if i < out_shape[1] and j < out_shape[2]:  # Ensure valid output indices.
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = (
            acc  # Write result.
        )


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
