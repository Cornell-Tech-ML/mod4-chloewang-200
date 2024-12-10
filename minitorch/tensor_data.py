from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for errors related to tensor indexing."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor index into a single-dimensional position in storage
    based on tensor strides.

    Args:
    ----
        index: Index tuple representing the position in a multidimensional array.
        strides: The strides of the tensor, which specify how far to move in each dimension.

    Returns:
    -------
        int: The single-dimensional position corresponding to the index in the storage.

    """
    position = 0
    for idx, stride in zip(index, strides):
        position += idx * stride
    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an ordinal (flat index) to a multidimensional index in the shape.

    Args:
    ----
        ordinal: Flat index in the tensor storage.
        shape: Shape of the tensor.
        out_index: Output array that will store the multidimensional index.

    Returns:
    -------
        None

    """
    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Converts a `big_index` from a larger shape to an `out_index` for a smaller shape using
    broadcasting rules.

    Args:
    ----
        big_index: Index for the larger shape.
        big_shape: Shape of the larger tensor.
        shape: Shape of the smaller tensor.
        out_index: Output array for the smaller tensor's index.

    Returns:
    -------
        None

    """
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            out_index[i] = 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two tensor shapes to a compatible shape.

    Args:
    ----
        shape1: First tensor shape.
        shape2: Second tensor shape.

    Returns:
    -------
        UserShape: A new shape that results from broadcasting `shape1` and `shape2`.

    Raises:
    ------
        IndexingError: If the shapes cannot be broadcasted.

    """
    result_shape = []
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)

    shape1 = (1,) * (max_len - len1) + shape1  # type: ignore
    shape2 = (1,) * (max_len - len2) + shape2  # type: ignore

    for dim1, dim2 in zip(shape1, shape2):
        if dim1 == dim2:
            result_shape.append(dim1)
        elif dim1 == 1:
            result_shape.append(dim2)
        elif dim2 == 1:
            result_shape.append(dim1)
        else:
            raise IndexingError(f"Cannot broadcast shapes {shape1} and {shape2}")

    return tuple(result_shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Compute strides for a given shape for a contiguous tensor.

    Args:
    ----
        shape: The shape of the tensor.

    Returns:
    -------
        UserStrides: The corresponding strides for the tensor shape.

    """
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    """A class representing a tensor's data, including storage, shape, and strides.
    Supports basic tensor operations like indexing, reshaping, and broadcasting.
    """

    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        """Initialize a new `TensorData` object with storage, shape, and strides.

        Args:
        ----
            storage: Either a sequence of floats or a NumPy array for tensor storage.
            shape: Shape of the tensor.
            strides: Optional strides for the tensor. If not provided, contiguous strides will be used.

        """
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert the tensor data to CUDA device memory."""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check whether the tensor layout is contiguous, i.e., outer dimensions have larger strides
        than inner dimensions.

        Returns
        -------
            bool: True if the tensor is contiguous.

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Static method to broadcast two shapes to create a new union shape.

        Args:
        ----
            shape_a: First tensor shape.
            shape_b: Second tensor shape.

        Returns:
        -------
            UserShape: The broadcasted shape.

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Get the flat storage index corresponding to a multidimensional tensor index.

        Args:
        ----
            index: Either an integer or a tuple of indices.

        Returns:
        -------
            int: The flat storage index.

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:
            aindex = array(index)

        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generate all valid indices for this tensor.

        Args:
        ----
            None

        Yields:
        ------
            Iterable[UserIndex]: An iterable of valid multidimensional indices.

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index for the tensor.

        Args:
        ----
            None

        Returns:
        -------
            UserIndex: A valid random index in the tensor.

        """
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Get the value at a specific index.

        Args:
        ----
            key: The multidimensional index.

        Returns:
        -------
            float: The value at the specified index.

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set the value at a specific index.

        Args:
        ----
            key: The multidimensional index.
            val: The value to set at the index.

        Returns:
        -------
            None

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the core tensor data as a tuple.

        Args:
        ----
            None

        Returns:
        -------
            Tuple[Storage, Shape, Strides]: The tensor's storage, shape, and strides as a tuple.

        """
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor based on the given order.

        Args:
        ----
            *order: A permutation of the dimensions.

        Returns:
        -------
            TensorData: A new `TensorData` with the dimensions permuted.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)
        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        """Convert the tensor to a string representation.

        Args:
        ----
            None

        Returns:
        -------
            str: The string representation of the tensor.

        """
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
