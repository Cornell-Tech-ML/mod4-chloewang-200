from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    """Base class for all optimizers.

    This class holds the common functionalities and attributes for optimization algorithms
    such as storing and managing the parameters that will be updated during training.

    Attributes
    ----------
    parameters (Sequence[Parameter]): A sequence of `Parameter` objects to optimize.

    """

    def __init__(self, parameters: Sequence[Parameter]):
        """Initialize the Optimizer with the given parameters.

        Args:
        ----
        parameters (Sequence[Parameter]): The list of parameters (typically model weights)
                                          that will be optimized during training.

        """
        self.parameters = parameters


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer.

    This class implements the SGD optimization algorithm, which updates the parameters by
    moving in the direction of the negative gradient. It supports both scalar derivatives
    and tensor gradients depending on the type of the parameters.

    Attributes
    ----------
    parameters (Sequence[Parameter]): A sequence of `Parameter` objects to optimize.
    lr (float): The learning rate for the SGD optimizer. Defaults to 1.0.

    Methods
    -------
    zero_grad() -> None:
        Resets the gradients of all parameters to zero, preparing for the next optimization step.

    step() -> None:
        Performs a single optimization step, updating the parameters using the current gradients.

    """

    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        """Initialize the SGD optimizer with the given parameters and learning rate.

        Args:
        ----
        parameters (Sequence[Parameter]): The list of parameters (typically model weights)
                                          that will be optimized during training.
        lr (float): The learning rate that determines the step size during optimization. Defaults to 1.0.

        """
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Reset all gradients to zero.

        This method sets the derivative or gradient of each parameter to `None`, effectively
        clearing any accumulated gradients from previous steps.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Perform a single optimization step.

        This method updates the parameters by subtracting the learning rate multiplied by the
        derivative (for scalars) or the gradient (for tensors). It modifies each parameter in
        the `parameters` list by applying the gradient descent rule.
        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
