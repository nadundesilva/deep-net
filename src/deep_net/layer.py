"""Copyright (c) 2021, Deep Net. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Callable
from numpy.typing import ArrayLike
from deep_net.initializers import Initializer
from deep_net.activations import Activation
import numpy as np


class Layer:
    """
    Represents a single layer of a neural network.
    """

    _size: int
    _learning_rate: float
    _activation: Activation
    _W_initializer: Initializer
    _b_initializer: Initializer

    # Optimized paramaeters
    _W: ArrayLike = None
    _b: ArrayLike = None

    # Cached values (stored after forward pass and cleared after backward pass)
    _A_prev: ArrayLike = None

    def __init__(
        self,
        size: int,
        learning_rate: int,
        create_activation: Callable[[], Activation],
        W_initializer: Initializer,
        b_initializer: Initializer,
    ):
        """
        Initialize the layer of the Neural Network.

        :param size: The size of the Neural Network layer
        :param learning_rate: The learning rate to be used by this layer
        :param create_activation: A function returning an activation function to be used by this layer
        """
        self._size = size
        self._learning_rate = learning_rate
        self._activation = create_activation()
        self._W_initializer = W_initializer
        self._b_initializer = b_initializer

    def init_parameters(self, prev_layer_size: int) -> None:
        """
        Initialize the parameter tensors of a neural network.

        :param prev_layer_size: The size of the previous layer
        :param initializer: The initializer to be used for initializing the parameter tensors of this layer
        """
        W_shape = (self._size, prev_layer_size)
        if self._W is not None:
            if self._W.shape != W_shape:
                raise ValueError(
                    "Shape of the preset W does not match the expected shape"
                )
        else:
            self._W = self._W_initializer.init_tensor(W_shape)

        b_shape = (self._size, 1)
        if self._b is not None:
            if self._b.shape != b_shape:
                raise ValueError(
                    "Shape of the preset b does not match the expected shape"
                )
        else:
            self._b = self._b_initializer.init_tensor(b_shape)

    def propagate_forward(self, A_prev: ArrayLike) -> ArrayLike:
        """
        Propagate forward through the layer to predict a value.

        :param A_prev: The activation of the previous layer
        :returns: The activation of this layer
        """
        if self._W is None or self._b is None:
            raise ValueError("Layer parameters needs to initialized first")
        if len(A_prev.shape) != 2 or A_prev.shape[0] != self._W.shape[1]:
            raise ValueError(
                "Provided data tensor of invalid shape, expected: ("
                + str(self._W.shape[1])
                + ", mini_batch_size) provided: "
                + str(A_prev.shape)
            )

        self._A_prev = A_prev
        Z = np.dot(self._W, A_prev) + self._b
        return self._activation.map(Z)

    def propagate_backward(self, dA: ArrayLike) -> ArrayLike:
        """
        Propagate backward through the layer to train using mini batch Gradient Descent.
        The parameters of this layer should be updated in the backward pass as well.

        :param dA: The derivative of the activation of the current layer with respect to the loss
        :returns: The derivative of the activation of the previous layer with respect to the loss
        """
        if self._W is None or self._b is None:
            raise ValueError("Layer parameters needs to initialized first")

        if self._A_prev is None:
            raise ValueError(
                "backward propagation can only be performed after one pass of forward propagation"
            )
        if (
            len(dA.shape) != 2
            or dA.shape[0] != self._W.shape[0]
            or dA.shape[1] != self._A_prev.shape[1]
        ):
            raise ValueError(
                "Provided derivatives tensor of invalid shape, expected: ("
                + str(self._W.shape[0])
                + ", "
                + str(self._A_prev.shape[1])
                + ") provided: "
                + str(dA.shape)
            )

        dZ = dA * self._activation.derivative()
        dW = np.dot(dZ, self._A_prev.T) / self._A_prev.shape[1]
        db = np.sum(dZ, axis=1, keepdims=True) / self._A_prev.shape[1]
        dA_prev = np.dot(self._W.T, dZ)

        self._update_params(dW, db)
        self._A_prev = None
        return dA_prev

    @property
    def size(self):
        return self._size

    @property
    def parameters(self):
        return (self._W, self._b)

    @parameters.setter
    def parameters(self, new_params):
        if self._W is None or self._W.shape == new_params[0].shape:
            self._W = new_params[0]
        else:
            raise ValueError("New shape of W need to be equal to the previous shape")

        if self._b is None or self._b.shape == new_params[1].shape:
            self._b = new_params[1]
        else:
            raise ValueError("New shape of b need to be equal to the previous shape")

    def _update_params(self, dW: ArrayLike, db: ArrayLike) -> None:
        """
        Update the parameters of the current layer.

        :param dW: The derivative of the weights of this layer with respect to the loss
        :param db: The derivative of the biases of this layer with respect to the loss
        """
        self._W = self._W - self._learning_rate * dW
        self._b = self._b - self._learning_rate * db
