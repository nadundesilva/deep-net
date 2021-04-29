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

from numpy.typing import ArrayLike
import numpy as np


class Activation:
    """
    Abstract activation function which should be inherited by all the activation implementations.
    """

    def map(self, Z: ArrayLike) -> ArrayLike:
        """
        Map the input tensor to an output by applying the activation for each element in the tensor.

        :param Z: The input tensor
        :returns: The output tensor with the activation applied to each element
        """
        raise NotImplementedError()

    def derivative(self) -> ArrayLike:
        """
        Get the derivative of each element as a tensor.

        :returns: The output tensor with the the derivative of the activation applied to each element
        """
        raise NotImplementedError()


class Sigmoid(Activation):
    """
    Sigmoid activation function.

    sigmoid(x) = 1 / (1 + e ** -x)
    sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
    """

    _A: ArrayLike = None

    def map(self, Z: ArrayLike) -> ArrayLike:
        self._A = 1 / (1 + np.exp(-Z))
        return self._A

    def derivative(self) -> ArrayLike:
        if self._A is None:
            raise ValueError("map should be called before derivate")
        dA = self._A * (1 - self._A)
        self._A = None
        return dA


class ReLU(Activation):
    """
    Rectified Linear Unit activation function.

    relu(x) = max(x, 0)
    relu_derivative(x) = 1 if x > 0, 0 if x < 0
    """

    _Z: ArrayLike = None

    def map(self, Z: ArrayLike) -> ArrayLike:
        self._Z = Z
        return np.maximum(self._Z, np.zeros(self._Z.shape))

    def derivative(self) -> ArrayLike:
        if self._Z is None:
            raise ValueError("map should be called before derivate")
        dA = np.where(self._Z > 0, np.ones(self._Z.shape), np.zeros(self._Z.shape))
        self._Z = None
        return dA


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unity activation function.

    leaky_relu(x) = max(x, alpha * x)
    leaky_relu_derivative(x) = 1 if x > 0, 0.01 if x < 0
    """

    _alpha: float
    _Z: ArrayLike = None

    def __init__(self, alpha: int = 0.01):
        """
        Initialize the activation.

        :param alpha: The gradient of the function when the input is lower than zero
        """
        self._alpha = alpha

    def map(self, Z: ArrayLike) -> ArrayLike:
        self._Z = Z
        return np.maximum(self._Z, self._Z * self._alpha)

    def derivative(self) -> ArrayLike:
        if self._Z is None:
            raise ValueError("map should be called before derivate")
        dA = np.where(
            self._Z > 0, np.ones(self._Z.shape), np.full(self._Z.shape, self._alpha)
        )
        self._Z = None
        return dA


class Tanh(Activation):
    """
    Hyperbolic Tangent activation function.

    tanh(x) = (e ** x - e ** -x) / (e ** x + e ** -x)
    tanh_derivative(x) = (1 - tanh(x) ** 2)
    """

    _A: ArrayLike = None

    def map(self, Z: ArrayLike) -> ArrayLike:
        self._A = np.tanh(Z)
        return self._A

    def derivative(self) -> ArrayLike:
        if self._A is None:
            raise ValueError("map should be called before derivate")
        dA = 1 - self._A ** 2
        self._A = None
        return dA
