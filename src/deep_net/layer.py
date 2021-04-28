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
    _size: int
    _batch_size: int
    _learning_rate: float
    _activation: Activation

    # Optimized paramaeters
    _W: ArrayLike
    _b: ArrayLike

    # Cached values (stored after forward pass and cleared after backward pass)
    _A_prev: ArrayLike

    def __init__(
        self,
        size: int,
        learning_rate: float,
        create_activation: Callable[[], Activation],
    ):
        self._size = size
        self._learning_rate = learning_rate
        self._activation = create_activation()

    def init_parameters(self, prev_layer_size: int, initializer: Initializer) -> None:
        self._batch_size = 1
        self._W = initializer.init_tensor((self._size, prev_layer_size))
        self._b = initializer.init_tensor((self._size, self._batch_size))

    def propagate_forward(self, A_prev: ArrayLike) -> ArrayLike:
        self._A_prev = A_prev
        Z = np.dot(self._W, A_prev) + self._b
        return self._activation.map(Z)

    def propagate_backward(self, dA) -> ArrayLike:
        dZ = dA * self._activation.derivative()
        dW = np.dot(dZ, self._A_prev.T) / self._batch_size
        db = np.sum(dZ, axis=1, keepdims=True) / self._batch_size
        dA_prev = np.dot(self._W.T, dZ)

        self._update_params(dW, db)
        self._A_prev = None
        return dA_prev

    def _update_params(self, dW, db) -> None:
        self._W = self._W - self._learning_rate * dW
        self._b = self._b - self._learning_rate * db
