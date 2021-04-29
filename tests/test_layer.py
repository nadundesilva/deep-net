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

from typing import Tuple
from deep_net.layer import Layer
from deep_net.initializers import Initializer
from deep_net.activations import Activation
from numpy.typing import ArrayLike
import numpy as np


def test_layer_forward_propagation():
    class MockInitializer(Initializer):
        def init_tensor(self, shape: Tuple) -> np.ndarray:
            assert shape[0] == 2 and (shape[1] == 3 or shape[1] == 1)
            if shape[1] == 3:
                return np.array([[1, 2, 3], [4, 5, 6]])
            else:
                return np.array([[7], [8]])

    class MockActivation(Activation):
        def map(self, Z: ArrayLike) -> ArrayLike:
            expected_Z = [[81], [190]]
            np.testing.assert_array_equal(Z, expected_Z)
            return 2 * Z + 1

        def derivative(self) -> ArrayLike:
            return np.array([[0.3], [0.2]])

    A_prev = np.array([[11], [12], [13]])
    expected_A = np.array([[163], [381]])
    expected_dA_prev = np.array([[4.075], [5.15], [6.225]])

    layer = Layer(2, 0.01, lambda: MockActivation())
    layer.init_parameters(3, MockInitializer())

    A = layer.propagate_forward(A_prev)
    np.testing.assert_array_equal(A, expected_A)

    dA = np.array([[0.25], [5]])
    dA_prev = layer.propagate_backward(dA)
    np.testing.assert_array_equal(dA_prev, expected_dA_prev)
