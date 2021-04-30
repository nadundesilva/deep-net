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
import pytest


test_data = [
    (
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[7], [8]]),
        np.array([[11], [12], [13]]),
        np.array([[81], [190]]),
        np.array([[163], [381]]),
        np.array([[0.3], [0.2]]),
        np.array([[4.075], [5.15], [6.225]]),
    ),
    (
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[7], [8]]),
        np.array([[11, 14, 17, 20], [12, 15, 18, 21], [13, 16, 19, 22]]),
        np.array([[81, 99, 117, 135], [190, 235, 280, 325]]),
        np.array([[163, 199, 235, 271], [381, 471, 561, 651]]),
        np.array([[0.3, 1.2, 0.007, 1.002], [0.2, 0.3, 0.4, 0.1]]),
        np.array(
            [
                [4.075, 6.3, 8.00175, 2.2505],
                [5.15, 8.1, 10.0035, 3.001],
                [6.225, 9.9, 12.00525, 3.7515],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    """W_init, b_init, A_prev, expected_Z, expected_A, activation_derivative, expected_dA_prev""",
    test_data,
)
def test_layer_forward_propagation(
    W_init: ArrayLike,
    b_init: ArrayLike,
    A_prev: ArrayLike,
    expected_Z: ArrayLike,
    expected_A: ArrayLike,
    activation_derivative: ArrayLike,
    expected_dA_prev: ArrayLike,
):
    batch_size = len(b_init[0])
    prev_layer_size = len(W_init)
    layer_size = len(W_init[0])

    class MockInitializer(Initializer):
        def init_tensor(self, shape: Tuple) -> np.ndarray:
            assert shape[0] == prev_layer_size and (
                shape[1] == layer_size or shape[1] == batch_size
            )
            if shape[1] == layer_size:
                return W_init
            else:
                return b_init

    class MockActivation(Activation):
        def map(self, Z: ArrayLike) -> ArrayLike:
            np.testing.assert_array_equal(Z, expected_Z)
            return 2 * Z + 1

        def derivative(self) -> ArrayLike:
            return activation_derivative

    layer = Layer(2, 0.01, lambda: MockActivation())
    layer.init_parameters(3, MockInitializer())

    A = layer.propagate_forward(A_prev)
    np.testing.assert_array_equal(A, expected_A)

    dA = np.array([[0.25], [5]])
    dA_prev = layer.propagate_backward(dA)
    np.testing.assert_array_equal(dA_prev, expected_dA_prev)
