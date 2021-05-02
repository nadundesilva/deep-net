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
from deep_net.initializers import Initializer, Constant
from deep_net.activations import Activation, ReLU
from numpy.typing import ArrayLike
import numpy as np
import pytest


test_data = [
    (
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[7], [8]]),
        np.array([[11, 14, 17, 20, 23], [12, 15, 18, 21, 24], [13, 16, 19, 22, 25]]),
        np.array([[81, 99, 117, 135, 153], [190, 235, 280, 325, 370]]),
        np.array([[163, 199, 235, 271, 307], [381, 471, 561, 651, 741]]),
        np.array([[0.25, 1, 1, 1, 1], [5, 1, 1, 1, 1]]),
        np.array(
            [[0.3, 0.112, 1.22, 0.002, 0.973], [0.2, 2.311, 0.113, 0.002, 0.8711]]
        ),
        np.array(
            [
                [4.075, 9.356, 1.672, 0.01, 4.4574],
                [5.15, 11.779, 3.005, 0.014, 6.3015],
                [6.225, 14.202, 4.338, 0.018000000000000002, 8.1456],
            ]
        ),
        np.array([[0.908896, 1.904132, 2.899368], [3.8692994, 4.8607052, 5.852111]]),
        np.array([[6.995236], [7.9914058]]),
    ),
    (
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[7], [8]]),
        np.array([[11, 14, 17, 20], [12, 15, 18, 21], [13, 16, 19, 22]]),
        np.array([[81, 99, 117, 135], [190, 235, 280, 325]]),
        np.array([[163, 199, 235, 271], [381, 471, 561, 651]]),
        np.array([[0.25, 1.23, 0.01, 0.003], [5.11, 6.123, 5.463, 2.52]]),
        np.array([[0.3, 1.2, 0.007, 1.002], [0.2, 0.3, 0.4, 0.1]]),
        np.array(
            [
                [4.163, 8.823599999999999, 8.74087, 1.011006],
                [5.26, 12.1365, 10.92614, 1.266012],
                [6.357, 15.4494, 13.11141, 1.521018],
            ]
        ),
        np.array(
            [[0.946124225, 1.942239035, 2.938353845], [3.8021325, 4.78889225, 5.775652]]
        ),
        np.array([[6.99611481], [7.98675975]]),
    ),
]


@pytest.mark.parametrize(
    """W_init, b_init, A_prev, expected_Z, expected_A, dA, activation_derivative, expected_dA_prev,
    W_optimized, b_optimized""",
    test_data,
)
def test_layer_forward_propagation(
    W_init: ArrayLike,
    b_init: ArrayLike,
    A_prev: ArrayLike,
    expected_Z: ArrayLike,
    expected_A: ArrayLike,
    dA: ArrayLike,
    activation_derivative: ArrayLike,
    expected_dA_prev: ArrayLike,
    W_optimized: ArrayLike,
    b_optimized: ArrayLike,
):
    batch_size = b_init.shape[1]
    current_layer_size = W_init.shape[0]
    prev_layer_size = W_init.shape[1]

    class MockWeightsInitializer(Initializer):
        def init_tensor(self, shape: Tuple) -> np.ndarray:
            assert shape[0] == current_layer_size and shape[1] == prev_layer_size
            return W_init

    class MockBiasInitializer(Initializer):
        def init_tensor(self, shape: Tuple) -> np.ndarray:
            assert shape[0] == current_layer_size and shape[1] == 1
            return b_init

    class MockActivation(Activation):
        def map(self, Z: ArrayLike) -> ArrayLike:
            np.testing.assert_array_equal(Z, expected_Z)
            return 2 * Z + 1

        def derivative(self) -> ArrayLike:
            return activation_derivative

    layer = Layer(
        current_layer_size,
        0.01,
        lambda: MockActivation(),
        MockWeightsInitializer(),
        MockBiasInitializer(),
    )
    layer.init_parameters(prev_layer_size)

    A = layer.propagate_forward(A_prev)
    np.testing.assert_array_equal(A, expected_A)

    dA_prev = layer.propagate_backward(dA)
    np.testing.assert_array_equal(dA_prev, expected_dA_prev)

    assert layer.size == 2
    np.testing.assert_array_equal(layer.parameters[0], W_optimized)
    np.testing.assert_array_equal(layer.parameters[1], b_optimized)


def test_updating_layer_params():
    layer = Layer(5, 0.002, lambda: ReLU(), Constant(13), Constant(13))
    layer.init_parameters(7)

    assert layer.size == 5
    np.testing.assert_array_equal(layer.parameters[0], np.full((5, 7), 13))
    np.testing.assert_array_equal(layer.parameters[1], np.full((5, 1), 13))

    layer.parameters = (np.full((5, 7), 1), np.full((5, 1), 2))

    assert layer.size == 5
    np.testing.assert_array_equal(layer.parameters[0], np.full((5, 7), 1))
    np.testing.assert_array_equal(layer.parameters[1], np.full((5, 1), 2))


def test_updating_layer_params_with_invalid_shapes():
    layer = Layer(5, 0.002, lambda: ReLU(), Constant(13), Constant(13))
    layer.init_parameters(7)

    assert layer.size == 5
    np.testing.assert_array_equal(layer.parameters[0], np.full((5, 7), 13))
    np.testing.assert_array_equal(layer.parameters[1], np.full((5, 1), 13))

    with pytest.raises(ValueError) as e:
        layer.parameters = (np.full((5, 6), 1), np.full((5, 1), 2))
    assert "ValueError('New shape of W need to be equal to the previous shape')" in str(
        e
    )

    with pytest.raises(ValueError) as e:
        layer.parameters = (np.full((5, 7), 1), np.full((5, 2), 2))
    assert "ValueError('New shape of b need to be equal to the previous shape')" in str(
        e
    )


def test_setting_layer_params():
    layer = Layer(5, 0.002, lambda: ReLU(), Constant(13), Constant(13))
    layer.parameters = (np.full((5, 7), 1), np.full((5, 1), 2))
    layer.init_parameters(7)

    assert layer.size == 5
    np.testing.assert_array_equal(layer.parameters[0], np.full((5, 7), 1))
    np.testing.assert_array_equal(layer.parameters[1], np.full((5, 1), 2))


def test_setting_layer_params_with_invalid_W_shape():
    layer = Layer(5, 0.002, lambda: ReLU(), Constant(13), Constant(13))
    layer.parameters = (np.full((5, 6), 1), np.full((5, 1), 2))

    with pytest.raises(ValueError) as e:
        layer.init_parameters(7)
    assert (
        "ValueError('Shape of the preset W does not match the expected shape')"
        in str(e)
    )


def test_setting_layer_params_with_invalid_b_shape():
    layer = Layer(5, 0.002, lambda: ReLU(), Constant(13), Constant(13))
    layer.parameters = (np.full((5, 7), 1), np.full((5, 3), 2))

    with pytest.raises(ValueError) as e:
        layer.init_parameters(7)
    assert (
        "ValueError('Shape of the preset b does not match the expected shape')"
        in str(e)
    )


def test_propagation_without_param_init():
    layer = Layer(5, 0.002, lambda: ReLU(), Constant(13), Constant(13))

    with pytest.raises(ValueError) as e:
        layer.propagate_forward(np.array([[12], [14], [13.2]]))
    assert "ValueError('Layer parameters needs to initialized first')" in str(e)

    with pytest.raises(ValueError) as e:
        layer.propagate_backward(np.array([[0.84], [0.14], [0.342]]))
    assert "ValueError('Layer parameters needs to initialized first')" in str(e)


def test_back_propagation_without_forward_propagation():
    layer = Layer(5, 0.002, lambda: ReLU(), Constant(13), Constant(13))
    layer.init_parameters(3)

    with pytest.raises(ValueError) as e:
        layer.propagate_backward(np.array([[0.25], [5], [1.2]]))
    assert (
        "ValueError('backward propagation can only be performed after one pass of forward propagation')"
        in str(e)
    )


def test_forward_propagation_with_invalid_shape():
    layer = Layer(7, 0.002, lambda: ReLU(), Constant(13), Constant(13))
    layer.init_parameters(3)

    with pytest.raises(ValueError) as e:
        layer.propagate_forward(np.full((3, 6, 11), 12))
    assert (
        "ValueError('Provided data tensor of invalid shape, expected: (3, mini_batch_size) provided: (3, 6, 11)')"
        in str(e)
    )

    with pytest.raises(ValueError) as e:
        layer.propagate_forward(np.full((4, 6), 12))
    assert (
        "ValueError('Provided data tensor of invalid shape, expected: (3, mini_batch_size) provided: (4, 6)')"
        in str(e)
    )


def test_backward_propagation_with_invalid_shape():
    layer = Layer(7, 0.002, lambda: ReLU(), Constant(13), Constant(13))
    layer.init_parameters(3)
    layer.propagate_forward(np.full((3, 6), 12))

    with pytest.raises(ValueError) as e:
        layer.propagate_backward(np.full((7, 6, 11), 12))
    assert (
        "ValueError('Provided derivatives tensor of invalid shape, expected: (7, 6) provided: (7, 6, 11)')"
        in str(e)
    )

    with pytest.raises(ValueError) as e:
        layer.propagate_backward(np.full((8, 6), 12))
    assert (
        "ValueError('Provided derivatives tensor of invalid shape, expected: (7, 6) provided: (8, 6)')"
        in str(e)
    )

    with pytest.raises(ValueError) as e:
        layer.propagate_backward(np.full((7, 7), 12))
    assert (
        "ValueError('Provided derivatives tensor of invalid shape, expected: (7, 6) provided: (7, 7)')"
        in str(e)
    )
