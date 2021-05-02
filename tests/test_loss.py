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

from deep_net.loss import LossFunc, BinaryCrossEntropy
import numpy as np
import pytest

test_data = [
    [
        BinaryCrossEntropy(),
        np.array([[1, 0, 0, 1, 1]]),
        np.array([[0.921, 0.213, 0.31, 0.49, 0.57]]),
        np.array([[1.4798821911489495]]),
        np.array(
            [
                [
                    -0.21715526601520088,
                    0.25412960609911056,
                    0.2898550724637682,
                    -0.40816326530612246,
                    -0.3508771929824562,
                ]
            ]
        ),
    ]
]


def test_loss_interface():
    loss = LossFunc()
    with pytest.raises(NotImplementedError) as e:
        loss.calculate(np.random.rand(6, 3), np.random.rand(6, 3))
    with pytest.raises(NotImplementedError) as e:
        loss.derivative(np.random.rand(6, 3), np.random.rand(6, 3))


@pytest.mark.parametrize(
    """loss_function, Y, Y_hat, expected_cost, expected_derivative""", test_data
)
def test_loss_functions(loss_function, Y, Y_hat, expected_cost, expected_derivative):
    cost = loss_function.calculate(Y, Y_hat)
    np.testing.assert_array_equal(cost, expected_cost)

    derivative = loss_function.derivative(Y, Y_hat)
    np.testing.assert_array_equal(derivative, expected_derivative)


@pytest.mark.parametrize(
    """loss_function, Y, Y_hat, expected_cost, expected_derivative""", test_data
)
def test_loss_with_mismatched_shapes(
    loss_function, Y, Y_hat, expected_cost, expected_derivative
):
    with pytest.raises(ValueError) as e:
        loss_function.calculate(np.random.rand(2, 3), np.random.rand(2, 4))
    assert (
        "ValueError('The shape of Y: (2, 3) is not equal to the shape of Y_hat: (2, 4)')"
        in str(e)
    )

    with pytest.raises(ValueError) as e:
        loss_function.derivative(np.random.rand(2, 5), np.random.rand(2, 6))
    assert (
        "ValueError('The shape of Y: (2, 5) is not equal to the shape of Y_hat: (2, 6)')"
        in str(e)
    )
