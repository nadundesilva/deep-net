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

from unittest import TestCase
from deep_net.activations import Sigmoid, Relu, Tanh
import numpy as np
import tests.utils as utils
import pytest


activation_input = np.array([[4, 5.1, -100], [0, 14000.2, -1.3]])

additional_activation_input = np.array([[11, -31], [7.3, -13.1]])

test_data = [
    (
        Sigmoid(),
        np.array(
            [
                [0.9820137900379085, 0.9939401985084158, 3.7200759760208356e-44],
                [0.5, 1.0, 0.2141650169574414],
            ]
        ),
        np.array(
            [
                [0.017662706213291107, 0.006023080297466842, 3.7200759760208356e-44],
                [0.25, 0, 0.16829836246906024],
            ]
        ),
        np.array(
            [
                [0.999983298578152, 3.442477108469858e-14],
                [0.9993249172693672, 2.0452264415637373e-06],
            ]
        ),
        np.array(
            [
                [1.670114291046157e-05, 3.4424771084697395e-14],
                [0.0006746269939395714, 2.04522225861254e-06],
            ]
        ),
    ),
    (
        Relu(),
        np.array([[4, 5.1, 0], [0, 14000.2, 0]]),
        np.array([[1, 1, 0], [0, 1, 0]]),
        np.array([[11, 0], [7.3, 0]]),
        np.array([[1, 0], [1, 0]]),
    ),
    (
        Tanh(),
        np.array(
            [
                [0.999329299739067, 0.9999256621257943, -1.0],
                [0.0, 1.0, -0.8617231593133063],
            ]
        ),
        np.array(
            [
                [0.0013409506830258655, 0.0001486702222919245, 0],
                [1, 0, 0.25743319670309406],
            ]
        ),
        np.array([[0.9999999994421064, -1.0], [0.999999087295143, -0.999999999991634]]),
        np.array(
            [
                [1.115787240379973e-09, 0.0],
                [1.8254088810509828e-06, 1.673194915952081e-11],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    """activation, expected_map_output, expected_derivative_output,
    expected_additional_map_output, expected_additional_derivative_output""",
    test_data,
)
def test_activations(
    activation,
    expected_map_output,
    expected_derivative_output,
    expected_additional_map_output,
    expected_additional_derivative_output,
):
    map_output = activation.map(activation_input)
    assert map_output.shape == expected_map_output.shape
    utils.visit_multi_dimensional_array_pair(
        map_output, expected_map_output, utils.assertEqual
    )

    derivative_output = activation.derivative()
    assert derivative_output.shape == expected_derivative_output.shape
    utils.visit_multi_dimensional_array_pair(
        derivative_output, expected_derivative_output, utils.assertEqual
    )

    additional_map_output = activation.map(additional_activation_input)
    assert additional_map_output.shape == expected_additional_map_output.shape
    utils.visit_multi_dimensional_array_pair(
        additional_map_output, expected_additional_map_output, utils.assertEqual
    )

    additional_derivative_output = activation.derivative()
    assert (
        additional_derivative_output.shape
        == expected_additional_derivative_output.shape
    )
    utils.visit_multi_dimensional_array_pair(
        additional_derivative_output,
        expected_additional_derivative_output,
        utils.assertEqual,
    )


@pytest.mark.parametrize(
    """activation, expected_map_output, expected_derivative_output,
    expected_additional_map_output, expected_additional_derivative_output""",
    test_data,
)
def test_activation_derivative_without_map(
    activation,
    expected_map_output,
    expected_derivative_output,
    expected_additional_map_output,
    expected_additional_derivative_output,
):
    with pytest.raises(ValueError) as e:
        activation.derivative()
    assert "ValueError('map should be called before derivate')" in str(e)


@pytest.mark.parametrize(
    """activation, expected_map_output, expected_derivative_output,
    expected_additional_map_output, expected_additional_derivative_output""",
    test_data,
)
def test_activation_repeated_derivative(
    activation,
    expected_map_output,
    expected_derivative_output,
    expected_additional_map_output,
    expected_additional_derivative_output,
):
    activation.map(activation_input)
    activation.derivative()
    with pytest.raises(ValueError) as e:
        activation.derivative()
    assert "ValueError('map should be called before derivate')" in str(e)
