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
from deep_net.activations import sigmoid
import numpy as np
import tests.utils as utils


class ActivationsTestCase(TestCase):
    def test_sigmoid_activation(self):
        activation_input = np.array([[4, 5.1, -100], [0, 14000.2, -1.3]])
        expected_output = np.array(
            [
                [0.9820137900379085, 0.9939401985084158, 3.7200759760208356e-44],
                [0.5, 1.0, 0.2141650169574414],
            ]
        )

        activation_output = sigmoid(activation_input)
        utils.visit_multi_dimensional_array_pair(
            activation_output,
            expected_output,
            lambda itemA, itemB: self.assertEqual(itemA, itemB),
        )
