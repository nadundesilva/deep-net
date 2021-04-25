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
from deep_net.initializers import Constant, Random
from typing import Tuple, Callable
import numpy as np
import tests.utils as utils


test_shapes = [
    tuple([np.random.randint(1, 100) for y in range(x)]) for x in range(1, 3)
]


def test_constant_initializer():
    test_constants = [-13, 0, 57.5]
    for test_constant in test_constants:
        for shape in test_shapes:
            initializer = Constant(test_constant)
            matrix = initializer.init_matrix(shape)

            assert matrix.shape == shape
            utils.visit_multi_dimensional_array(
                matrix, lambda item: utils.assertEqual(item, test_constant)
            )


def test_random_initializer():
    for shape in test_shapes:
        initializer = Random()
        matrix = initializer.init_matrix(shape)

        assert matrix.shape == shape
        utils.visit_multi_dimensional_array(
            matrix, lambda item: utils.assertEqual(type(item), np.float64)
        )
