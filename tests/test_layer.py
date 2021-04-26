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

    A_prev = np.array([[11], [12], [13]])
    expected_A = np.array([[81], [190]])

    layer = Layer(2)
    layer.init_parameters(3, MockInitializer())

    A = layer.activate(A_prev)
    np.testing.assert_array_equal(A, expected_A)
