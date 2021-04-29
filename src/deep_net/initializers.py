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
import numpy as np


class Initializer:
    """
    Abstract initializer which should be initialized by all tensor initializers.
    """

    def init_tensor(shape: Tuple) -> np.ndarray:
        """
        Initialize a tensor which is to be used as parameters.

        :param shape: The shape of the tensor to be initialized
        :returns: The initialized tensor
        """
        raise NotImplementedError()


class Constant(Initializer):
    """
    Initializer capable of initializing a tensor setting all elements to the constant.
    """

    _constant: float

    def __init__(self, constant: float):
        """
        Initialize the initializer.

        :param constant: The constant to be filled into the tensor
        """
        self._constant = constant

    def init_tensor(self, shape: Tuple) -> np.ndarray:
        return np.full(shape, self._constant, dtype=float)


class Random(Initializer):
    """
    Initializer capable of initializing a tensor setting all elements to random values.
    """

    def init_tensor(self, shape: Tuple) -> np.ndarray:
        return np.random.random(shape).astype("float64")
