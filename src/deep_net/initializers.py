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
    def init_multi_dimensional_array(shape: Tuple) -> np.ndarray:
        raise Exception("Not Implemented")


class Constant(Initializer):
    constant: float

    def __init__(self, constant: float):
        self.constant = constant

    def init_multi_dimensional_array(self, shape: Tuple) -> np.ndarray:
        return np.full(shape, self.constant, dtype=float)


class Random(Initializer):
    def init_multi_dimensional_array(self, shape: Tuple) -> np.ndarray:
        return np.random.random(shape).astype("float64")