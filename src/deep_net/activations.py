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

import numpy as np


class Activation:
    def propagate_forward(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def propagate_backward(self) -> np.ndarray:
        raise NotImplementedError()


class Sigmoid(Activation):
    def propagate_forward(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))


class Relu(Activation):
    def propagate_forward(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(Z, np.zeros(Z.shape))


class Tanh(Activation):
    def propagate_forward(self, Z: np.ndarray) -> np.ndarray:
        return np.tanh(Z)
