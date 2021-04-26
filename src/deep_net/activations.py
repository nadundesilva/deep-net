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

from numpy.typing import ArrayLike
import numpy as np


class Activation:
    def map(self, Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError()

    def derivative(self) -> ArrayLike:
        raise NotImplementedError()


class Sigmoid(Activation):
    _A: ArrayLike = None

    def map(self, Z: ArrayLike) -> ArrayLike:
        self._A = 1 / (1 + np.exp(-Z))
        return self._A

    def derivative(self) -> ArrayLike:
        if self._A is None:
            raise ValueError("map should be called before derivate")
        dA = self._A * (1 - self._A)
        self._A = None
        return dA


class Relu(Activation):
    _Z: ArrayLike = None

    def map(self, Z: ArrayLike) -> ArrayLike:
        self._Z = Z
        return np.maximum(Z, np.zeros(Z.shape))

    def derivative(self) -> ArrayLike:
        if self._Z is None:
            raise ValueError("map should be called before derivate")
        dA = np.where(self._Z > 0, np.ones(self._Z.shape), np.zeros(self._Z.shape))
        self._Z = None
        return dA


class Tanh(Activation):
    _A: ArrayLike = None

    def map(self, Z: ArrayLike) -> ArrayLike:
        self._A = np.tanh(Z)
        return self._A

    def derivative(self) -> ArrayLike:
        if self._A is None:
            raise ValueError("map should be called before derivate")
        dA = 1 - self._A ** 2
        self._A = None
        return dA
