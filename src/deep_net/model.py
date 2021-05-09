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

from typing import Callable, Generator, List, Tuple
from deep_net.layer import Layer
from deep_net.loss import LossFunc
from numpy.typing import ArrayLike
import numpy as np


class Model:
    _layers: List[Layer]
    _loss_function: LossFunc

    def __init__(self, input_size: int, layers: List[Layer], loss_function: LossFunc):
        self._layers = layers
        self._loss_function = loss_function

        prev_layer_size = input_size
        for layer in self._layers:
            layer.init_parameters(prev_layer_size)
            prev_layer_size = layer.size

    def fit(
        self,
        data_batches: Callable[[], Generator[Tuple[ArrayLike, ArrayLike], None, None]],
        epoch_count: int,
    ) -> List[float]:
        costs = []
        for i in range(epoch_count):
            epoch_costs = []
            for X_batch, Y_batch in data_batches():
                # Forward propagation
                A_prev = X_batch
                for layer in self._layers:
                    A_prev = layer.propagate_forward(A_prev)
                Y_hat = A_prev

                # Calculate loss
                cost = float(self._loss_function.calculate(Y_batch, Y_hat))
                epoch_costs.append(cost)

                # Backward propagation
                dA = self._loss_function.derivative(Y_batch, Y_hat)
                for layer in reversed(self._layers):
                    dA = layer.propagate_backward(dA)

            mean_cost = sum(epoch_costs) / len(epoch_costs)
            if i % (epoch_count / 10) == 0 or i == epoch_count - 1:
                print("Epoch: " + str(i + 1) + " Cost: " + str(mean_cost))
            costs.append(mean_cost)
        return costs
