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

from typing import Generator, Tuple
from numpy.typing import ArrayLike
from tests import TESTS_SHOW_PLOTS
from deep_net.model import Model
from deep_net.layer import Layer
from deep_net.activations import ReLU, Sigmoid
from deep_net.initializers import Random, Constant
from deep_net.loss import BinaryCrossEntropy
from sklearn import datasets
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def test_model():
    # Generate data sets
    data_set = datasets.make_blobs(
        n_samples=10000,
        n_features=10,
        centers=2,
        random_state=5,
    )
    data_set = (data_set[0], data_set[1].reshape((-1, 1)))

    data_set_size = data_set[0].shape[0]
    feature_count = data_set[0].shape[1]
    batch_size = 100
    epoch_count = 1000

    sigmoid_activation_generator = lambda: Sigmoid()
    relu_activation_generator = lambda: ReLU()

    random_initializer = Random()
    zeros_initializer = Constant(0)

    layers = [
        Layer(
            7,
            1e-2,
            relu_activation_generator,
            random_initializer,
            zeros_initializer,
        ),
        Layer(
            4,
            1e-2,
            relu_activation_generator,
            random_initializer,
            zeros_initializer,
        ),
        Layer(
            1,
            1e-2,
            sigmoid_activation_generator,
            random_initializer,
            zeros_initializer,
        ),
    ]

    def get_data_batches() -> Generator[Tuple[ArrayLike, ArrayLike], None, None]:
        for i in range(math.ceil(data_set_size / batch_size)):
            X_batch = data_set[0][
                i * batch_size : min(data_set_size, (i + 1) * batch_size), :
            ]
            Y_batch = data_set[1][
                i * batch_size : min(data_set_size, (i + 1) * batch_size), :
            ]
            yield (X_batch.T, Y_batch.T)

    model = Model(feature_count, layers, BinaryCrossEntropy())
    costs = model.fit(get_data_batches, epoch_count)

    if TESTS_SHOW_PLOTS:
        graph_points_distance = int(epoch_count / 100)
        line_plot = sns.lineplot(
            x=[x for x in range(len(costs))][::graph_points_distance],
            y=costs[::graph_points_distance],
        )
        line_plot.set(xlabel="Epoch", ylabel="Loss")
        plt.title("Loss")
        plt.show()

    assert costs[-1] == 0.0298282125462872
