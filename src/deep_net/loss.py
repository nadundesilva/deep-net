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

from deep_net import EPSILON
from numpy.typing import ArrayLike
import numpy as np


class LossFunc:
    """
    Abstract interface related to loss.
    """

    def calculate(self, Y: ArrayLike, Y_hat: ArrayLike):
        """
        Calcualte the loss.

        :param Y: The actual Y values tensor
        :param Y_hat: The predicted Y values tensor
        """
        raise NotImplementedError()

    def derivative(self, Y: ArrayLike, Y_hat: ArrayLike):
        """
        Calcualte the derivative of loss.

        :param Y: The actual Y values tensor
        :param Y_hat: The predicted Y values tensor
        """
        raise NotImplementedError()


class BinaryCrossEntropy(LossFunc):
    """
    Loss function for calculating the Binary Cross Entropy.
    """

    def calculate(self, Y: ArrayLike, Y_hat: ArrayLike) -> ArrayLike:
        if Y.shape != Y_hat.shape:
            raise ValueError(
                "The shape of Y: "
                + str(Y.shape)
                + " is not equal to the shape of Y_hat: "
                + str(Y_hat.shape)
            )

        batch_size = Y.shape[1]
        return -(
            np.dot(Y, np.log(Y_hat).T)
            + np.dot((1 - Y + EPSILON), np.log(1 - Y_hat + EPSILON).T) / batch_size
        )

    def derivative(self, Y: ArrayLike, Y_hat: ArrayLike) -> ArrayLike:
        if Y.shape != Y_hat.shape:
            raise ValueError(
                "The shape of Y: "
                + str(Y.shape)
                + " is not equal to the shape of Y_hat: "
                + str(Y_hat.shape)
            )

        batch_size = Y.shape[1]
        return ((Y_hat - Y) / (Y_hat * (1 - Y_hat) + EPSILON)) / batch_size
