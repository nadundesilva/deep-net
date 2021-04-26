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
from typing import Callable
import numpy as np


def visit_tensor(tensor: ArrayLike, visitItem: Callable[[any], None]):
    visit_tensor_pair(tensor, None, lambda itemA, itemB: visitItem(itemA))


def visit_tensor_pair(
    tensorA: ArrayLike, tensorB: ArrayLike, visitItem: Callable[[any, any], None]
):
    if tensorA is not None and tensorB is not None and tensorA.shape != tensorB.shape:
        raise Exception(
            "Shapes of the two matrices "
            + str(tensorA.shape)
            + ", "
            + str(tensorB.shape)
            + " not equal"
        )
    if len(tensorA.shape) > 1:
        for i in range(len(tensorA)):
            visit_tensor_pair(
                tensorA[i], None if tensorB is None else tensorB[i], visitItem
            )
    else:
        for i in range(len(tensorA)):
            visitItem(tensorA[i], None if tensorB is None else tensorB[i])


def assertEqual(itemA, itemB):
    assert itemA == itemB
