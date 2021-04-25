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

from typing import Callable
import numpy as np


def visit_multi_dimensional_array(matrix: np.ndarray, visitItem: Callable[[any], None]):
    visit_multi_dimensional_array_pair(
        matrix, None, lambda itemA, itemB: visitItem(itemA)
    )


def visit_multi_dimensional_array_pair(
    matrixA: np.ndarray, matrixB: np.ndarray, visitItem: Callable[[any, any], None]
):
    if matrixA is not None and matrixB is not None and matrixA.shape != matrixB.shape:
        raise Exception(
            "Shapes of the two matrices "
            + str(matrixA.shape)
            + ", "
            + str(matrixB.shape)
            + " not equal"
        )
    if len(matrixA.shape) > 1:
        for i in range(len(matrixA)):
            visit_multi_dimensional_array_pair(
                matrixA[i], None if matrixB is None else matrixB[i], visitItem
            )
    else:
        for i in range(len(matrixA)):
            visitItem(matrixA[i], None if matrixB is None else matrixB[i])


def assertEqual(itemA, itemB):
    assert itemA == itemB
