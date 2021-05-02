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

from deep_net.loss import Loss
import numpy as np
import pytest


def test_loss_interface():
    loss = Loss()
    with pytest.raises(NotImplementedError) as e:
        loss.calculate(np.random.rand(6, 3), np.random.rand(6, 3))
    with pytest.raises(NotImplementedError) as e:
        loss.derivative(np.random.rand(6, 3), np.random.rand(6, 3))
