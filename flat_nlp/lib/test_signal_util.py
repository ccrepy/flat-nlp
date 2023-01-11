# Copyright 2022 Flat NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test utility for signal library."""

import types
import numpy as np

from flat_nlp.lib import np_signal_util


def _convert_namespace_to_hc(namespace):
  return types.SimpleNamespace(**{
      k: np_signal_util.hc(np.array([v]))[0]
      for k, v in namespace.__dict__.items()
  })


TEST_TENSOR = np.array([
    [
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 2., 0., 0., 0., 0., 0., 0.],
    ],
    [
        [1., 2., 0., 0., 0., 0., 0., 0.],
        [0., 0., 3., 4., 0., 0., 0., 0.],
    ],
])

TEST_STRIDED_TENSOR = np.array([
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.],
    [1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 3., 4., 0., 0., 0., 0.],
])

TEST_ENERGY_NORMALIZED_TENSOR = np.array([
    [
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.],
    ],
    [
        [1. / 2.23, 2. / 2.23, 0., 0., 0., 0., 0., 0.],  # sqrt(1+4) ~= 2.23
        [0., 0., 3. / 5., 4. / 5., 0., 0., 0., 0.],  # sqrt(9+16) == 5
    ],
])

TEST_RFFT_TENSOR = np.array([
    [
        [1. + 0.j, 1. + 0.j, 1. - 0.j, 1. + 0.j, 1. + 0.j],
        [2. + 0.j, 1.41 - 1.41j, 0. - 2.j, -1.41 - 1.41j, -2. + 0.j],
    ],
    [
        [3. + 0.j, 2.41 - 1.41j, 1. - 2.j, -0.41 - 1.41j, -1. + 0.j],
        [7. + 0.j, -2.83 - 5.83j, -3. + 4.j, 2.83 + 0.17j, -1. + 0.j],
    ],
])

TEST_HC_TENSOR = np.array([
    [
        [1., 1., 1., 1., 1., 0., -0., 0.],
        [2., 1.41, 0., -1.41, -2., -1.41, -2., -1.41],
    ],
    [
        [3., 2.41, 1., -0.41, -1., -1.41, -2., -1.41],
        [7., -2.83, -3., 2.83, -1., 0.17, 4., -5.83],
    ],
])

TESTSET_SIGNAL_CONVOLUTION = types.SimpleNamespace(
    s1=np.array([[1, 2, 0, 0]]),
    s2=np.array([[0, 0, 1, 0]]),
)
TESTSET_HC_SIGNAL_CONVOLUTION = _convert_namespace_to_hc(
    TESTSET_SIGNAL_CONVOLUTION)

TESTSET_SIGNAL_DISTANCE = types.SimpleNamespace(
    s1=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
    s2=np.array([[0, 0, 1, 0], [0, 1, 0, 0]]),
    s3=np.array([[1, 2, 3, 0], [2, 3, 4, 0]]),
)
TESTSET_HC_SIGNAL_DISTANCE = _convert_namespace_to_hc(TESTSET_SIGNAL_DISTANCE)
