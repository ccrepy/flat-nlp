# Copyright 2023 Flat NLP Authors.
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

"""Tests for fftw_distance."""

import numpy as np

from flat_nlp.lib import np_distance
from flat_nlp.lib.python import fftw_distance

from absl.testing import absltest
from absl.testing import parameterized


class FftwDistanceTest(parameterized.TestCase):

  @parameterized.parameters(
      (fftw_distance.StridedFlatDistanceFn,),
      (fftw_distance.StridedHalfComplexFlatDistanceFn,),
  )
  def test_numpy_type(self, FftwDistanceFn):  # pylint: disable=invalid-name
    distance_fn = FftwDistanceFn(1, 2)

    with self.assertRaisesRegex(TypeError, 'expected numpy.double data'):
      distance_fn(np.array([1, 2]), np.array([1, 2]))

    # constructed type
    try:
      distance_fn(
          np.array([1, 2], dtype=np.float64), np.array([1, 2],
                                                       dtype=np.float64))
    except TypeError as e:
      self.fail(e)

    # casted type
    try:
      distance_fn(
          np.array([1, 2]).astype(np.float64),
          np.array([1, 2]).astype(np.float64))
    except TypeError as e:
      self.fail(e)

  @parameterized.parameters(
      (fftw_distance.StridedFlatDistanceFn, np_distance.StridedFlatDistanceFn),
      (fftw_distance.StridedHalfComplexFlatDistanceFn,
       np_distance.StridedHalfComplexFlatDistanceFn),
  )
  def test_flat_distance(self, FftwDistanceFn, NumpyDistanceFn):  # pylint: disable=invalid-name
    signal_dim, signal_len = 4, 8
    distance_fn = FftwDistanceFn(signal_dim, signal_len)

    a = np.random.rand(signal_dim, signal_len).astype(np.float64).ravel()
    b = np.random.rand(signal_dim, signal_len).astype(np.float64).ravel()

    self.assertAlmostEqual(distance_fn(a, a), 0.)

    self.assertAlmostEqual(
        distance_fn(a, b),
        NumpyDistanceFn(signal_dim, signal_len)(a, b),)


if __name__ == '__main__':
  absltest.main()
