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

"""Tests for np_distance."""

import numpy as np

from flat_nlp.lib import np_distance

from flat_nlp.lib import test_signal_util

from absl.testing import absltest
from absl.testing import parameterized


@parameterized.parameters(
    (np_distance.convolve_signal, test_signal_util.TESTSET_SIGNAL_CONVOLUTION),
    (np_distance.convolve_hc_signal,
     test_signal_util.TESTSET_HC_SIGNAL_CONVOLUTION),
)
class ConvolutionTest(parameterized.TestCase):

  def test_convolution_shift_change(self, convolve_fn, testset):
    out_s = convolve_fn(
        testset.s1, testset.s2, flip_s2_along_time=False, normalize=False)
    np.testing.assert_allclose(out_s, [[0, 0, 1., 2.]], atol=0.0001)

  def test_convolution_scale_change(self, convolve_fn, testset):
    out_s = convolve_fn(
        testset.s1, 2 * testset.s2, flip_s2_along_time=False, normalize=False)
    np.testing.assert_allclose(out_s, [[0, 0, 2., 4.]], atol=0.0001)

  def test_correlation(self, convolve_fn, testset):
    out_s = convolve_fn(
        testset.s1, testset.s2, flip_s2_along_time=True, normalize=True)
    np.testing.assert_allclose(
        out_s, [[0., 0.447214, 0.894427, 0.]], atol=0.0001)
    self.assertAlmostEqual(np.linalg.norm(out_s), 1)

  def test_autocorrelation(self, convolve_fn, testset):
    out_s = convolve_fn(
        testset.s1, testset.s1, flip_s2_along_time=True, normalize=True)
    np.testing.assert_allclose(out_s, [[.4, 0., 0.4, 1.]], atol=0.0001)
    self.assertEqual(np.max(out_s), 1)


class DistanceTest(parameterized.TestCase):

  @parameterized.parameters(
      (np_distance.flat_distance, test_signal_util.TESTSET_SIGNAL_DISTANCE),
      (np_distance.hc_flat_distance,
       test_signal_util.TESTSET_HC_SIGNAL_DISTANCE),
  )
  def test_distance(self, distance_fn, testset):
    # identity of indiscernibles
    self.assertAlmostEqual(distance_fn(testset.s1, testset.s1), 0.)

    # symmetry
    self.assertEqual(
        distance_fn(testset.s1, testset.s2),
        distance_fn(testset.s2, testset.s1))

    # triangular inequality
    self.assertGreaterEqual(
        distance_fn(testset.s1, testset.s2) +
        distance_fn(testset.s2, testset.s3),
        distance_fn(testset.s1, testset.s3))

    # value check
    self.assertAlmostEqual(distance_fn(testset.s1, testset.s2), .8660254)
    self.assertAlmostEqual(distance_fn(testset.s1, testset.s3), .7694958)
    self.assertAlmostEqual(distance_fn(testset.s2, testset.s3), .7337358)

  @parameterized.parameters(
      (np_distance.flat_distance, np_distance.StridedFlatDistanceFn,
       test_signal_util.TESTSET_SIGNAL_DISTANCE),
      (np_distance.hc_flat_distance,
       np_distance.StridedHalfComplexFlatDistanceFn,
       test_signal_util.TESTSET_HC_SIGNAL_DISTANCE),
  )
  def test_strided_distance(self, distance_fn, StridedDistanceFn, testset):  # pylint: disable=invalid-name
    strided_distance_fn = StridedDistanceFn(2, 4)

    self.assertEqual(
        strided_distance_fn(testset.s1.ravel(), testset.s2.ravel()),
        distance_fn(testset.s1, testset.s2))


if __name__ == '__main__':
  absltest.main()
