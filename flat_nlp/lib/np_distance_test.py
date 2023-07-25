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

"""Tests for np_distance."""

# pylint: disable=line-too-long

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from flat_nlp.lib import np_distance
from flat_nlp.lib import np_signal_util
from flat_nlp.lib import test_signal_util


class ConvolutionTest(absltest.TestCase):

  def test_simple_convolution(self):
    s1 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s1
    s2 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s2

    s_out = np_distance.convolve_signal(s1, s2)
    np.testing.assert_allclose(s_out, [[0, 1, 2, 0]], atol=0.0001)

  def test_simple_hc_convolution(self):
    hc_s1 = test_signal_util.TESTSET_HC_SIGNAL_CONVOLUTION.s1
    hc_s2 = test_signal_util.TESTSET_HC_SIGNAL_CONVOLUTION.s2

    hc_s_out = np_distance.convolve_hc_signal(hc_s1, hc_s2)
    (s_out,) = np_signal_util.ihc(np.array([hc_s_out]))
    np.testing.assert_allclose(s_out, [[0, 1, 2, 0]], atol=0.0001)

  def test_convolution_shift_change(self):
    s1 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s1
    s2 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s2

    s_out = np_distance.convolve_signal(s1, np.flip(s2, axis=-1))
    np.testing.assert_allclose(s_out, [[0, 0, 1, 2]], atol=0.0001)

  def test_convolution_scale_change(self):
    s1 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s1
    s2 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s2

    s_out = np_distance.convolve_signal(s1, 2 * np.flip(s2, axis=-1))
    np.testing.assert_allclose(s_out, [[0, 0, 2, 4]], atol=0.0001)

  def test_correlation(self):
    s1 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s1
    s1_energy = np.sqrt(np.power(s1, 2).sum(axis=-1))
    s1_norm = s1 / s1_energy[np.newaxis, :]

    s2 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s2
    s2_energy = np.sqrt(np.power(s2, 2).sum(axis=-1))
    s2_norm = s2 / s2_energy[np.newaxis, :]

    s_out = np_distance.convolve_signal(s1_norm, s2_norm)
    np.testing.assert_allclose(
        s_out, [[0, .447214, .894427, 0]], atol=0.0001)
    self.assertAlmostEqual(np.linalg.norm(s_out), 1.)

  def test_autocorrelation(self):
    s1 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s1
    s1_energy = np.sqrt(np.power(s1, 2).sum(axis=-1))
    s1_norm = s1 / s1_energy[np.newaxis, :]

    s_out = np_distance.convolve_signal(s1_norm, s1_norm)
    np.testing.assert_allclose(
        s_out, [[.4, 0, .4, 1]], atol=0.0001)
    self.assertAlmostEqual(np.max(s_out), 1.)


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
