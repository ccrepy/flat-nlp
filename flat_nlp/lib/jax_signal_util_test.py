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

"""Tests for jax_signal_util."""

import numpy as np

from absl.testing import absltest
from flat_nlp.lib import jax_signal_util
from flat_nlp.lib import np_signal_util
from flat_nlp.lib import test_signal_util


class JaxSignalUtilTest(absltest.TestCase):

  def test_stride_signal(self):
    np.testing.assert_allclose(
        jax_signal_util.stride_signal(test_signal_util.TEST_TENSOR),
        np_signal_util.stride_signal(test_signal_util.TEST_TENSOR))

  def test_unstride_signal(self):
    np.testing.assert_allclose(
        jax_signal_util.unstride_signal(test_signal_util.TEST_STRIDED_TENSOR, 2,
                                        8),
        np_signal_util.unstride_signal(test_signal_util.TEST_STRIDED_TENSOR, 2,
                                       8))

  def test_rfft_to_hc(self):
    np.testing.assert_allclose(
        jax_signal_util.rfft_to_hc(test_signal_util.TEST_RFFT_TENSOR),
        np_signal_util.rfft_to_hc(test_signal_util.TEST_RFFT_TENSOR))

  def test_hc_to_rfft(self):
    np.testing.assert_allclose(
        jax_signal_util.hc_to_rfft(test_signal_util.TEST_HC_TENSOR),
        np_signal_util.hc_to_rfft(test_signal_util.TEST_HC_TENSOR))

  def test_hc(self):
    np.testing.assert_allclose(
        jax_signal_util.hc(test_signal_util.TEST_TENSOR),
        np_signal_util.hc(test_signal_util.TEST_TENSOR),
        atol=0.0001)

  def test_ihc(self):
    np.testing.assert_allclose(
        jax_signal_util.ihc(test_signal_util.TEST_HC_TENSOR),
        np_signal_util.ihc(test_signal_util.TEST_HC_TENSOR),
        atol=0.0001)

  def test_hc_conjugate(self):
    np.testing.assert_allclose(
        jax_signal_util.hc_conjugate(test_signal_util.TEST_HC_TENSOR),
        np_signal_util.hc_conjugate(test_signal_util.TEST_HC_TENSOR))

  def test_hc_itemwise_pdt(self):
    np.testing.assert_allclose(
        jax_signal_util.hc_itemwise_pdt(test_signal_util.TEST_HC_TENSOR,
                                        test_signal_util.TEST_HC_TENSOR),
        np_signal_util.hc_itemwise_pdt(test_signal_util.TEST_HC_TENSOR,
                                       test_signal_util.TEST_HC_TENSOR),
        atol=0.0001)

  def test_normalize_signal_l2(self):
    np.testing.assert_allclose(
        jax_signal_util.normalize_signal_l2(
            test_signal_util.TEST_TENSOR, epsilon=1e-6),
        np_signal_util.normalize_signal_l2(test_signal_util.TEST_TENSOR),
        atol=0.0001)

  def test_normalize_rfft_signal_l2(self):
    with self.assertRaises(NotImplementedError):
      jax_signal_util.normalize_rfft_signal_l2(None)

  def test_normalize_hc_signal_l2(self):
    with self.assertRaises(NotImplementedError):
      jax_signal_util.normalize_hc_signal_l2(None)

  def test_energy(self):
    np.testing.assert_allclose(
        jax_signal_util.energy(test_signal_util.TEST_TENSOR),
        np_signal_util.energy(test_signal_util.TEST_TENSOR))

  def test_rfft_energy(self):
    np.testing.assert_allclose(
        jax_signal_util.rfft_energy(test_signal_util.TEST_RFFT_TENSOR),
        np_signal_util.rfft_energy(test_signal_util.TEST_RFFT_TENSOR))

  def test_hc_energy(self):
    np.testing.assert_allclose(
        jax_signal_util.hc_energy(test_signal_util.TEST_HC_TENSOR),
        np_signal_util.hc_energy(test_signal_util.TEST_HC_TENSOR))

  def test_normalize_signal_energy(self):
    np.testing.assert_allclose(
        jax_signal_util.normalize_signal_energy(test_signal_util.TEST_TENSOR),
        np_signal_util.normalize_signal_energy(test_signal_util.TEST_TENSOR))

  def test_normalize_signal_energy_with_epsilon(self):
    normalized_t = jax_signal_util.normalize_signal_energy(
        test_signal_util.TEST_TENSOR, epsilon=-1.)

    np.testing.assert_equal(
        np.unique(np.isfinite(normalized_t), axis=-1),
        [[[False], [True]], [[True], [True]]])

  def test_normalize_rfft_signal_energy(self):
    np.testing.assert_allclose(
        jax_signal_util.normalize_rfft_signal_energy(
            test_signal_util.TEST_RFFT_TENSOR),
        np_signal_util.normalize_rfft_signal_energy(
            test_signal_util.TEST_RFFT_TENSOR))

  def test_normalize_rfft_signal_energy_with_epsilon(self):
    normalized_t = jax_signal_util.normalize_rfft_signal_energy(
        test_signal_util.TEST_RFFT_TENSOR, epsilon=-1.)

    np.testing.assert_equal(
        np.unique(np.isfinite(normalized_t), axis=-1),
        [[[False], [True]], [[True], [True]]])

  def test_normalize_hc_signal_energy(self):
    np.testing.assert_allclose(
        jax_signal_util.normalize_hc_signal_energy(
            test_signal_util.TEST_HC_TENSOR),
        np_signal_util.normalize_hc_signal_energy(
            test_signal_util.TEST_HC_TENSOR))

  def test_normalize_hc_signal_energy_with_epsilon(self):
    normalized_t = jax_signal_util.normalize_hc_signal_energy(
        test_signal_util.TEST_HC_TENSOR, epsilon=-1.)

    np.testing.assert_equal(
        np.unique(np.isfinite(normalized_t), axis=-1),
        [[[False], [True]], [[True], [True]]])

if __name__ == '__main__':
  absltest.main()
