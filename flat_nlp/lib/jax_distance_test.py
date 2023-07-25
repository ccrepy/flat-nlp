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

"""Tests for jax_distance."""

import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from flat_nlp.lib import jax_distance
from flat_nlp.lib import jax_signal_util
from flat_nlp.lib import np_distance
from flat_nlp.lib import test_signal_util


class JaxDistanceTest(absltest.TestCase):

  def test_convolve_signal(self):
    s1 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s1
    s2 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s2

    np.testing.assert_allclose(
        jax_distance.convolve_signal(s1, s2),
        np_distance.convolve_signal(s1, s2),
        )

  def test_convolve_same_signal(self):
    s1 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s1

    np.testing.assert_allclose(
        jax_distance.convolve_signal(s1, s1),
        np_distance.convolve_signal(s1, s1),
        )

  def test_flat_distance(self):
    s1 = test_signal_util.TESTSET_SIGNAL_DISTANCE.s1
    s2 = test_signal_util.TESTSET_SIGNAL_DISTANCE.s2
    s3 = test_signal_util.TESTSET_SIGNAL_DISTANCE.s3

    np.testing.assert_allclose(
        jax_distance.flat_distance(s1, s2),
        np_distance.flat_distance(s1, s2),
        atol=0.0001,
    )
    np.testing.assert_allclose(
        jax_distance.flat_distance(s1, s3),
        np_distance.flat_distance(s1, s3),
        atol=0.0001,
    )
    np.testing.assert_allclose(
        jax_distance.flat_distance(s2, s3),
        np_distance.flat_distance(s2, s3),
        atol=0.0001,
    )

  def test_flat_distance_with_epsilon(self):
    s1 = test_signal_util.TESTSET_SIGNAL_DISTANCE.s1

    np.testing.assert_equal(
        jax_distance.flat_distance(s1, s1, epsilon=-0.1), jnp.array([jnp.nan])
    )

  def test_convolve_rfft_signal(self):
    s1 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s1
    s2 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s2

    rfft_s1, rfft_s2 = jnp.fft.rfft(s1), jnp.fft.rfft(s2)

    np.testing.assert_allclose(
        jax_distance.convolve_signal(s1, s2),
        jnp.fft.irfft(jax_distance.convolve_rfft_signal(rfft_s1, rfft_s2)),
    )

  def test_rfft_flat_distance(self):
    s1 = test_signal_util.TESTSET_HC_SIGNAL_DISTANCE.s1
    s2 = test_signal_util.TESTSET_HC_SIGNAL_DISTANCE.s2
    s3 = test_signal_util.TESTSET_HC_SIGNAL_DISTANCE.s3

    rfft_s1, rfft_s2, rfft_s3 = (
        jnp.fft.rfft(s1),
        jnp.fft.rfft(s2),
        jnp.fft.rfft(s3),
    )

    np.testing.assert_allclose(
        jax_distance.flat_distance(s1, s2),
        jax_distance.rfft_flat_distance(rfft_s1, rfft_s2),
        atol=0.0001,
    )
    np.testing.assert_allclose(
        jax_distance.flat_distance(s1, s3),
        jax_distance.rfft_flat_distance(rfft_s1, rfft_s3),
        atol=0.0001,
    )
    np.testing.assert_allclose(
        jax_distance.flat_distance(s2, s3),
        jax_distance.rfft_flat_distance(rfft_s2, rfft_s3),
        atol=0.0001,
    )

  def test_rfft_flat_distance_with_epsilon(self):
    rfft_s1 = jnp.fft.rfft(test_signal_util.TESTSET_SIGNAL_DISTANCE.s1)

    np.testing.assert_equal(
        jax_distance.rfft_flat_distance(rfft_s1, rfft_s1, epsilon=-0.1),
        jnp.array([jnp.nan]),
    )

  def test_convolve_hc_signal(self):
    s1 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s1
    s2 = test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s2

    hc_s1 = test_signal_util.TESTSET_HC_SIGNAL_CONVOLUTION.s1
    hc_s2 = test_signal_util.TESTSET_HC_SIGNAL_CONVOLUTION.s2

    np.testing.assert_allclose(
        jax_distance.convolve_signal(s1, s2),
        jax_signal_util.ihc(jax_distance.convolve_hc_signal(hc_s1, hc_s2)),
    )

  def test_hc_flat_distance(self):
    hc_s1 = test_signal_util.TESTSET_HC_SIGNAL_DISTANCE.s1
    hc_s2 = test_signal_util.TESTSET_HC_SIGNAL_DISTANCE.s2
    hc_s3 = test_signal_util.TESTSET_HC_SIGNAL_DISTANCE.s3

    np.testing.assert_allclose(
        jax_distance.hc_flat_distance(hc_s1, hc_s2),
        np_distance.hc_flat_distance(hc_s1, hc_s2),
        atol=0.0001,
    )
    np.testing.assert_allclose(
        jax_distance.hc_flat_distance(hc_s1, hc_s3),
        np_distance.hc_flat_distance(hc_s1, hc_s3),
        atol=0.0001,
    )
    np.testing.assert_allclose(
        jax_distance.hc_flat_distance(hc_s2, hc_s3),
        np_distance.hc_flat_distance(hc_s2, hc_s3),
        atol=0.0001,
    )

  def test_hc_flat_distance_with_epsilon(self):
    hc_s1 = test_signal_util.TESTSET_HC_SIGNAL_DISTANCE.s1

    np.testing.assert_equal(
        jax_distance.hc_flat_distance(hc_s1, hc_s1, epsilon=-0.1),
        jnp.array([jnp.nan]),
    )


if __name__ == '__main__':
  absltest.main()
