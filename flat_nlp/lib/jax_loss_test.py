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

"""Tests for jax_loss."""

import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from flat_nlp.lib import jax_loss
from flat_nlp.lib import jax_signal_util
from flat_nlp.lib import np_distance
from flat_nlp.lib import test_signal_util


class JaxLossTest(absltest.TestCase):

  def test_convolve_tensor(self):
    t1 = jnp.array([test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s1])
    t2 = jnp.array([test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s2])

    np.testing.assert_allclose(
        jax_loss.convolve_tensor(t1, t2)[0],
        np_distance.convolve_signal(
            t1[0], t2[0], flip_s2_along_time=True, normalize=False))

  def test_convolve_same_tensor(self):
    t1 = jnp.array([test_signal_util.TESTSET_SIGNAL_CONVOLUTION.s1])

    np.testing.assert_allclose(
        jax_loss.convolve_tensor(t1, t1)[0],
        np_distance.convolve_signal(
            t1[0], t1[0], flip_s2_along_time=True, normalize=False))

  def test_flat_loss(self):
    s1 = test_signal_util.TESTSET_SIGNAL_DISTANCE.s1
    s2 = test_signal_util.TESTSET_SIGNAL_DISTANCE.s2
    s3 = test_signal_util.TESTSET_SIGNAL_DISTANCE.s3

    t1 = jax_signal_util.normalize_signal(jnp.array([s1, s1, s2]))
    t2 = jax_signal_util.normalize_signal(jnp.array([s2, s3, s3]))

    np.testing.assert_allclose(
        jax_loss.flat_loss(t1, t2), [
            np_distance.flat_distance(s1, s2),
            np_distance.flat_distance(s1, s3),
            np_distance.flat_distance(s2, s3),
        ], atol=0.0001)

  def test_flat_loss_with_epsilon(self):
    s1 = test_signal_util.TESTSET_SIGNAL_DISTANCE.s1
    t1 = jax_signal_util.normalize_signal(jnp.array([s1]))

    np.testing.assert_equal(jax_loss.flat_loss(t1, t1, epsilon=-.1), [jnp.nan])

  def test_convolve_hc_tensor(self):
    with self.assertRaises(NotImplementedError):
      jax_loss.convolve_hc_tensor(None, None)

  def test_hc_flat_loss(self):
    with self.assertRaises(NotImplementedError):
      jax_loss.hc_flat_loss(None, None)


if __name__ == '__main__':
  absltest.main()
