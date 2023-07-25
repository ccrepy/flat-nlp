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

"""Tests for jax_util."""

# pylint: disable=g-complex-comprehension

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from absl.testing import absltest
from flat_nlp.encoding.lib import test_encoding_util
from flat_nlp.learning.lib import jax_util


class JaxUtilTest(absltest.TestCase):

  def test_l2_normalize(self):
    np.testing.assert_array_almost_equal(
        jax_util.l2_normalize(jnp.array([
            [1., 2., 3.],
            [4., 5., 6.],
        ])),
        np.array([[0.267261, 0.534522, 0.801784],
                  [0.455842, 0.569803, 0.683764]]))

  def test_vmap_product(self):
    np.testing.assert_array_almost_equal(
        jax_util.vmap_product(
            lambda a, b: jnp.abs(a - b),
            jnp.array([1, 2, 3]),
            jnp.array([1, 2]),
        ),
        jnp.array([
            [0., 1., 2.],
            [1., 0., 1.],
        ]),
    )


class JaxPretrainedEmbeddingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # fit to vocabulary
    self.encoder = tf.keras.layers.TextVectorization()
    self.encoder.adapt([
        'query a',
        'query b',
    ])

    # build and init the layer
    self.pretrained_embeddings_module = (
        jax_util.build_pretrained_embeddings_module(
            self.encoder, test_encoding_util.SIMPLE_EMBEDDING_MODEL
        )
    )

    self.params = self.pretrained_embeddings_module.init(
        jax.random.PRNGKey(0),
        jnp.array([jnp.ones(2, jnp.int32)]),
        method=self.pretrained_embeddings_module.lookup_and_encode,
    )

  def test_lookup_and_encode(self):
    encoded_x = self.encoder([
        'query a',
        'query b',
        'query c',
    ]).numpy()

    y = [
        self.pretrained_embeddings_module.apply(
            self.params,
            x,
            method=self.pretrained_embeddings_module.lookup_and_encode,
        )
        for x in encoded_x
    ]

    np.testing.assert_array_equal(
        y,
        np.array([
            [[1., 1.], [1., 0.], [1., 0.]],  # query a
            [[1., 0.], [1., 2.], [1., 0.]],  # query b
            [[1., 0.], [1., 0.], [1., 0.]],  # query c
        ]))

  def test_lookup(self):
    tokens = self.pretrained_embeddings_module.apply(
        self.params,
        jnp.array([0, 1, 2, 3]),
        method=self.pretrained_embeddings_module.lookup)
    np.testing.assert_array_almost_equal(
        tokens,
        np.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 1., 1.],
            [0., 2., 0.]
        ]))


if __name__ == '__main__':
  absltest.main()
