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

"""Tests for embedding_util."""

import numpy as np

from flat_nlp.encoding.lib import embedding_util
from flat_nlp.encoding.lib import test_encoding_util

from absl.testing import absltest


class EmbeddingUtilTest(absltest.TestCase):

  def test_oov_no_strategy(self):
    np.testing.assert_allclose(
        embedding_util.build_embedding_matrix(
            test_encoding_util.SIMPLE_EMBEDDING_MODEL, ['a'], None),
        [
            [1., 0., 0.],  #  a
        ],
    )

    with self.assertRaisesRegex(KeyError, "'d'"):
      _ = embedding_util.build_embedding_matrix(
          test_encoding_util.SIMPLE_EMBEDDING_MODEL, ['d'], None)

  def test_oov_ignore_strategy(self):
    np.testing.assert_allclose(
        embedding_util.build_embedding_matrix(
            test_encoding_util.SIMPLE_EMBEDDING_MODEL, ['a', 'd'], 'ignore'),
        [
            [1., 0., 0.],  #  a
            [0., 0., 0.],  #  d
        ],
    )

  def test_oov_decompose_strategy(self):
    np.testing.assert_allclose(
        embedding_util.build_embedding_matrix(
            test_encoding_util.SIMPLE_EMBEDDING_MODEL,
            ['a', 'b', 'd', 'ab', 'cd'], 'decompose'),
        [
            [1., 0., 0.],  #  a
            [0., 2., 0.],  #  b
            [0., 0., 0.],  #  d
            [.5, 1., 0.],  #  ab
            [0., 0., 1.5],  #  cd
        ],
    )

  def test_oov_decompose_strategy_oov_char(self):
    np.testing.assert_allclose(
        embedding_util.build_embedding_matrix(
            test_encoding_util.SIMPLE_EMBEDDING_MODEL, ['🐘', '🐘a'],
            'decompose'),
        [
            [0., 0., 0.],  #  🐘
            [.5, 0., 0.],  #  🐘a
        ],
    )


if __name__ == '__main__':
  absltest.main()
