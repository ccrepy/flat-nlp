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

"""Tests for bow_ranker."""

import numpy as np

from flat_nlp.encoding.lib import test_encoding_util
from flat_nlp.ranking.lib import bow_ranker

from absl.testing import absltest

_TEST_QUERY_LIST = ('hello here', 'hi there', 'hello there')

_EXPECTED_SELF_SIMILARITY_SCORES = (-0.1950497, -0.25383483, -0.14220668)


class BowRankerTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(BowRankerTest, cls).setUpClass()
    cls.encoder = test_encoding_util.SIMPLE_TEXT8_ENCODER

  def test_bow_encoding(self):
    ranker_model = bow_ranker.BowRanker(self.encoder)
    self.assertEqual(ranker_model.bow_encoding(_TEST_QUERY_LIST).shape, (3, 5))

  def test_ranker(self):
    ranker_model = bow_ranker.BowRanker(self.encoder)
    ranker_model.fit_queries(_TEST_QUERY_LIST)

    np.testing.assert_allclose(
        ranker_model.predict_queries(_TEST_QUERY_LIST),
        _EXPECTED_SELF_SIMILARITY_SCORES,
        atol=0.0001)


if __name__ == '__main__':
  absltest.main()
