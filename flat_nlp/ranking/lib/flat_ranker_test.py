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

"""Tests for flat_ranker."""

import numpy as np

from flat_nlp.encoding.lib import test_encoding_util
from flat_nlp.ranking.lib import flat_ranker

from absl.testing import absltest
from absl.testing import parameterized

_TEST_QUERY_LIST = ('hello here', 'hi there', 'hello there')

_EXPECTED_SELF_SIMILARITY_SCORES = (-0.1938740, -0.1495607, -0.1870395)


class FlatRankerTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(FlatRankerTest, cls).setUpClass()
    cls.encoder = test_encoding_util.SIMPLE_TEXT8_ENCODER

  def test_aggregate_ranker(self):
    full_model = flat_ranker.FlatRanker(self.encoder, leaf_size=1)
    full_model.fit_queries(_TEST_QUERY_LIST)
    agg_model = flat_ranker.aggregate_ranker(full_model)

    self.assertLen(full_model.kde_model.tree_.data, 3)
    self.assertLen(agg_model.kde_model.tree_.data, 2)

  def test_ranker(self):
    ranker_model = flat_ranker.FlatRanker(self.encoder)
    ranker_model.fit_queries(_TEST_QUERY_LIST)

    np.testing.assert_allclose(
        ranker_model.predict_queries(_TEST_QUERY_LIST),
        _EXPECTED_SELF_SIMILARITY_SCORES,
        atol=0.0001)


if __name__ == '__main__':
  absltest.main()
