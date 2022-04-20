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

"""Tests for stratified_sampling."""

from flat_nlp.encoding.lib import test_encoding_util
from flat_nlp.grouping.lib import stratified_sampling
from flat_nlp.public import queryset_util
from absl.testing import absltest

_TEST_QUERY_LIST = ('hello here', 'hi there', 'hello there', 'i like the cat',
                    'i ate some fish')

# Dummy function to select all queries
_select_all = lambda df: df.index


class StratifiedSamplingTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(StratifiedSamplingTest, cls).setUpClass()
    cls.encoder = test_encoding_util.SIMPLE_TEXT8_ENCODER

  def test_sampling_select_one(self):
    sample_size = 2
    test_df = queryset_util.QuerysetDf(
        {'pretokenized_string': _TEST_QUERY_LIST})

    sampled_df = stratified_sampling.sample_queryset_df(
        self.encoder, test_df, sample_size,
        stratified_sampling._select_one_per_cluster)

    self.assertLen(test_df, 5)
    self.assertLen(sampled_df, sample_size)

  def test_sampling_drop_duplicates(self):
    sample_size = 2
    test_df = queryset_util.QuerysetDf(
        {'pretokenized_string': _TEST_QUERY_LIST + _TEST_QUERY_LIST})

    sampled_df = stratified_sampling.sample_queryset_df(self.encoder, test_df,
                                                        sample_size,
                                                        _select_all)

    self.assertLen(test_df, 10)
    self.assertLen(sampled_df, test_df.pretokenized_string.nunique())


if __name__ == '__main__':
  absltest.main()
