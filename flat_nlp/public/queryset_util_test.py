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

"""Tests for queryset_util."""

from tensorflow.python.util.protobuf import compare

from google.protobuf import text_format

from absl.testing import absltest
from absl.testing import parameterized

from flat_nlp.public import flat_pb2
from flat_nlp.public import queryset_util

_TEST_QUERYSET = """
query_list {
  pretokenized_string: "query A"
}

query_list {
  pretokenized_string: "query B"
}
"""


def _parse_queryset(queryset_pbtxt):
  queryset = flat_pb2.QuerySet()
  text_format.Parse(queryset_pbtxt, queryset)
  return queryset


class QuerysetUtilTest(parameterized.TestCase, compare.Proto2Assertions):

  @parameterized.parameters(
      ('en', '  hello you', 'hello you'),
      ('en', 'hello you?', 'hello you ?'),
      ('en', 'the (0:cat) is here', 'the ( 0 : cat ) is here'),
      ('en', ' hello you', 'hello you'),
      ('en', ['hi there!', 'hello you '], ['hi there !', 'hello you']),
      (queryset_util.build_tokenizer_fn('en'), '  hello you', 'hello you'),
  )
  def test_pretokenize(self, tokenizer_fn_or_locale, raw_string,
                       pretokenized_string):
    self.assertEqual(
        queryset_util.pretokenize(tokenizer_fn_or_locale, raw_string),
        pretokenized_string)

  def test_pretokenize_no_locale(self):
    with self.assertRaisesRegex(ValueError, 'the locale must be specified'):
      queryset_util.pretokenize('', 'dummy_string')

  def test_convert_queryset_to_df(self):
    df = queryset_util.convert_queryset_to_df(_parse_queryset(_TEST_QUERYSET))
    self.assertLen(df, 2)

  def test_convert_queryset_to_df_empty(self):
    with self.assertRaisesRegex(ValueError, 'the queryset is empty'):
      queryset_util.convert_queryset_to_df(flat_pb2.QuerySet())

  def test_query_str_list(self):
    self.assertEqual(
        queryset_util.query_str_list(_parse_queryset(_TEST_QUERYSET)),
        ['query A', 'query B'])

  def test_mask_queryset(self):
    self.assertProto2Equal(
        """
    query_list {
      pretokenized_string: "query A"
    }""",
        queryset_util.mask_queryset(
            _parse_queryset(_TEST_QUERYSET), [True, False]))


if __name__ == '__main__':
  absltest.main()
