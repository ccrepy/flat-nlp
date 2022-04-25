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

"""Queryset utils."""

import itertools
from typing import Callable, Sequence, Union

import pandas as pd

from google.protobuf import json_format
from flat_nlp.public import flat_pb2


class QuerysetDf(pd.DataFrame):
  id: pd.Series
  pretokenized_string: pd.Series
  weight: pd.Series
  log_proba: pd.Series
  lm_score: pd.Series


_TokenizerFn = Callable[[str], Sequence[str]]


def pretokenize(
    tokenizer_fn_or_locale: Union[_TokenizerFn, str],
    utf8_str_or_list: Union[str, Sequence[str]]) -> Union[str, Sequence[str]]:
  """Pretokenizes a string or a list of string."""
  if not tokenizer_fn_or_locale:
    raise ValueError('the locale must be specified for the tokenizer')

  tokenizer_fn = build_tokenizer_fn(tokenizer_fn_or_locale) if isinstance(
      tokenizer_fn_or_locale, str) else tokenizer_fn_or_locale

  if isinstance(utf8_str_or_list, str):
    return ' '.join(tokenizer_fn(utf8_str_or_list))
  else:
    return [' '.join(tokenizer_fn(utf8_str)) for utf8_str in utf8_str_or_list]


def convert_queryset_to_df(queryset: flat_pb2.QuerySet) -> QuerysetDf:
  """Converts a QuerySet object into a DataFrame.

  The pandas.DataFrame has the same columns as the QuerySet.Query proto.

  Arguments:
    queryset: QuerySet to convert

  Returns:
    the queryset represented as a DataFrame.

  Raises:
    ValueError: the queryset is empty
  """
  if not queryset.query_list:
    raise ValueError('the queryset is empty')

  df = pd.DataFrame.from_dict(
      json_format.MessageToDict(
          queryset,
          including_default_value_fields=True,
          preserving_proto_field_name=True)['query_list'])
  return df


def query_str_list(queryset: flat_pb2.QuerySet) -> Sequence[str]:
  return [query.pretokenized_string for query in queryset.query_list]


def mask_queryset(queryset: flat_pb2.QuerySet,
                  mask: Sequence[bool]) -> flat_pb2.QuerySet:
  """Masks the query_list of a queryset."""
  masked_queryset = flat_pb2.QuerySet()
  masked_queryset.CopyFrom(queryset)
  masked_queryset.ClearField('query_list')

  masked_queryset.query_list.extend(
      itertools.compress(queryset.query_list, mask))

  return masked_queryset
