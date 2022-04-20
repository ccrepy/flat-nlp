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

"""Bag of word Ranker.

This simple model provide a ranking based on Kernel Density Estimation and a bag
of word feature from embeddings.

It is intended to be used for testing purposes as a dummy model to plug into
more complex systems.
"""

from typing import Optional, Sequence

import numpy as np

from sklearn import neighbors
from sklearn import preprocessing

from flat_nlp.encoding.lib import encoder_util

from flat_nlp.public import queryset_util

DEFAULT_BANDWIDTH = .4
DEFAULT_LEAF_SIZE = 5


class BowRanker:
  """Bag of word Ranker.

  Attributes:
    encoder: encoder_util.Encoder to use.
    kde_model: the underlying Kernel Density Estimation model.
  """

  def __init__(self,
               encoder: encoder_util.Encoder,
               bandwidth: float = DEFAULT_BANDWIDTH,
               leaf_size: int = DEFAULT_LEAF_SIZE):
    if encoder.is_output_strided or encoder.is_output_hc:
      raise NotImplementedError(
          'Bag of words ranker only supports non strided time domain encoders.')
    self.encoder = encoder
    self.kde_model = neighbors.KernelDensity(
        bandwidth=bandwidth, leaf_size=leaf_size)

  def bow_encoding(self, pretokenized_string_list: Sequence[str]) -> np.ndarray:
    """Encodes a list of a pretokenized string as bag of words.

    Each vector are normalized so the L2 distance is equivalent to cosine
    distance.

    Arguments:
      pretokenized_string_list: pretokenized queries to encode.

    Returns:
      matrix representing each query as averaged bag of words.
    """
    return preprocessing.normalize(
        self.encoder.call(list(pretokenized_string_list)).numpy().mean(axis=-1))

  def fit(self, queryset_df: queryset_util.QuerysetDf) -> 'BowRanker':
    """Fits the model on a queryset.

    Arguments:
      queryset_df: the queryset to evaluate as a dataframe, with columns -
        pretokenized_string - weight (optional, set to 1. if not defined)

    Returns:
      the trained model.
    """

    if 'weight' not in queryset_df.columns:
      queryset_df['weight'] = 1.

    train_x = self.bow_encoding(queryset_df.pretokenized_string)
    self.kde_model.fit(train_x, sample_weight=queryset_df.weight)
    return self

  def fit_queries(self,
                  pretokenized_string_list: Sequence[str],
                  weight_list: Optional[Sequence[int]] = None):
    """Fits the model on a list of queries."""
    return self.fit(
        queryset_util.QuerysetDf({
            'pretokenized_string': pretokenized_string_list,
            'weight': weight_list if weight_list else 1.
        }))

  def predict(self, queryset_df: queryset_util.QuerysetDf) -> Sequence[float]:
    """Predicts score for a new queryset.

    Arguments:
      queryset_df: the queryset to evaluate as a dataframe, with columns -
        pretokenized_string

    Returns:
      the probabilities of the testing set.
    """
    test_x = self.bow_encoding(queryset_df.pretokenized_string)
    return self.kde_model.score_samples(test_x)

  def predict_queries(
      self, pretokenized_string_list: Sequence[str]) -> Sequence[float]:
    """Predicts score for a new list of queries."""
    return self.predict(
        queryset_util.QuerysetDf(
            {'pretokenized_string': pretokenized_string_list}))
