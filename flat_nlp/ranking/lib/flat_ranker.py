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

"""Flat Ranker.

This model provide a ranking based on Kernel Density Estimation in a custom
metric space.

The metric space is built on top a word embeddings and defines a distance based
on maximum cross correlation between 2 signals.

"""

from typing import Optional, Sequence

from sklearn import neighbors

from flat_nlp.encoding.lib import encoder_adapter
from flat_nlp.encoding.lib import encoder_util
from flat_nlp.public import queryset_util
from flat_nlp.ranking.lib import kde_util

_DEFAULT_BANDWIDTH = .4
_DEFAULT_LEAF_SIZE = 4


# TODO(b/168749910): apply "Postponed Evaluation of Annotations" when available.
def aggregate_ranker(initial_ranker: 'FlatRanker') -> 'FlatRanker':
  """Aggregates a trained FlatRanker model into a simpler one."""
  agg_ranker = FlatRanker(initial_ranker.encoder)
  agg_ranker.kde_model = kde_util.aggregate_kde_model(initial_ranker.kde_model)
  return agg_ranker


class FlatRanker():
  """Flat Ranker.

  This ranker uses Flat distance as a custom metric with kernel
  density estimation in order to produce a log likelihood scoring for unseen
  queries wrt to a reference set.

  Attributes:
    encoder: encoder_util.Encoder to use.
    kde_model: the underlying Kernel Density Estimation model.
  """

  def __init__(self,
               encoder: encoder_util.Encoder,
               bandwidth: float = _DEFAULT_BANDWIDTH,
               leaf_size: int = _DEFAULT_LEAF_SIZE) -> None:
    self.encoder = encoder_adapter.StridedOutputEncoderAdapter(encoder)
    distance_fn = encoder_util.build_distance_fn(self.encoder)

    self.kde_model = neighbors.KernelDensity(
        bandwidth=bandwidth,
        leaf_size=leaf_size,
        metric='pyfunc',
        metric_params={'func': distance_fn})

  def fit(self, queryset_df: queryset_util.QuerysetDf) -> 'FlatRanker':
    """Fits the model on a queryset.

    Arguments:
      queryset_df: the queryset to evaluate as a dataframe, with columns -
        pretokenized_string - weight (optional, set to 1. if not defined)

    Returns:
      the trained model.
    """
    if 'weight' not in queryset_df.columns:
      queryset_df['weight'] = 1.

    train_x = self.encoder.call(queryset_df.pretokenized_string).numpy()
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
    test_x = self.encoder.call(queryset_df.pretokenized_string).numpy()
    return self.kde_model.score_samples(test_x)

  def predict_queries(
      self, pretokenized_string_list: Sequence[str]) -> Sequence[float]:
    """Predicts score for a new list of queries."""
    return self.predict(
        queryset_util.QuerysetDf(
            {'pretokenized_string': pretokenized_string_list}))
