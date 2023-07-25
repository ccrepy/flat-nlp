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

"""Utility library for Kernel Density Estimation."""

from typing import Sequence

import numpy as np

from scipy import optimize

from sklearn import neighbors

_1dSignal = np.ndarray  # pylint: disable=invalid-name


def aggregate_kde_model(
    source_kde: neighbors.KernelDensity) -> neighbors.KernelDensity:
  """Aggregates a trained kde model into a simpler one.

  Aggregated models are faster to run at the cost of some accuracy loss.
  The tradeoff speed / precision is defined by the leaf_size of the initial kde
  model.

  Arguments:
    source_kde: the initial kde model.

  Returns:
    The aggregated kde model.
  """
  agg_kde = neighbors.KernelDensity(**source_kde.get_params())

  # read data from the balltree index
  data, ix_mapping, node_list, _ = source_kde.tree_.get_arrays()
  if source_kde.tree_.sample_weight is not None:
    data_weight = np.array(source_kde.tree_.sample_weight)
  else:
    data_weight = np.ones(len(ix_mapping))

  # select only the aggregated leaf
  #   we need a list in order to leverage the vectorized numpy functions.
  leaf_ix_mapping = [
      ix_mapping[beg:end + 1] for beg, end, is_leaf, _ in node_list if is_leaf
  ]

  source_x = [
      np.average(data[mapped_ix], axis=0, weights=data_weight[mapped_ix])
      for mapped_ix in leaf_ix_mapping
  ]
  source_weight = [
      data_weight[mapped_ix].sum() for mapped_ix in leaf_ix_mapping
  ]

  # train the aggregated model
  agg_kde.fit(np.ascontiguousarray(source_x), sample_weight=source_weight)
  return agg_kde


def normalize_kde_pdf(predicted_log_proba: Sequence[float],
                      maximum_pdf: float) -> Sequence[float]:
  """Normalizes log proba output from a kde model to guarantee output in ]0, 1]."""
  return np.clip(np.exp(predicted_log_proba) / maximum_pdf, 0., 1.)


def naive_search_maximum(trained_kde_model: neighbors.KernelDensity) -> float:
  """Computes the maximum of the proba for each point among the training set.

  This is a naive approach as the global maximum of the pdf is likely between
  points.
  Hence, using this maximum to normalize, is acceptable iif the normalized
  output is capped to 1. after normalization.

  Arguments:
    trained_kde_model: the kde model for which we want the global maximum.

  Returns:
    The estimated maximum of the pdf.
  """
  x_train, _, _, _ = trained_kde_model.tree_.get_arrays()
  y_train = trained_kde_model.score_samples(x_train)

  return max(np.exp(y_train))


def newton_search_maximum(trained_kde_model: neighbors.KernelDensity) -> float:
  """Estimates the global maximum of the pdf using a newton like approach.

  The default implementation of minimize with bounded constraints is:
    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb

  This iterative method might not be suitable when the model is trained too many
    points or too many dimensions.
  For reference, ~50 queries in 1024 dim takes about ~5min.

  We do not need the convergence to get an approximate of the principal mode of
    the pdf.

  Arguments:
    trained_kde_model: the kde model for which we want the global maximum.

  Returns:
    The estimated maximum of the pdf.
  """
  x_train, _, _, _ = trained_kde_model.tree_.get_arrays()
  x_bound = list(zip(x_train.min(axis=0), x_train.max(axis=0)))

  eval_pdf = lambda x: -trained_kde_model.score_samples([x])[0]
  y_eval = [optimize.minimize(eval_pdf, x, bounds=x_bound).fun for x in x_train]
  return max(np.exp(-1. * np.array(y_eval)))
