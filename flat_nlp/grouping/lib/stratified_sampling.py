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

"""Stratified sampling.

This library provide a framework to run sampling based on agglomerative
clustering and Flat distance.

The goal is to reduce the size of a dataset while minimizing the linguistic
coverage loss.

"""

from typing import Callable, Sequence, Union

import scipy  # pylint: disable=unused-import
from sklearn import cluster
from sklearn import metrics

from flat_nlp.encoding.lib import encoder_adapter
from flat_nlp.encoding.lib import encoder_util

from flat_nlp.public import queryset_util

DFIndex = Sequence[Union[int, str]]


def _select_one_per_cluster(
    clustered_queryset_df: queryset_util.QuerysetDf) -> DFIndex:
  return clustered_queryset_df.reset_index().groupby('cluster').first()['index']


def _cluster_queryset_df(
    encoder: encoder_util.Encoder,
    queryset_df: queryset_util.QuerysetDf,
    n_clusters: int,
) -> Sequence[int]:
  """Clusters a queryset.

  Step before stratfied sampling. Useful for checking the clusters on colab
  before performing sampling

  Arguments:
    encoder: encoder_util.Encoder to use for pairwise distance.
    queryset_df: set of queries to sample.
    n_clusters: number of clusters to look for

  Returns:
    the cluster label associated to each query.
  """
  encoder = encoder_adapter.StridedOutputEncoderAdapter(encoder)
  distance_fn = encoder_util.build_distance_fn(encoder)

  pairwise_distances = metrics.pairwise_distances(
      encoder.call(queryset_df.pretokenized_string).numpy(),
      metric=distance_fn)  # see https://stackoverflow.com/a/53850942

  agglomerative_clusterer = cluster.AgglomerativeClustering(
      n_clusters=n_clusters, affinity='precomputed', linkage='complete')
  agglomerative_clusterer.fit(pairwise_distances)

  return agglomerative_clusterer.labels_


def sample_queryset_df(
    encoder: encoder_util.Encoder, queryset_df: queryset_util.QuerysetDf,
    n_clusters: int, ix_selection_fn: Callable[[queryset_util.QuerysetDf],
                                               DFIndex]
) -> queryset_util.QuerysetDf:
  """Performs stratfied sampling on a queryset.

  Arguments:
    encoder: encoder_util.Encoder to use for pairwise distance.
    queryset_df: set of queries to sample.
    n_clusters: number of clusters to look for, size of the sample in the case
      one query per cluster is selected.
    ix_selection_fn: selection function for each cluster.

  Returns:
    the sampled queryset as a dataframe.
  """
  work_queryset_df = queryset_df.drop_duplicates('pretokenized_string').copy()
  work_queryset_df['cluster'] = _cluster_queryset_df(encoder, work_queryset_df,
                                                     n_clusters)

  selected_ix = ix_selection_fn(work_queryset_df)
  return queryset_df.loc[selected_ix].copy()
