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

"""Tests for kde_util."""

import numpy as np

from sklearn import neighbors

from flat_nlp.ranking.lib import kde_util
from absl.testing import absltest


class KdeUtilTest(absltest.TestCase):

  def test_aggregate_kde_model(self):
    source_kde = neighbors.KernelDensity(leaf_size=2).fit(np.eye(8))
    agg_kde = kde_util.aggregate_kde_model(source_kde)

    self.assertLen(source_kde.tree_.data, 8)
    self.assertLen(agg_kde.tree_.data, 2)

  def test_normalize_kde_pdf(self):
    log_proba = np.log([0.2, 0.4, 0.1, 0.45])

    self.assertSequenceAlmostEqual(
        kde_util.normalize_kde_pdf(log_proba, 0.4).tolist(),
        [0.5, 1., 0.25, 1.])

  def test_naive_search_maximum(self):
    kde_model = neighbors.KernelDensity().fit([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4],
    ])
    self.assertAlmostEqual(
        kde_util.naive_search_maximum(kde_model), 0.006896340330)

  def test_newton_search_maximum(self):
    kde_model = neighbors.KernelDensity().fit([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4],
    ])
    self.assertAlmostEqual(
        kde_util.newton_search_maximum(kde_model), 0.007076478018)


if __name__ == '__main__':
  absltest.main()
