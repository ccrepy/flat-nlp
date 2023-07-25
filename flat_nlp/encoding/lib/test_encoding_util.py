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

"""Test utility for encoding module."""

import gensim
import numpy as np

from flat_nlp.encoding.lib import pretrained_embedding_encoder


class MockEmbeddingModel():
  """Test class to mock and EmbeddingModel."""

  def __init__(self, vector_size, embedding_mapping):
    self.vector_size = vector_size
    self.embedding_mapping = embedding_mapping

  def __getitem__(self, token_list):
    return np.array([self.embedding_mapping[token] for token in token_list])

  def __contains__(self, token):
    return token in self.embedding_mapping


SIMPLE_EMBEDDING_MODEL = MockEmbeddingModel(3, {
    'query': (1, 1, 1),
    'a': (1, 0, 0),
    'b': (0, 2, 0),
    'c': (0, 0, 3),
})

SIMPLE_TEXT8_ENCODER = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
    gensim.models.KeyedVectors.load_word2vec_format(
        'third_party/flat_nlp/encoding/lib/testdata/text8.5.vec',
        binary=False,
        encoding='utf8'),
    8,
    normalization_mode=None,
    oov_strategy=None)
