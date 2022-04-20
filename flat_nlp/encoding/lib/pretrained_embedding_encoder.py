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

"""Library for pretrained word embeddings encoder.

The inputs strings are expected to be pretokenized, tokens will be looked up
into the associated pretrained word embedding model.

"""

import functools
from typing import Callable, Union

import numpy as np
import tensorflow as tf

from flat_nlp.encoding.lib import embedding_util
from flat_nlp.encoding.lib import encoder_util

layers = tf.keras.layers


def _strip_regex(regex_to_strip: str,
                 inputs: encoder_util.StrTensor) -> encoder_util.StrTensor:
  """Strips a regex and lowercase a tensor of strings."""
  try:
    lowercase_inputs = tf.strings.lower(inputs)
  except tf.errors.InvalidArgumentError:
    return inputs
  return tf.strings.regex_replace(lowercase_inputs, regex_to_strip, '')


def _build_tf_encoder_standardization(
    normalization_mode: Union[None, str]
) -> Union[None, str, Callable[[encoder_util.StrTensor],
                               encoder_util.StrTensor]]:
  """Builds the standardization mode for tf Textvectorization."""

  if not bool(normalization_mode):
    return None

  if normalization_mode == 'strip_any_punc':
    return 'lower_and_strip_punctuation'

  if normalization_mode == 'strip_standalone_punc':
    # Unlike usual punctuation strip, we remove only standalone punctuation
    # For instance "The <cat> , is here !!" becomes: "The <cat> is here"
    return functools.partial(_strip_regex,
                             r'\B[!#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']+\B')

  raise NotImplementedError(
      'cannot find the requested normalization mode (%s)' % normalization_mode)


class PretrainedEmbeddingEncoder(
    layers.experimental.preprocessing.TextVectorization, encoder_util.Encoder):
  """Encoder to convert pretokenized queries as a sequence of word embeddings.

  The dimensions of the resulting tensor are:
    result[n_pretokenized_str, nd_embedding_space, n_token]

  Attributes:
    embedding_model: the embedding model to use.
    adapt_before_call: automatically include ad adapt(...) step before call(...)
    normalization_mode: text normalization to use.
    oov_strategy: out of vocabulary strategy to use.
    embedding_matrix: precomputed embedding matrix, index 0 and 1 are reserved
      for padding and oov tokens.
  """

  def __init__(self,
               embedding_model: embedding_util.EmbeddingModel,
               max_sentence_length: int,
               adapt_before_call: bool = True,
               *,
               normalization_mode: Union[None, str],
               oov_strategy: Union[None, str]):

    if max_sentence_length & (max_sentence_length - 1) != 0:
      raise ValueError('max_sentence_length must be a power of 2')

    super(PretrainedEmbeddingEncoder, self).__init__(
        standardize=_build_tf_encoder_standardization(normalization_mode),
        split='whitespace',
        output_mode='int',
        output_sequence_length=max_sentence_length)

    self.embedding_model = embedding_model
    self.normalization_mode = normalization_mode
    self.oov_strategy = oov_strategy
    self.adapt_before_call = adapt_before_call
    self.embedding_matrix = None

  @property
  def n_vector(self) -> int:
    return self._output_sequence_length

  @property
  def vector_size(self) -> int:
    return self.embedding_model.vector_size

  def adapt(self, inputs: encoder_util.StrTensor) -> None:
    """Adapts the encoder to a list of pretokenized strings."""
    super(PretrainedEmbeddingEncoder, self).adapt(inputs)

    input_vocabulary = self.get_vocabulary()[2:]
    input_embeddings = embedding_util.build_embedding_matrix(
        self.embedding_model, input_vocabulary, self.oov_strategy)
    reserved_embeddings = np.zeros((2, self.embedding_model.vector_size))
    self.embedding_matrix = np.vstack([reserved_embeddings, input_embeddings])

  def call(self, inputs: encoder_util.StrTensor) ->...:
    """Encodes a list of pretokenized strings."""
    if self.adapt_before_call:
      self.adapt(inputs)
    inputs_id = super(PretrainedEmbeddingEncoder, self).call(inputs)
    return tf.transpose(
        tf.nn.embedding_lookup(self.embedding_matrix, inputs_id),
        perm=[0, 2, 1])

  def adapt_and_call(self, inputs: encoder_util.StrTensor) ->...:
    """Adapts and encodes a list of pretokenized strings."""
    if not self.adapt_before_call:
      self.adapt(inputs)
    return self.call(inputs)

  @property
  def is_output_hc(self) -> bool:
    return False

  @property
  def is_output_strided(self) -> bool:
    return False
