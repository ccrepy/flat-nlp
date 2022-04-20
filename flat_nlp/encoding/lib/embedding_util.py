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

"""Utility library for word embeddings."""

import itertools
from typing import Sequence, Union

import numpy as np
import typing_extensions


class EmbeddingModel(typing_extensions.Protocol):
  """Interface which defines a pretrained word embedding model.

  The contract is:
    * my_model.vector_size: size of the vector space
    * my_model[list_of_token]: matrix corresponding to the tokens requested
    * token in my_model: ability of the model to produce embedding for
                           one requested token

  optionally, a `contains_bulk(...)` method can be implemented, it is useful for
    remote models.
  """
  vector_size: int

  def __getitem__(self, key_list: Sequence[str]) -> np.ndarray:
    ...

  def __contains__(self, key: str) -> bool:
    ...


def _has_embedding(embedding_model: EmbeddingModel,
                   token_list: Sequence[str]) -> Sequence[bool]:
  """Checks if an embedding model has a representation for a list of words."""
  if hasattr(embedding_model, 'contains_bulk'):
    return embedding_model.contains_bulk(token_list)
  else:
    return [token in embedding_model for token in token_list]


def build_embedding_matrix(embedding_model: EmbeddingModel,
                           requested_token_list: Sequence[str],
                           oov_strategy: Union[None, str]) -> np.ndarray:
  """Extracts embeddings representation as a matrix.

  This abstraction standardize the way an PretrainedEmbeddingEncoder query
  tokens.
  The extra logic is useful when the embedding model is remote (to minimize the
    calls)

  Another benefit is having a clear strategy for oov, we propose 3 strategies:

    * None: an error will be raised
    * ignore: the corresponding embedding will be replaced by 0.
    * decompose: average the embeddings of each chars of the token,
                  This is especially useful for CJK languages when tokenization
                  is not consistent.

  Arguments:
    embedding_model: embedding model to use, it can be a remote model
    requested_token_list: list of token to fetch.
    oov_strategy: fallback strategy for out of vocabulary tokens

  Returns:
    the numpy array corresponding to the requested tokens.
  """

  embedding_matrix = np.zeros(
      (len(requested_token_list), embedding_model.vector_size))

  if not bool(oov_strategy):
    return embedding_model[requested_token_list]

  if oov_strategy == 'ignore':
    try:
      return build_embedding_matrix(embedding_model, requested_token_list, None)
    except KeyError:
      availalble_token_mask = _has_embedding(embedding_model,
                                             requested_token_list)

      # Fetch the token embeddings
      available_token_list = list(
          itertools.compress(requested_token_list, availalble_token_mask))

      if available_token_list:
        available_embedding_mapping = dict(
            zip(available_token_list, embedding_model[available_token_list]))
      else:
        available_embedding_mapping = {}

      # Fill the matrix
      for i, token in enumerate(requested_token_list):
        if token in available_embedding_mapping:
          embedding_matrix[i] = available_embedding_mapping[token]

    return np.ascontiguousarray(embedding_matrix)

  if oov_strategy == 'decompose':
    try:
      return build_embedding_matrix(embedding_model, requested_token_list, None)
    except KeyError:
      availalble_token_mask = _has_embedding(embedding_model,
                                             requested_token_list)

      # Fetch the char embeddings
      missing_token_list = itertools.compress(
          requested_token_list, np.logical_not(availalble_token_mask))
      decomposed_missing_token = list(set(itertools.chain(*missing_token_list)))
      char_embedding_mapping = dict(
          zip(
              decomposed_missing_token,
              build_embedding_matrix(embedding_model, decomposed_missing_token,
                                     'ignore')))

      # Fetch the token embeddings
      available_token_list = list(
          itertools.compress(requested_token_list, availalble_token_mask))
      available_embedding_mapping = dict(
          zip(
              available_token_list,
              build_embedding_matrix(embedding_model, available_token_list,
                                     None)))

      # Fill the matrix
      for i, token in enumerate(requested_token_list):
        if token in available_embedding_mapping:
          embedding_matrix[i] = available_embedding_mapping[token]
        else:
          # `char_embedding_mapping` is always populated possibly only with 0.
          #   due to 'ignore' mode used above.
          embedding_matrix[i] = np.average(
              [char_embedding_mapping[c] for c in list(token)], axis=0)

      return np.ascontiguousarray(embedding_matrix)

  raise NotImplementedError('cannot find the requested oov strategy (%s)' %
                            oov_strategy)
