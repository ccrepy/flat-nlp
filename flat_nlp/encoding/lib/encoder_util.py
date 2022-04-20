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

"""Utility library for encoder."""

from typing import Any, Callable

import numpy as np
import typing_extensions

from flat_nlp.lib import np_distance
from flat_nlp.lib.python import fftw_distance

# TODO(ccrepy): add a better type hinting
StrTensor = Any


class Encoder(typing_extensions.Protocol):
  """Interface which defines an encoder.

  The contract is:
    * my_model.n_vector: max length of one sequence
    * my_model.vector_size: size of the vector space
    * my_model.call(inputs): encoded list of inputs
    * my_model.is_output_hc: output is in halfcomplex format
    * my_model.is_output_strided: output is strided
  """

  n_vector: int
  vector_size: int

  def call(self, inputs: StrTensor) ->...:
    ...

  is_output_hc: bool
  is_output_strided: bool


def build_distance_fn(
    encoder: Encoder) -> Callable[[np.ndarray, np.ndarray], float]:
  """Builds the correct flat distance for a givne encoder."""
  if not encoder.is_output_hc and not encoder.is_output_strided:
    return np_distance.flat_distance

  if encoder.is_output_hc and not encoder.is_output_strided:
    return np_distance.hc_flat_distance

  if not encoder.is_output_hc and encoder.is_output_strided:
    return fftw_distance.StridedFlatDistanceFn(encoder.vector_size,
                                               encoder.n_vector)

  if encoder.is_output_hc and encoder.is_output_strided:
    return fftw_distance.StridedHalfComplexFlatDistanceFn(
        encoder.vector_size, encoder.n_vector)

  raise NotImplementedError
