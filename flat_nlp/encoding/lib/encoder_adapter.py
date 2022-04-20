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

"""Adapters for encoders.

Adapters are expected to be idempotent and not degrade their input informations,
They are used as a compatibility layer between Flat NLP components or other
systems.

For instance, some Sklearn models do not support tensor shape but only matrix,
one solution is to stride the 3d array into a 2d array.

"""

from flat_nlp.encoding.lib import encoder_util

from flat_nlp.lib import tf_signal_util


class BaseEncoderAdapter(encoder_util.Encoder):
  """Wrapper encoder base class."""

  def __init__(self, base_encoder: encoder_util.Encoder):
    self.base_encoder = base_encoder

  @property
  def n_vector(self) -> int:
    return self.base_encoder.n_vector

  @property
  def vector_size(self) -> int:
    return self.base_encoder.vector_size

  def call(self, inputs: encoder_util.StrTensor) ->...:
    raise NotImplementedError()

  @property
  def is_output_hc(self) -> bool:
    return self.base_encoder.is_output_hc

  @property
  def is_output_strided(self) -> bool:
    return self.base_encoder.is_output_strided


class HalfComplexEncoderAdapter(BaseEncoderAdapter):
  """HalfComplex encoder adapter."""

  @property
  def is_output_hc(self) -> bool:
    return True

  def call(self, inputs: encoder_util.StrTensor) ->...:
    if self.base_encoder.is_output_strided:
      raise ValueError(
          'HalfComplex encoding is not supported for strided inputs.')

    base_output = self.base_encoder.call(inputs)
    if not self.base_encoder.is_output_hc:
      return tf_signal_util.hc(base_output)
    return base_output


class StridedOutputEncoderAdapter(BaseEncoderAdapter):
  """Strided output encoder adapter."""

  @property
  def is_output_strided(self) -> bool:
    return True

  def call(self, inputs: encoder_util.StrTensor) ->...:
    base_output = self.base_encoder.call(inputs)
    if not self.base_encoder.is_output_strided:
      return tf_signal_util.stride_signal(base_output)
    return base_output
