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

"""Encoder Factory."""

from typing import Mapping

from flat_nlp.encoding.lib import encoder_adapter
from flat_nlp.encoding.lib import encoder_util
from flat_nlp.encoding.lib import pretrained_embedding_encoder
from flat_nlp.public import flat_pb2


def _load_base_encoder(config: flat_pb2.EncoderConfig) -> encoder_util.Encoder:
  """Loads a base encoder from a configuration spec."""
  if config.HasField('tf_model_checkpoint'):
    raise NotImplementedError('only pretrained_embedding_encoder is supported.')

  if config.HasField('servomatic_model'):
    raise NotImplementedError('only pretrained_embedding_encoder is supported.')

  if config.HasField('pretrained_embedding_encoder_config'):
    return _load_pretrained_embedding_encoder(
        config.pretrained_embedding_encoder_config)

  raise ValueError('encoder_config does not specify a base encoder model')


def load_encoder(config: flat_pb2.EncoderConfig) -> encoder_util.Encoder:
  """Loads an encoder from a configuration spec, this may include an adapter."""
  encoder = _load_base_encoder(config)

  if config.adapt_to_hc_output:
    encoder = encoder_adapter.HalfComplexEncoderAdapter(encoder)

  if config.adapt_to_strided_output:
    encoder = encoder_adapter.StridedOutputEncoderAdapter(encoder)

  return encoder


def load_available_encoder(
    multi_encoder_config: flat_pb2.MultiEncoderConfig
) -> Mapping[str, encoder_util.Encoder]:
  return {
      identifier: load_encoder(config) for identifier, config in
      multi_encoder_config.available_encoder_config.items()
  }
